#!/usr/bin/env bash
# GPU window "v31" (~65 min of production outage planned, cap WINDOW_S, default 4500 s = 75 min): Unsloth Dynamic v3.1
# UD-IQ4_XS (Hub revision d9620119 of 2026-11-04, 4 shards, the n-gram table in shard 4; pack built 2026-11-09 by
# tools/iq_pack.py --compat-bf16 of Strata-212p) against production's v3.0 UD-IQ4_XS, everything else equal: production's
# engine, server checkout, engine env, server env and args.  v3.1 moves bits around: 345 of 1224 tensors have another type
# (attention / GDN / shared-expert projections Q8_0 -> Q6_K, token_embd Q8_0 -> IQ4_XS, gate/up IQ3_S -> IQ3_XXS / IQ4_XS /
# IQ2_S, downs IQ4_NL -> Q5_0 / Q2_0), experts 59.52 -> 60.38 GB, the same tokenizer and chat template.
#   A   production.json as is (v3.0)
#   V   A with the v3.1 files: --pack, --native (GGUF shard 1), tokenizer, vision.model, model_name ...-v31, its own log.
#       Production's layer split (12,25,37) puts ~+0.8 GiB on CUDA1 and ~+1.5 GiB on CUDA2 (CUDA0 -1.5, CUDA3 -0.5); by the
#       engine's room lines of 2026-11-08 it fits (CUDA1 ~0.15 GiB spare).  When V's engine log does not say "every expert is
#       in the GPU caches", V runs once more as V2 with layer_split $V_SPLIT2 (13,25,37: CUDA1 ~1.3 GiB spare by the same
#       count) and V2 replaces V in the comparison (INFO: another split)
# Per arm one fresh server on 127.0.0.1:8091 (strata-bench.scope, MemoryMax $MEMMAX, empty tier dir, its own engine log):
#   up     1 warm-up request; the engine's expert placement (layer split lines, "every expert is in the GPU caches"), VRAM
#   smoke  v31_smoke.py: the 6 request shapes Genesis sends via G3 (thinking off, json_object, streamed tool call with
#          reasoning_budget + chat_template_kwargs, the tool result turn, xhigh with reasoning_budget 512, an image)  ~0.5 min
#   short  bench_concurrency 1/2/4/8/16 users x 2, 512 tokens, ~100-token prompts, greedy (window-212's short)   ~3 min
#   long   bench_concurrency 16 users x 2, 512 tokens, ~1000-token prompts, greedy (window-212's long)            ~1.5 min
#   mt     bench_multiturn 16 users x 3 turns --doc 8000 greedy (window-212's mt)                                  ~1.5 min
#   cold   1 request with a ~128K-token prompt, 32 tokens: TTFT = one cold read                                   ~0.5 min
#   eval   eval/quick_eval.py: MMLU-Pro 240 + GSM8K 150, greedy, thinking off, 8 workers (answers-<tag>-<arm>.jsonl)  ~12-15 min
#   + engine start ~2-3 min.  Order V, A (full), then while time is left r2 = A, V again (short + long only): VAAV
# The summary (after production is back): per arm short agg tok/s 1/4/16 + per-user decode, long agg + TTFT, mt wall +
# follow-up TTFT, cold 128K TTFT, MMLU-Pro / GSM8K with the paired count (questions only one arm got right, exact McNemar
# p), smoke, VRAM, expert fit, peak RSS.  Nothing is deployed: a switch to v3.1 needs the owner's "deploy ja".
# Isolation and restore as window-borrow.sh: test servers only on 127.0.0.1:8091, tier dirs under
# /media/raid0/strata-kvspill-bench/v31 (capped at TIER_CAP_GB, removed at the end), results in results/v31-<MMDD-HHMM>; production.json, the unit
# and its drop-in are only read.  Production is stopped only once it reports loaded and started again on EVERY exit path:
# the EXIT trap, v31-guard.service (waits for this PID: kills the step, stops strata-bench.scope, starts strata.service) on a
# SIGKILL / OOM kill of this script, and the independent backstop v31-restore.timer (WINDOW_S + 15 min after the stop).
#   ./window-v31.sh --dry-run          # prep checks + configs + the plan; production untouched
#   ./window-v31.sh                    # the window (GPU window: coordinate with the bench session first)
#   ./window-v31.sh --summary DIR      # the summary table again from results/v31-...
#   env: WINDOW_S=4500 V_SPLIT2=13,25,37 R2=1 EVAL=1 MEMMAX=224G TIER_CAP_GB=8
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs
URL=http://127.0.0.1:8090; TPORT=8091; PY=$R/Strata/.venv/bin/python
SRV=$R/Strata-143-prod; SRV_REV=a099d295                    # production's server checkout (strata.service WorkingDirectory)
V31_DIR=/media/raid0/Models/Strata-GGUF/unsloth-UD-IQ4_XS-v31
V31_GGUF=$V31_DIR/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
V31_PACK=$R/Strata-data/packs/unsloth-ud-iq4_xs-v31
V31_NAME=qwen3.8-flash-next-unsloth-ud-iq4_xs-v31
declare -A V31_BYTES=([00001]=10946618 [00002]=49604491008 [00003]=15256783680 [00004]=28800138432)   # setup.py UNSLOTH_V31_SHARDS
DRY=0; SUMMARY_DIR=
while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) DRY=1; shift ;;
        --summary) SUMMARY_DIR=$(readlink -f "${2:?--summary needs results/v31-... }"); shift 2 ;;
        -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument $1 (--dry-run, --summary DIR)"; exit 2 ;;
    esac
done
WINDOW_S=${WINDOW_S:-4500}; RESERVE=480                     # outage cap (stop -> restart), room kept for the restart
ARM_EST=${ARM_EST:-1500}; ARM_CAP=${ARM_CAP:-2700}          # a full arm: expected / hard cap (s)
V_CAP=${V_CAP:-2100}                                        # V's / V2's cap: A (est. ARM_EST) still fits after a slow V
R2_EST=${R2_EST:-480}; R2_CAP=${R2_CAP:-1200}               # an r2 arm (start + short + long)
V_SPLIT2=${V_SPLIT2:-13,25,37}; R2=${R2:-1}; EVAL=${EVAL:-1}
TIER_CAP_GB=${TIER_CAP_GB:-8}                               # the test servers' disk tier cap (production: 500)
MEMMAX=${MEMMAX:-224G}
TAG=v31w$(date +%m%d%H%M)                                   # label prefix (eval_results.jsonl and the sweep files are shared)
OUT=${SUMMARY_DIR:-$L/results/v31-$(date +%m%d-%H%M)}; [ "$DRY" = 1 ] && OUT=$L/results/v31-dry
CF=$OUT/cfg; TIER=/media/raid0/strata-kvspill-bench/v31
RUNIT=v31-restore; GUNIT=v31-guard
for v in $(compgen -e | grep '^STRATA_'); do unset "$v"; done   # nothing of ours from the calling shell
trap 'echo "window aborted before production was stopped"; exit 1' INT TERM HUP

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
vram() { nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tr '\n' ';'; }
# kills only engines of $R/Strata* (production is stopped when this runs)
free_gpus() { local i p e; for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do e=$(readlink -f "/proc/$p/exe" 2>/dev/null)
        case "$e" in "$R"/Strata*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
prod_loaded() { curl -s -m 3 "$URL/health" | grep -q '"loaded": *true'; }
start_prod() { log "starting production as before (production.json -> $(readlink "$C/production.json"))"
    systemctl --user start strata.service; local t0; t0=$(date +%s)
    until prod_loaded; do
        systemctl --user is-active -q strata.service || { log "strata.service died during the start FAIL"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "production start timeout FAIL"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s: $(curl -s -m 5 "$URL/health" | cut -c1-200)"; }
bench_down() { systemctl --user stop strata-bench.scope 2>/dev/null; systemctl --user reset-failed strata-bench.scope 2>/dev/null
    free_gpus || log "GPUs busy after $1"; }
disarm_restore() { systemctl --user stop "$RUNIT.timer" 2>/dev/null
    systemctl --user reset-failed "$RUNIT.timer" "$RUNIT.service" 2>/dev/null; }
arm_restore() {   # an independent backstop outside this process: starts production at $1 even after a SIGKILL
    disarm_restore
    systemd-run --user --quiet --unit="$RUNIT" --on-calendar="$(date -d "@$1" '+%F %T')" \
        --timer-property=AccuracySec=1s /bin/sh -c 'systemctl --user stop strata-bench.scope; systemctl --user start strata.service' \
        && systemctl --user is-active -q "$RUNIT.timer"; }
disarm_guard() { systemctl --user stop "$GUNIT.service" 2>/dev/null; systemctl --user reset-failed "$GUNIT.service" 2>/dev/null; }
arm_guard() {   # restores production within seconds when this script dies without its EXIT trap (SIGKILL, OOM kill)
    disarm_guard
    systemd-run --user --quiet --collect --unit="$GUNIT" /bin/sh -c 'tail --pid="$1" -f /dev/null
        g=$(cat "$2" 2>/dev/null); [ -n "$g" ] && [ "$(ps -o comm= -p "$g" 2>/dev/null)" = timeout ] && /bin/kill -s KILL -- "-$g"
        systemctl --user stop strata-bench.scope; systemctl --user start strata.service' sh "$$" "$OUT/step.pgid" || return 1
    sleep 2; systemctl --user is-active -q "$GUNIT.service"; }

STOPPED=0; STEP_PID=; WD=; BACK_DONE=0; T0=$(date +%s)
back() { [ "$BACK_DONE" = 1 ] && return 0; BACK_DONE=1; trap '' INT TERM HUP
    [ -n "$WD" ] && { pkill -P "$WD" 2>/dev/null; kill "$WD" 2>/dev/null; }
    if [ -n "$STEP_PID" ] && kill -0 "$STEP_PID" 2>/dev/null; then
        kill -INT -- -"$STEP_PID" 2>/dev/null || kill -INT "$STEP_PID" 2>/dev/null
        for _ in $(seq 30); do kill -0 "$STEP_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL -- -"$STEP_PID" 2>/dev/null; fi
    [ "$STOPPED" = 1 ] || return 0
    bench_down end
    if start_prod || { log "production start failed: one more try"; systemctl --user restart strata.service; start_prod; }; then
        disarm_restore; disarm_guard; log "$RUNIT.timer and $GUNIT.service removed"
    else log "PRODUCTION DID NOT COME BACK - check strata.service now ($RUNIT.timer and $GUNIT.service left armed)"; fi
    local P PE NOWCFG; P=$(systemctl --user show -p MainPID --value strata.service)
    PE=$(for c in $(pgrep -P "$P" 2>/dev/null); do readlink "/proc/$c/exe"; done | tr '\n' ' ')
    NOWCFG=$(sha256sum "$(readlink -f "$C/production.json")" | cut -c1-16)
    log "    [prod] ==> production.json sha256 $NOWCFG (at the start $PRODCFG_SHA), engine running: ${PE:-?}: $([ "$NOWCFG" = "$PRODCFG_SHA" ] && echo "$PE" | grep -q "$(basename "$PEXE")" && echo PASS || echo FAIL)"
    rm -rf "$TIER"
    log "window done after $(( ($(date +%s) - T0) / 60 )) min of production outage"; }
final() { back; [ "$STOPPED" = 1 ] && { summary "$OUT" 2>&1 | tee "$OUT/summary.txt" | tee -a "$OUT/window.log" >/dev/null
    cat "$OUT/summary.txt"; grep -hE 'PASS|FAIL|SKIP|WARN' "$OUT/window.log" | tail -60; }; }
on_sig() { log "window aborted: ${1:-signal}"; exit 130; }    # -> EXIT trap -> final (back + summary)
left() { echo $(( T0 + WINDOW_S - RESERVE - $(date +%s) )); }
# step <name> <expected s> <cap s> <cmd...>: skipped when the expected time no longer fits the window; the cap is cut
# to what is left; the output goes to $OUT/<name>.txt and its '==>' verdict lines into window.log
step() { local name=$1 est=$2 limit=$3; shift 3; local l rc; l=$(left)
    if [ "$l" -lt "$est" ]; then log "step $name: SKIP (needs ~${est} s, ${l} s left in the window)"; echo skip > "$OUT/rc_$name"; return 1; fi
    [ "$limit" -gt "$l" ] && limit=$l
    log "step $name: start (expected ~${est} s, cap ${limit} s); VRAM (index, used, free): $(vram)"
    timeout -s INT -k 120 "$limit" bash -c '"$@"' "$name" "$@" > "$OUT/$name.txt" 2>&1 < /dev/null &
    STEP_PID=$!; echo "$STEP_PID" > "$OUT/step.pgid"; wait "$STEP_PID"; rc=$?; STEP_PID=; : > "$OUT/step.pgid"
    echo $rc > "$OUT/rc_$name"
    [ $rc -eq 124 ] && log "step $name: TIMEOUT after ${limit} s FAIL"
    grep -aE '^==>' "$OUT/$name.txt" | cut -c1-700 | sed "s/^/    [$name] /" | tee -a "$OUT/window.log" >/dev/null
    log "step $name: exit $rc"
    bench_down "$name"; return $rc; }

# ---- helpers the arm runs (exported: step runs them under timeout through bash -c) ------------------------------------
tup() { curl -s -m 3 "http://127.0.0.1:$TPORT/health" | grep -q '"loaded": *true'; }
# runcfg <NAME> <run>: a copy of $CF/NAME.json for one server start: its own engine log and a fresh, empty tier dir
# capped at $TIER_CAP_GB (the disk tier only takes RAM-cache evictions, which these benches do not cause; the cap bounds
# the disk all the same while the disk is near its floor - every arm gets it)
runcfg() { local d=$TIER/$2 f=$CF/run-$2.json; rm -rf "$d"; mkdir -p "$d"; rm -f "$OUT/engine-$2.log"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; c['env']['STRATA_CONV_TIER_CAP_GB']=sys.argv[5]; c['log']=sys.argv[3]; json.dump(c,open(sys.argv[4],'w'),indent=1)" \
        "$CF/$1.json" "$d" "$OUT/engine-$2.log" "$f" "$TIER_CAP_GB" && echo "$f"; }
# srv <cmd...>: the command with production's server checkout and server env, on 127.0.0.1:$TPORT
srv() { ( cd "$L" && env SERVER_DIR="$SRV" $SRVENV PORT=$TPORT BIND=127.0.0.1 MEMMAX=$MEMMAX "$@" ); }
up() { srv env KEEP=1 ./run_strata_variant.sh "$1" "$2" 1 --rounds 1 --max-tok 16 --prompt-tok 100 --temperature 0 2>&1 | tail -3
    tup || { echo "==> $1: the test server did not come up FAIL"; systemctl --user stop strata-bench.scope; return 1; }; }
rss_peak() { local cg; cg=$(systemctl --user show -p ControlGroup --value strata-bench.scope 2>/dev/null)
    if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ]; then
        awk '{printf "%.2f", $1 / 1073741824}' "/sys/fs/cgroup$cg/memory.peak"; else echo "?"; fi; }
# arm_run <CFG> <run label> <full|r2> [fitgate]: one server start and the arm's steps (r2: short + long only); fitgate: the
# arm ends after the warm-up when its engine did not put every expert into the GPU caches (V: V2 runs instead)
arm_run() { local a=$1 lab=$2 mode=$3 gate=${4:-} cfg rc=0 u=http://127.0.0.1:$TPORT elog model B
    cfg=$(runcfg "$a" "$lab") || return 1; elog=$OUT/engine-$lab.log
    model=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_name'])" "$cfg")
    up "$TAG-$lab-up" "$cfg" || return 1
    grep -aE 'layer split: CUDA[0-9] runs|room for experts|expert cache auto|every expert is in the GPU caches|native pack:' "$elog" > "$OUT/$lab-placement.txt"
    if grep -aq 'every expert is in the GPU caches' "$elog"; then
        echo "==> $lab experts: $(grep -a -m1 -oE 'every expert is in the GPU caches \([0-9]+ of [0-9]+\)' "$elog") INFO"
    else echo nofit > "$OUT/fit-$lab"
        echo "==> $lab experts: NOT every expert in the GPU caches: $(grep -aE 'layer split: CUDA[0-9] runs' "$elog" | sed 's/.*runs //' | tr '\n' ';' | cut -c1-400) WARN"; fi
    echo "$(vram)" > "$OUT/vram-$lab.txt"; echo "==> $lab VRAM after the warm-up (index, used, free): $(cat "$OUT/vram-$lab.txt") INFO"
    if [ -n "$gate" ] && [ -e "$OUT/fit-$lab" ]; then
        echo "==> $lab: ends after the warm-up (not every expert in the GPU caches: the next arm takes another split) INFO"
        systemctl --user stop strata-bench.scope; return 3; fi
    if [ "$mode" = full ]; then
        timeout -s INT 600 "$PY" -I "$L/v31_smoke.py" --url "$u" --model "$model" --label "$TAG-$lab" --out "$OUT/smoke.jsonl" \
            > "$OUT/$lab-smoke.txt" 2>&1 || rc=1
        grep -a '^==>' "$OUT/$lab-smoke.txt" || echo "==> smoke $lab: no verdict (see $OUT/$lab-smoke.txt) FAIL"; fi
    B="$PY -I -B $L/bench_concurrency.py --url $u --temperature 0"
    if tup; then timeout -s INT 900 $B --label "$TAG-$lab-short" --levels 1,2,4,8,16 --rounds 2 --max-tok 512 --prompt-tok 100 \
            --out "$OUT/sweep.jsonl" > "$OUT/$lab-short.txt" 2>&1 || rc=1
        echo "==> $lab short: $(grep -aE '^ *16 ' "$OUT/$lab-short.txt" | head -1 | tr -s ' ' | cut -c1-160) INFO"
    else echo "==> $lab: the server is gone before short FAIL"; return 1; fi
    if tup; then timeout -s INT 600 $B --label "$TAG-$lab-long" --levels 16 --rounds 2 --max-tok 512 --warmup 0 \
            --out "$OUT/sweep.jsonl" > "$OUT/$lab-long.txt" 2>&1 || rc=1
    else echo "==> $lab: the server is gone after short FAIL"; return 1; fi
    if [ "$mode" = full ]; then
        if tup; then timeout -s INT 900 "$PY" -I -B "$L/bench_multiturn.py" --url "$u" --users 16 --turns 3 --doc 8000 --temperature 0 \
                --label "$TAG-$lab-mt" --out "$OUT/multiturn.jsonl" > "$OUT/$lab-mt.txt" 2>&1 || rc=1
            echo "==> $lab multiturn: $(tail -1 "$OUT/$lab-mt.txt" | cut -c1-200) INFO"
        else echo "==> $lab: the server is gone after long FAIL"; return 1; fi
        if tup; then timeout -s INT 600 $B --label "$TAG-$lab-cold" --levels 1 --rounds 1 --max-tok 32 --prompt-tok 128000 --warmup 0 \
                --out "$OUT/sweep.jsonl" > "$OUT/$lab-cold.txt" 2>&1 || rc=1
        else echo "==> $lab: the server is gone after multiturn FAIL"; return 1; fi
        if [ "$EVAL" = 1 ] && tup; then
            timeout -s INT 1800 "$PY" -I -B "$L/eval/quick_eval.py" --url "$u" --model "$model" --label "$TAG-$lab" --workers 8 \
                --max-tokens 2048 > "$OUT/$lab-eval.txt" 2>&1 || rc=1
            echo "==> $lab eval: $(tail -1 "$OUT/$lab-eval.txt" | cut -c1-200) INFO"; fi
        tup || { echo "==> $lab: the server is gone after the eval FAIL"; rc=1; }
    fi
    echo "$(rss_peak)" > "$OUT/rss-$lab.txt"; echo "==> rss $lab: peak $(cat "$OUT/rss-$lab.txt") GiB (memory.peak) INFO"
    grep -acE 'CUDA error|out of memory|cudaMalloc failed' "$elog" | sed "s/^/==> $lab engine errors (CUDA error \/ out of memory): /;s/$/ INFO/"
    sleep 2; systemctl --user stop strata-bench.scope; return $rc; }
export -f tup runcfg srv up rss_peak arm_run vram
export R L C CF OUT TIER TPORT PY SRV MEMMAX TAG EVAL TIER_CAP_GB

summary() {   # summary <OUT>: the comparison table (python: the sweep / multiturn rows, the eval answers, the smoke lines)
    python3 - "$1" "$L/eval" <<'PYEOF'
import json, math, os, re, sys, glob
out, evdir = sys.argv[1:3]
def rows(f):
    try:
        return [json.loads(l) for l in open(os.path.join(out, f)) if l.strip()]
    except FileNotFoundError:
        return []
sw, mt = rows("sweep.jsonl"), rows("multiturn.jsonl")
tag = None
for r in sw + mt:
    m = re.match(r"(v31w\d+)-", r["label"]); tag = tag or (m and m.group(1))
labs = []
for r in sw:
    lab = r["label"][len(tag) + 1:].rsplit("-", 1)[0] if tag else r["label"]
    if lab not in labs: labs.append(lab)
def get(lab, kind, users):
    for r in sw:
        if r["label"] == f"{tag}-{lab}-{kind}" and r["users"] == users: return r
def ev(lab):
    p = os.path.join(evdir, f"answers-{tag}-{lab}.jsonl")
    if not os.path.exists(p): return None
    return {(x["set"], x["i"]): x["ok"] for x in map(json.loads, open(p))}
print(f"window {tag}: arms {' '.join(labs)} (r2 = second start, short + long only)")
print(f"{'arm':6} {'short agg 1/4/16 tok/s':>24} {'dec p50 @1':>10} {'long agg16':>10} {'long ttft p50/p95':>17} "
      f"{'mt wall/fu p50':>15} {'cold128K ttft':>13} {'MMLU-Pro':>8} {'GSM8K':>6} {'smoke':>6} {'fit':>4} {'rss GiB':>7}")
smoke = {}
for l in rows("smoke.jsonl"):
    smoke[l["label"][len(tag) + 1:]] = "PASS" if all(l["ok"].values()) else "FAIL:" + "".join(k for k, v in l["ok"].items() if not v)
for lab in labs:
    s = [get(lab, "short", n) for n in (1, 4, 16)]; lg = get(lab, "long", 16); cd = get(lab, "cold", 1)
    m = next((r for r in mt if r["label"] == f"{tag}-{lab}-mt"), None)
    e = None; ep = os.path.join(evdir, "eval_results.jsonl")
    if os.path.exists(ep):
        e = next((x for x in map(json.loads, open(ep)) if x.get("label") == f"{tag}-{lab}"), None)
    rss = open(os.path.join(out, f"rss-{lab}.txt")).read().strip() if os.path.exists(os.path.join(out, f"rss-{lab}.txt")) else "?"
    fit = "no" if os.path.exists(os.path.join(out, f"fit-{lab}")) else "yes"
    print(f"{lab:6} {'/'.join(str(x['agg_out_tps']) if x else '-' for x in s):>24} {str(s[0]['dec_tps_p50']) if s[0] else '-':>10} "
          f"{str(lg['agg_out_tps']) if lg else '-':>10} {(str(lg['ttft_p50']) + '/' + str(lg['ttft_p95'])) if lg else '-':>17} "
          f"{(str(m['wall_s']) + '/' + str(m.get('follow_ttft_p50'))) if m else '-':>15} "
          f"{(str(cd['ttft_p50']) + ' s') if cd else '-':>13} {str(e['mmlu_pro']) if e else '-':>8} {str(e['gsm8k']) if e else '-':>6} "
          f"{smoke.get(lab, '-'):>6} {fit:>4} {rss:>7}")
def pair(x, y, name):
    a, b = ev(x), ev(y)
    if not a or not b: return
    for st in ("mmlu_pro", "gsm8k"):
        ks = [k for k in a if k[0] == st and k in b]
        only_a = sum(1 for k in ks if a[k] and not b[k]); only_b = sum(1 for k in ks if b[k] and not a[k])
        n = only_a + only_b
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
        print(f"  {name} {st}: {len(ks)} questions, only {x} right {only_a}, only {y} right {only_b}, both right "
              f"{sum(1 for k in ks if a[k] and b[k])}; exact McNemar p = {p:.3f}")
v = "V2" if "V2" in labs else "V"
pair("A", v, f"paired A vs {v}:")
for kind, users in (("short", 1), ("short", 4), ("short", 16), ("long", 16)):
    pairs = [(get(x, kind, users), get(y, kind, users)) for x, y in ((v, "A"), (v + "r2" if v + "r2" in labs else "Vr2", "Ar2"))]
    d = [f"{100 * (vv['agg_out_tps'] / aa['agg_out_tps'] - 1):+.1f} %" for vv, aa in pairs if vv and aa]
    if d: print(f"  {kind} {users} users: {v} vs A agg tok/s {' / '.join(d)} (run 1 / r2)")
PYEOF
}
if [ -n "$SUMMARY_DIR" ]; then summary "$SUMMARY_DIR"; exit 0; fi

# ---- prep (no GPU, production untouched) ------------------------------------------------------------------------------
PRODSRC=$(readlink -f "$C/production.json"); [ -r "$PRODSRC" ] || { echo "cannot read $C/production.json"; exit 1; }
cfg_exe() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" "$1"; }
PEXE=$(cfg_exe "$PRODSRC"); [ -x "$PEXE" ] || { echo "production's exe $PEXE missing"; exit 1; }
for i in 00001 00002 00003 00004; do f=$V31_DIR/Qwen3.8-Flash-Next-UD-IQ4_XS-$i-of-00004.gguf
    [ "$(stat -c %s "$f" 2>/dev/null)" = "${V31_BYTES[$i]}" ] || { echo "$f missing or not ${V31_BYTES[$i]} bytes"; exit 1; }
    [ -s "$f.done" ] || { echo "$f has no .done (sha256 not verified)"; exit 1; }; done
for f in native_experts.txt index.txt dense.bin tokenizer/vocab.json tokenizer/chat_template.jinja; do
    [ -s "$V31_PACK/$f" ] || { echo "the v3.1 pack lacks $f ($V31_PACK)"; exit 1; }; done
grep -q 'Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf' "$V31_PACK/native_experts.txt" || { echo "$V31_PACK is not the v3.1 set's pack"; exit 1; }
[ "$(git -C "$SRV" rev-parse --short=8 HEAD)" = "$SRV_REV" ] || { echo "$SRV is not at $SRV_REV (production's server)"; exit 1; }
git -C "$SRV" diff --quiet HEAD -- serve || { echo "$SRV has uncommitted changes under serve/"; exit 1; }
[ "$(systemctl --user show strata.service -p WorkingDirectory --value)" = "$SRV" ] || { echo "strata.service does not run from $SRV"; exit 1; }
for f in "$L/bench_concurrency.py" "$L/bench_multiturn.py" "$L/v31_smoke.py" "$L/eval/quick_eval.py"; do
    "$PY" -I "$f" --help > /dev/null || { echo "$f does not load"; exit 1; }; done
[ -s "$L/eval/data/sets.json" ] || { echo "eval/data/sets.json missing (quick_eval would fetch the questions during the window)"; exit 1; }
[ "$(df --output=avail -BG "$L" | tail -1 | tr -dc 0-9)" -ge 8 ] || { echo "less than 8 GB free under $L"; exit 1; }
curl -s -m 2 "http://127.0.0.1:$TPORT/health" > /dev/null && { echo "port $TPORT already serves something"; exit 1; }
systemctl --user is-active -q strata-bench.scope && { echo "strata-bench.scope is active: another bench is running"; exit 1; }
for u in "$RUNIT.timer" "$GUNIT.service" borrow-restore.timer borrow-guard.service; do
    systemctl --user is-active -q "$u" && { echo "$u is active: another window?"; exit 1; }; done
others=$(pgrep -af '(window-[A-Za-z0-9_-]+|night-[A-Za-z0-9_-]+|run_strata_variant|soak|ab_quick)\.sh' 2>/dev/null | grep -v -e "^$$ " -e pgrep | grep -v 'window-v31.sh')
[ -n "$others" ] && { echo "another window / bench script runs: $others"; exit 1; }
if [ "$DRY" != 1 ]; then
    systemctl --user is-active -q strata.service || { echo "strata.service is not active: start it first"; exit 1; }
    for p in $(gpu_apps); do [ "$(readlink -f "/proc/$p/exe" 2>/dev/null)" = "$(readlink -f "$PEXE")" ] \
        || { echo "GPU process $p ($(readlink -f "/proc/$p/exe" 2>/dev/null)) is not production's engine: not starting"; exit 1; }; done
fi
SRVENV=$(systemctl --user show strata.service -p Environment --value | tr ' ' '\n' | grep '^STRATA_' | tr '\n' ' ')
export SRVENV
PROD_SHA=$(sha256sum "$PEXE" | cut -c1-16); PRODCFG_SHA=$(sha256sum "$PRODSRC" | cut -c1-16)
[ "$DRY" = 1 ] && rm -rf "$OUT"
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$OUT" "$CF"
# the configs: A = production.json as is; V = A with the v3.1 files; V2 = V with layer_split $V_SPLIT2
python3 - "$PRODSRC" "$CF" "$V31_PACK" "$V31_GGUF" "$V31_NAME" "$V_SPLIT2" <<'PYEOF' || { echo "cannot write the configs"; exit 1; }
import json, sys, copy
src, cf, pack, gguf, name, split2 = sys.argv[1:7]
a = json.load(open(src)); json.dump(a, open(f"{cf}/A.json", "w"), indent=1)
v = copy.deepcopy(a); args = v["args"]
def setarg(flag, val):
    i = args.index(flag); args[i + 1] = val
old_pack = args[args.index("--pack") + 1]; old_gguf = args[args.index("--native") + 1]
setarg("--pack", pack); setarg("--native", gguf)
assert v["tokenizer"].startswith(old_pack), (v["tokenizer"], old_pack)
v["tokenizer"] = pack + v["tokenizer"][len(old_pack):]
if v.get("vision", {}).get("model") == old_gguf: v["vision"]["model"] = gguf
v["model_name"] = name
assert not any(x == old_pack or x.startswith(old_pack + "/") or x == old_gguf for x in args), "a v3.0 path is left in V's args"
json.dump(v, open(f"{cf}/V.json", "w"), indent=1)
v2 = copy.deepcopy(v); v2["layer_split"] = split2; json.dump(v2, open(f"{cf}/V2.json", "w"), indent=1)
PYEOF
POWER=$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ';')
log "window prep ($TAG): production $(basename "$PRODSRC") sha256 $PRODCFG_SHA, exe $(basename "$PEXE") sha256 $PROD_SHA, server $SRV @$SRV_REV; server env: $SRVENV; GPU power limits: $POWER"
for a in V V2; do log "    [cfg] $a vs A: $(python3 -c "
import json,sys; p=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
d=[f'{k}: {p.get(k)} -> {c[k]}' for k in c if k not in ('args','env') and p.get(k)!=c[k]]
d+=[f'arg {x} -> {y}' for x,y in zip(p['args'],c['args']) if x!=y]
d+=['env differs' for _ in [0] if p['env']!=c['env']]; print('; '.join(d))" "$CF/A.json" "$CF/$a.json")"; done
NEED_S=$(( 2 * ARM_EST + ( R2 == 1 ? 2 * R2_EST : 0 ) + 300 ))
if [ "$DRY" = 1 ]; then
    log "dry run: prep and configs ok, production untouched ($CF); expected outage ~$(( NEED_S / 60 )) min (cap $(( WINDOW_S / 60 )) min, steps until $(( (WINDOW_S - RESERVE) / 60 )))"
    echo "  step arm_V  (expected ~$(( ARM_EST / 60 )) min, cap $(( V_CAP / 60 ))): up, smoke, short 1/2/4/8/16 x2, long 16 x2, mt 16x3, cold 128K, eval$([ "$EVAL" = 1 ] || echo ' (EVAL=0: skipped)') on the v3.1 files; ends after its warm-up when not every expert is in the GPU caches"
    echo "  step arm_V2 only then (layer_split $V_SPLIT2), in V's place (cap $(( V_CAP / 60 )))"
    echo "  step arm_A  (expected ~$(( ARM_EST / 60 )) min, cap $(( ARM_CAP / 60 ))): the same on production.json as is"
    [ "$R2" = 1 ] && echo "  steps r2_A, r2_V (expected ~$(( R2_EST / 60 )) min each, while time is left): up, short, long"
    echo "  then: production restarted (EXIT trap; $GUNIT.service on a SIGKILL; $RUNIT.timer backstop), summary -> $OUT/summary.txt"
    exit 0
fi
[ "$NEED_S" -le $(( WINDOW_S - RESERVE )) ] || log "note: the arms need ~$(( NEED_S / 60 )) min, the window allows $(( (WINDOW_S - RESERVE) / 60 )): r2 may be skipped"

# ---- the window -------------------------------------------------------------------------------------------------------
for _ in $(seq 180); do prod_loaded && break; sleep 5; done
prod_loaded || { echo "production does not report loaded: not stopping it"; exit 1; }
RESTORE_TS=$(( $(date +%s) + WINDOW_S + 900 ))
arm_restore "$RESTORE_TS" || { echo "could not arm $RUNIT.timer: production left running"; exit 1; }
: > "$OUT/step.pgid"
arm_guard || { disarm_restore; disarm_guard; echo "could not arm $GUNIT.service: production left running"; exit 1; }
curl -s -m 5 "$URL/metrics" > "$OUT/prod-metrics-before.txt" 2>/dev/null
log "window start ($(( WINDOW_S / 60 )) min cap; $GUNIT.service waits for PID $$; $RUNIT.timer armed for $(date -d "@$RESTORE_TS" '+%F %T')): stopping strata.service (coordinate with the bench session first: done by the operator)"
trap '' INT TERM HUP
trap final EXIT
systemctl --user stop strata.service
STOPPED=1; T0=$(date +%s)
trap 'on_sig signal' INT TERM HUP
( sleep $(( WINDOW_S + 300 )); tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null | grep -q window-v31 && kill -TERM $$ ) 2>/dev/null & WD=$!
for _ in $(seq 60); do ss -Hltn 'sport = :8090' | grep -q . || break; sleep 2; done
free_gpus || on_sig "GPUs not free"

# V first (does v3.1 load and fit: ends after its warm-up when not every expert is in the GPU caches, V2 then), A, r2 in
# reverse: VAAV.  V and V2 are capped at V_CAP so that A (the baseline) still fits after them
step arm_V "$ARM_EST" "$V_CAP" arm_run V V full fitgate
VL=V                                             # V2 (another split) replaces V when V's engine did not hold every expert
if ! grep -aq 'every expert is in the GPU caches' "$OUT/engine-V.log" 2>/dev/null; then
    log "V: no 'every expert is in the GPU caches' line (no fit, or V failed before it): V2 with layer_split $V_SPLIT2"
    step arm_V2 "$ARM_EST" "$V_CAP" arm_run V2 V2 full
    grep -aq 'every expert is in the GPU caches' "$OUT/engine-V2.log" 2>/dev/null && VL=V2
fi
step arm_A "$ARM_EST" "$ARM_CAP" arm_run A A full
if [ "$R2" = 1 ]; then
    step r2_A "$R2_EST" "$R2_CAP" arm_run A Ar2 r2
    step "r2_$VL" "$R2_EST" "$R2_CAP" arm_run "$VL" "${VL}r2" r2
fi
exit 0                                           # -> EXIT trap: production back, then the summary
