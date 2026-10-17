#!/usr/bin/env bash
# GPU window for branch ll5-reads (Strata-ll5-reads, HEAD >= a5cf62b1): shorter prompt reads, phase 1 - T0a
# STRATA_SR_CKPT_WARM=1 (a warm checkpoint pool), T0b STRATA_SR_CKPT_LATE=1 (the turn checkpoint after BADM), T0c
# STRATA_SR_NODRAIN=1 (BGEN without the pipe drain), umbrella STRATA_SHORT_READS=1 - against production as it runs now
# (production.json read at run time: strata-g3-1441f-lanes4, overlap + KVS, solo pipeline, STRATA_SPEC_LANES=4,
# --kv-resident 28672, split 12,25,37, 16 slots; the window refuses a production config of another shape).
# Binary: strata-g3-1441h-reads (a copy of Strata-ll5-reads/build/strata).
# ~1 h 35-50 min expected, hard cap WINDOW_S (default 7200 s = 2 h, production restart included): a step that no longer
# fits is skipped (logged), steps are ordered by value.  Production is stopped for the window and started again
# unchanged (production.json is never written), also on abort (trap).  Every test engine logs to results/ (never
# production's log) and spills into /media/raid0/strata-kvspill-bench/reads (never production's tier dir); every test
# HTTP server listens on 127.0.0.1:8091 (never production's port 8090).  A signal (INT/TERM/HUP) before production is
# stopped only exits; after it, it interrupts the running step at once, ends the bench engines and starts production
# again.  An existing results/reads is moved aside to results/reads.prev-<time> at the start.
#   run:      cd /media/raid0/StrataQ38FN/linux && ./window-reads.sh
#   verdicts: grep -E 'PASS|FAIL|SKIP' results/reads/window.log
# Before the window (no GPU, production serving): the prompts of the HTTP workloads measured with the tokenizer (every
# request incl. its output < 262,144; the ovkv background documents > --kv-resident), the lanes4 window's serial solo
# reference checked for reuse (same binary as production's, same config without the lanes, same prompts), pictures.
# Steps (each with PASS/FAIL/INFO lines and a time budget; run in the order 0, 2, 3, 1 - the INFO-only A/B last):
#   0 phase-0 measurement: lanes_stdin.py --mode prefill (STRATA_TRACE=1 + STRATA_PREFILL_TIMING=1), reads of 170, 512
#     and 1200 tokens, idle and beside 15 decoding slots, 3 reps: production, then the new binary with the umbrella.
#     Per size: send->BADM, the read, the overhead outside it and the admission's phases (drain, checkpoint save /
#     evict, slot copy, slot freed), the prefill timing breakdown (GPU / host)
#   1 zero-code A/B --short-read 16 and 32 vs 64 (production's binary + config, the argument appended): 16 users short
#     + 1.2K prompts with mixed reply lengths (MIX=128), then bench_multiturn's agent mix (16 users x 4 turns, follow-ups
#     of ~50 fresh tokens: between 32 and 64, so the threshold decides their path); one run per arm (INFO; runs last)
#   2 exactness: batch_test16 greedy (16 slots, a cold 40K read beside 15) production vs the new binary env unset, each
#     switch alone and the umbrella (16/16 identical), boot VRAM / expert cache with T0b; ovkv_ident-style HTTP
#     conversations (4 background ~36K-token streams, cold joins 1.5K/12K/37K tokens, resumes, second turns) production
#     vs the umbrella with STRATA_SR_CKPT_LATE_VERIFY=1 and STRATA_SLOT_PARK_VERIFY=1; batch_test16 at T=0.7 seed 1234;
#     the lanes with the umbrella == serial solo spec (1 user pipelined + resume, switch 1-2-3-4-2-1, HOLD 2/3/4); the
#     late-copy fault (STRATA_SR_CKPT_LATE_TEST_FAULT=copy) over HTTP with parks: no exit, replies identical
#   3 speed: ab_quick 16 users ABBA x2 MIX=128 production vs the umbrella (identical replies, agg, TTFT p50/p95/max),
#     then 1-4 users ABBA x2 (the lanes must not regress: decode p50 per user, lane starts by lane count); the
#     umbrella's HTTP engines must boot with all three switches on
#   4 production restarted unchanged, health checked
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/reads; CF=$OUT/cfg
URL=http://127.0.0.1:8090; TPORT=8091; PY=$R/Strata/.venv/bin/python; DRV=$L/lanes_stdin.py; RT=$L/reads_tools.py
WT=$R/Strata-ll5-reads; B=$WT/build/strata; NEW=$R/Strata/engine/strata-g3-1441h-reads
WANT_COMMIT=a5cf62b1
PROD_EXE_WANT=${PROD_EXE_WANT:-strata-g3-1441f-lanes4}       # what production must run for this window's comparisons
PRODSRC=$(readlink -f "$C/production.json")                 # what production runs now (its config is only read)
SRV=$R/Strata-143-prod                                      # production's server checkout (ab_quick / run_strata_variant)
TIER=/media/raid0/strata-kvspill-bench/reads; ENGLOG=$L/results/engine-reads.log
LANES4=$L/results/lanes4                                    # the lanes4 window: its serial solo reference, its pictures
EXACT="--pcie-frac 0 --adapt-every 0 --suffix-draft 0"
XENV=(--env STRATA_IQ_MT_MIN=1 --env STRATA_BATCH_DECODE_SHARE=0 --env STRATA_DECODE_TIMING=1)
NO_OV=(--drop-env STRATA_PIPE_OVERLAP --drop-env STRATA_PIPE_OVERLAP_KVS)   # (an overlapped read sends lane users through the pump)
# the HTTP workloads (their prompts are measured in the prep with the tokenizer)
OV_ARGS="--bg 4 --bg-ctx 24000 --bg-tok 400 --joins 1000,8000,24000 --seed 7"
MT_ARGS="--users 16 --turns 4 --doc 2000 --add 8 --max-tok 192 --think 0.5"
WINDOW_S=${WINDOW_S:-7200}; RESERVE=480; T0=$(date +%s)
# until production is stopped a signal only ends the script (nothing to clean up; production keeps serving): the
# full abort (bench_down kills GPU processes of $R/Strata*, production's engine included) is armed after STOPPED=1
trap 'echo "window aborted before production was stopped"; exit 1' INT TERM HUP

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
vram() { nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tr '\n' ';'; }
free_gpus() { for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do case "$(readlink -f /proc/$p/exe 2>/dev/null)" in $R/Strata*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production as before (production.json -> $(readlink $C/production.json))"
    systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
        systemctl --user is-active -q strata.service || { log "strata.service died during the start FAIL"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "production start timeout FAIL"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s: $(curl -s -m 5 $URL/health | cut -c1-200)"; }
bench_down() { systemctl --user stop strata-bench.scope 2>/dev/null; systemctl --user reset-failed strata-bench.scope 2>/dev/null
    free_gpus || log "GPUs busy after $1"; }
STOPPED=0; STEP_PID=
# abort: armed only after STOPPED=1.  The step's timeout is its own process-group leader (it runs in the background
# and the script waits for it, so a signal interrupts the wait at once): INT to its group, then a short wait, then
# bench_down and production (signals are ignored from here on: a second Ctrl-C must not end the script between
# bench_down and start_prod)
abort() { trap '' INT TERM HUP; log "window aborted: $*"
    if [ -n "$STEP_PID" ] && kill -0 "$STEP_PID" 2>/dev/null; then
        kill -INT -- -"$STEP_PID" 2>/dev/null || kill -INT "$STEP_PID" 2>/dev/null
        for i in $(seq 30); do kill -0 "$STEP_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL -- -"$STEP_PID" 2>/dev/null; fi
    bench_down abort
    if [ "$STOPPED" = 1 ]; then start_prod || log "PRODUCTION DID NOT COME BACK - check strata.service now"; fi; exit 1; }
left() { echo $(( T0 + WINDOW_S - RESERVE - $(date +%s) )); }
# step <name> <expected s> <cap s> <cmd...>: skipped when the expected time no longer fits the window; the cap is cut
# to what is left; the output goes to $OUT/<name>.txt and its '==>' verdict lines into window.log
step() { local name=$1 est=$2 limit=$3; shift 3; local l; l=$(left)
    if [ "$l" -lt "$est" ]; then log "step $name: SKIP (needs ~${est} s, ${l} s left in the window)"; echo skip > "$OUT/rc_$name"; return 1; fi
    [ "$limit" -gt "$l" ] && limit=$l
    log "step $name: start (expected ~${est} s, cap ${limit} s); VRAM (index, used, free): $(vram)"
    timeout -s INT -k 120 "$limit" bash -c '"$@"' "$name" "$@" > "$OUT/$name.txt" 2>&1 < /dev/null &
    STEP_PID=$!; wait "$STEP_PID"; local rc=$?; STEP_PID=; echo $rc > "$OUT/rc_$name"
    [ $rc -eq 124 ] && log "step $name: TIMEOUT after ${limit} s FAIL"
    grep -aE '^==>' "$OUT/$name.txt" | cut -c1-700 | sed "s/^/    [$name] /" | tee -a "$OUT/window.log" >/dev/null
    log "step $name: exit $rc"
    bench_down "$name"; return $rc; }
# a step that is not run for a reason other than time: its rc file says skip, so its judges say SKIP too
noskip() { log "step $1: SKIP ($2)"; echo skip > "$OUT/rc_$1"; }
# judge <name> <steps> <cmd...>: a check without the GPU of what <steps> produced (a comma list: every one must have
# run, else SKIP - no verdict from stale files; empty: no gate, the command skips missing logs itself)
step_ran() { [ -f "$OUT/rc_$1" ] && [ "$(cat "$OUT/rc_$1")" != skip ]; }
judge() { local name=$1 from=$2 s; shift 2
    for s in ${from//,/ }; do step_ran "$s" || { log "    [$name] SKIP (step $s skipped)"; return 0; }; done
    timeout 600 "$@" > "$OUT/$name.txt" 2>&1
    grep -aE '^==>|^    ' "$OUT/$name.txt" | cut -c1-700 | sed "s/^/    [$name] /" | tee -a "$OUT/window.log" >/dev/null
    grep -qaE '^==>' "$OUT/$name.txt" || log "    [$name] no verdict (see $OUT/$name.txt) FAIL"; }
# scoped <cmd...>: in the bench scope (memory cap as run_strata_variant's; bench_down ends whatever is left of it)
scoped() { systemctl --user stop strata-bench.scope 2>/dev/null; systemctl --user reset-failed strata-bench.scope 2>/dev/null
    systemd-run --user --scope -q --unit=strata-bench -p MemoryMax=200G -p MemorySwapMax=0 "$@"; }
drv() { scoped "$PY" -I "$DRV" "$@"; }
# test HTTP servers: 127.0.0.1:$TPORT (run_strata_variant.sh / ab_quick.sh read PORT and BIND from the env), never
# production's 8090
ab() { ( cd $L && SERVER_DIR=$SRV PORT=$TPORT BIND=127.0.0.1 "$@" ); }
tup() { curl -s -m 3 "http://127.0.0.1:$TPORT/health" | grep -q '"loaded": *true'; }
# the HTTP configs' engine log is $ENGLOG (shared, appended): an arm's own part is cut from its line count before the
# start on (so an engine the server had to restart shows as a second 'engine started' line)
enl() { if [ -f "$ENGLOG" ]; then wc -l < "$ENGLOG"; else echo 0; fi; }
enl_copy() { tail -n +$(( $1 + 1 )) "$ENGLOG" > "$2" 2>/dev/null; }
# sr_arm <tag> <config>: phase 1, one server: 16 users short, then 1.2K prompts (MIX=128 both), then the agent mix
sr_arm() { local tag=$1 cfg=$2 n n0; n=$(basename "$cfg" .json); n0=$(enl)
    ab env KEEP=1 ./run_strata_variant.sh "ab-$tag-$n-short" "$cfg" 16 --rounds 2 --max-tok 512 --prompt-tok 100 \
        --temperature 0 --max-tok-min 128 2>&1 | tail -4
    if ! tup; then echo "==> $n: the test server did not come up FAIL"; enl_copy "$n0" "$OUT/engine-sr_$n.log"
        systemctl --user stop strata-bench.scope; return 1; fi
    ab env ATTACH=1 KEEP=1 ./run_strata_variant.sh "ab-$tag-$n-long" "$cfg" 16 --rounds 2 --max-tok 512 --temperature 0 \
        --warmup 0 --max-tok-min 128 2>&1 | tail -3
    ( cd "$L" && timeout -s INT 600 "$PY" -I bench_multiturn.py --url "http://127.0.0.1:$TPORT" --label "$tag-$n-mt" \
        $MT_ARGS --temperature 0 --out "$OUT/multiturn.jsonl" 2>&1 | tail -8 )
    sleep 2; enl_copy "$n0" "$OUT/engine-sr_$n.log"
    systemctl --user stop strata-bench.scope; }
# ov_arm <name> <config>: an ovkv_ident.py run (greedy) on its own server (a 1-request warm-up starts it)
ov_arm() { local name=$1 cfg=$2 n0 rc; n0=$(enl)
    ab env KEEP=1 ./run_strata_variant.sh "rdov-$name" "$cfg" 1 --rounds 1 --max-tok 16 --prompt-tok 100 --temperature 0 2>&1 | tail -3
    if ! tup; then echo "==> ov $name: the test server did not come up FAIL"; enl_copy "$n0" "$OUT/engine-ov_$name.log"
        systemctl --user stop strata-bench.scope; return 1; fi
    ( cd "$L" && timeout -s INT 1500 "$PY" -I ovkv_ident.py --url "http://127.0.0.1:$TPORT" $OV_ARGS --out "$OUT/ov_$name.json" )
    rc=$?; sleep 3; enl_copy "$n0" "$OUT/engine-ov_$name.log"
    python3 - "$OUT/ov_$name.json" "$name" <<'PYEOF'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print(f"==> ov {sys.argv[2]}: no result ({e}) FAIL"); sys.exit(0)
bad = [k for k, v in d.items() if v["error"]]
print(f"==> ov {sys.argv[2]}: {len(d)} replies, {len(bad)} with errors {bad[:4]} {'PASS' if d and not bad else 'FAIL'}")
PYEOF
    systemctl --user stop strata-bench.scope; return $rc; }
LANES_TIER_ROOT=$TIER
export -f scoped drv ab tup enl enl_copy sr_arm ov_arm
export PY DRV L SRV TPORT LANES_TIER_ROOT ENGLOG OUT OV_ARGS MT_ARGS   # (step runs them under timeout through bash -c)
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }
cfg_exe() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" "$1"; }

# ---- prep (no GPU; production still serving) ------------------------------------------------------------------------
[ -x "$B" ] || { echo "missing $B"; exit 1; }
git -C "$WT" merge-base --is-ancestor $WANT_COMMIT HEAD || { echo "$WT HEAD lacks $WANT_COMMIT (the review fixes)"; exit 1; }
# (the build may come before the commit's time: compare the sources' times with the binary's, not the commit's)
SRCS=(); for d in src include CMakeLists.txt; do [ -e "$WT/$d" ] && SRCS+=("$WT/$d"); done
NEWER=$(find "${SRCS[@]}" -newer "$B" \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \
        -o -name CMakeLists.txt -o -name '*.cmake' \) | head -3)
[ -z "$NEWER" ] || { echo "$B is older than $NEWER: rebuild first"; exit 1; }
git -C "$WT" diff --quiet HEAD -- src include CMakeLists.txt \
    || { echo "$WT has uncommitted changes under src/, include/ or CMakeLists.txt: the binary is not HEAD"; exit 1; }
for f in "$DRV" "$RT" "$L/ab_quick.sh" "$L/run_strata_variant.sh" "$L/compare_replies.py" "$L/bench_multiturn.py" \
         "$L/ovkv_ident.py" "$R/Strata-1441/tools/batch_test16.py" "$PRODSRC"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
for f in "$DRV" "$RT" "$L/ab_quick.sh" "$L/run_strata_variant.sh"; do [ -x "$f" ] || { echo "$f is not executable (chmod +x)"; exit 1; }; done
"$PY" -I "$DRV" --help > /dev/null || { echo "$DRV does not load"; exit 1; }
"$PY" -I "$RT" --help > /dev/null || { echo "$RT does not load"; exit 1; }
curl -s -m 2 "http://127.0.0.1:$TPORT/health" > /dev/null && { echo "port $TPORT already serves something: the test servers need it"; exit 1; }
PEXE=$(cfg_exe "$PRODSRC"); [ -x "$PEXE" ] || { echo "production's exe $PEXE missing"; exit 1; }
# production's shape: what every comparison of this window assumes (the reference is production as it runs)
python3 - "$PRODSRC" "$PROD_EXE_WANT" <<'PYEOF' || exit 1
import json, os, sys
c = json.load(open(sys.argv[1])); e = c["env"]; a = c["args"]
want = {"STRATA_SPEC_LANES": "4", "STRATA_SOLO_PIPELINE": "1", "STRATA_PIPE_OVERLAP": "1", "STRATA_PIPE_OVERLAP_KVS": "1",
        "STRATA_NAN_GUARD": "0", "STRATA_CKPT_POOL": "1"}
bad = {k: e.get(k) for k, v in want.items() if e.get(k) != v}
kv = a[a.index("--kv-resident") + 1] if "--kv-resident" in a else None
extra = [k for k in e if k.startswith("STRATA_SR_") or k in ("STRATA_SHORT_READS", "STRATA_BATCH_MTP")]
why = []
if bad: why.append(f"env differs: {bad}")
if kv != "28672": why.append(f"--kv-resident {kv}")
if "--short-read" in a or "--batch-mtp" in a: why.append("--short-read / --batch-mtp set")
if extra: why.append(f"already has {extra}")
if int(c.get("parallel") or 0) != 16: why.append(f"parallel {c.get('parallel')}")
if c.get("layer_split") != "12,25,37": why.append(f"layer_split {c.get('layer_split')}")
if os.path.basename(c["exe"]) != sys.argv[2]: why.append(f"exe {os.path.basename(c['exe'])} (want {sys.argv[2]})")
if why:
    print(f"production's config {sys.argv[1]} is not the one this window is written for: {'; '.join(why)}"); sys.exit(1)
PYEOF
# (before the copy: never overwrite the binary production runs)
[ "$(readlink -f "$PEXE")" != "$(readlink -f "$NEW")" ] || { echo "production already runs $NEW: nothing to compare against"; exit 1; }
# the window restores production to running: it must be running now (an inactive service is not started behind the
# user's back)
systemctl --user is-active -q strata.service || { echo "strata.service is not active: start it first"; exit 1; }
PROD_SHA=$(sha256sum "$PEXE" | cut -c1-16); PRODCFG_SHA=$(sha256sum "$PRODSRC" | cut -c1-16)
# the HTTP workloads' prompts, measured with the model's tokenizer (before anything is changed)
OVT=(--ov-bg 4 --ov-bg-ctx 24000 --ov-bg-tok 400 --ov-joins 1000,8000,24000 --ov-seed 7)
MTT=(--mt-users 16 --mt-turns 4 --mt-doc 2000 --mt-add 8 --mt-max-tok 192)
TOKCHK=$(taskset -c 0-63 nice -n 15 "$PY" -I "$RT" --mode tokcheck --config "$PRODSRC" --kvres 28672 "${OVT[@]}" "${MTT[@]}" 2>&1) \
    || { echo "$TOKCHK"; echo "the HTTP workloads do not fit (see above)"; exit 1; }
cmp -s "$B" "$NEW" || { cp "$B" "$NEW" && chmod +x "$NEW"; } || { echo "cannot copy $B to $NEW"; exit 1; }
# stale results (still before production is stopped): an earlier window's results/reads moves aside, its HTTP engine
# run logs go with it (their names are fixed: a skipped step must not leave an old log for a judge)
RUNLOGS=("$L"/results/engine-ab-srd-*.run.log "$L"/results/engine-ab-rd16-*.run.log "$L"/results/engine-ab-rdlow-*.run.log
         "$L"/results/engine-rdov-*.run.log)
if [ -e "$OUT" ]; then PREV=$OUT.prev-$(date +%Y%m%d-%H%M%S); mv "$OUT" "$PREV" || { echo "cannot move $OUT aside"; exit 1; }
    for f in "${RUNLOGS[@]}"; do [ -e "$f" ] && mv "$f" "$PREV"/; done; fi
for f in "${RUNLOGS[@]}"; do rm -f "$f"; done
rm -rf "$TIER"; mkdir -p "$OUT" "$CF" "$TIER"
echo "$TOKCHK" | sed 's/^/    [tokcheck] /' | tee -a "$OUT/window.log" >/dev/null
# sweep.jsonl rows of earlier windows with the same labels must not count (the sweep judges read from here on)
SWEEP0=$(cat "$L/results/sweep.jsonl" 2>/dev/null | wc -l)
log "window prep: $(basename $NEW) sha256 $(sha256sum $NEW | cut -c1-16) = $WT $(git -C $WT rev-parse --short HEAD) ($(git -C $WT log -1 --format=%s HEAD | cut -c1-80)); production: $(basename $PRODSRC) sha256 $PRODCFG_SHA, exe $(basename $PEXE) sha256 $PROD_SHA"

# configs, all from production's config: exe, the bench log, a tier dir of its own (empty), then the arm's changes:
# K=V sets an env key, -K drops one, ARG--opt=V sets an argument's value, ARG+--opt=V sets it or appends it
mkcfg() {   # $1 name, $2 exe, then changes ...
    local name=$1 exe=$2; shift 2; mkdir -p "$TIER/http-$name"
    python3 - "$PRODSRC" "$CF/$name.json" "$exe" "$TIER/http-$name" "$ENGLOG" "$@" <<'PYEOF' || exit 1
import json, sys
src, dst, exe, tier, log = sys.argv[1:6]
c = json.load(open(src)); c["exe"] = exe; c["log"] = log; e = c["env"]; e["STRATA_CONV_TIER_DIR"] = tier
for kv in sys.argv[6:]:
    if kv.startswith("ARG+"):
        k, v = kv[4:].split("=", 1); a = c["args"]
        if k in a: a[a.index(k) + 1] = v
        else: a += [k, v]
    elif kv.startswith("ARG"):
        k, v = kv[3:].split("=", 1); a = c["args"]; i = a.index(k); a[i + 1] = v
    elif kv.startswith("-"):
        e.pop(kv[1:], None)
    else:
        k, v = kv.split("=", 1); e[k] = v
json.dump(c, open(dst, "w"), indent=1)
PYEOF
}
SR=(STRATA_SHORT_READS=1)
LOGS=(STRATA_SPEC_LANES_LOG=1 STRATA_SOLO_PIPELINE_LOG=1)
mkcfg RD-PROD  "$PEXE"                                      # production as it runs
mkcfg RD-SR64  "$PEXE"                                      # phase 1: --short-read 64 (the default) ...
mkcfg RD-SR32  "$PEXE" ARG+--short-read=32                  # ... vs 32
mkcfg RD-SR16  "$PEXE" ARG+--short-read=16                  # ... vs 16
mkcfg RD-N0    "$NEW"                                       # the new binary, env unset (default path)
mkcfg RD-NA    "$NEW" STRATA_SR_CKPT_WARM=1                 # T0a
mkcfg RD-NB    "$NEW" STRATA_SR_CKPT_LATE=1                 # T0b
mkcfg RD-NC    "$NEW" STRATA_SR_NODRAIN=1                   # T0c
mkcfg RD-NU    "$NEW" "${SR[@]}"                            # the umbrella
mkcfg RD-NUV   "$NEW" "${SR[@]}" STRATA_SR_CKPT_LATE_VERIFY=1
mkcfg OV-PROD  "$PEXE" STRATA_SLOT_PARK_VERIFY=1 STRATA_TRACE=1
mkcfg OV-NUV   "$NEW" "${SR[@]}" STRATA_SR_CKPT_LATE_VERIFY=1 STRATA_SLOT_PARK_VERIFY=1 STRATA_TRACE=1
mkcfg OV-NF    "$NEW" "${SR[@]}" STRATA_SR_CKPT_LATE_TEST_FAULT=copy STRATA_SLOT_PARK_VERIFY=1 STRATA_TRACE=1
mkcfg LN-NU    "$NEW" "${SR[@]}" "${LOGS[@]}"               # the lane drives (production's lanes, the umbrella)
mkcfg LN-OFF   "$PEXE" -STRATA_SPEC_LANES -STRATA_SOLO_PIPELINE   # the serial solo reference (if it must be run)
mkcfg SP-PROD  "$PEXE" "${LOGS[@]}"                         # speed (log lines in both arms)
mkcfg SP-NU    "$NEW" "${SR[@]}" "${LOGS[@]}"
# a copy of a config with a fresh, empty tier dir of its own (batch_test16's 40K prompt is tiered: a later arm must
# not find an earlier arm's copy under STRATA_CONV_TIER_WARM)
tiercfg() { local d=$TIER/$2; rm -rf "$d"; mkdir -p "$d"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; json.dump(c,open(sys.argv[3],'w'),indent=1)" \
        "$1" "$d" "$CF/$2.json"; echo "$CF/$2.json"; }

# the serial solo reference of the lane drives: the lanes4 window's (production's binary, its config without the lanes,
# the exactness settings, --kv-resident 28672) when it still matches, else this window runs it (step ref)
REF_REUSED=0
if "$PY" -I "$RT" --mode refok --ref "$LANES4/ref.json" --ref-txt "$LANES4/ref.txt" --ref-cfg "$LANES4/cfg/LC-OFF-28672.json" \
        --window-log "$LANES4/window.log" --prod "$PRODSRC" --prod-sha "$PROD_SHA" --kvres 28672 > "$OUT/refok.txt" 2>&1; then
    cp "$LANES4/ref.json" "$OUT/ref.json" && REF_REUSED=1 && echo 0 > "$OUT/rc_ref"; fi
log "reference: $(tail -1 "$OUT/refok.txt" | cut -c1-300)"
# pictures for the vision lane case: the lanes4 window's (paths rewritten), else encoded now on the CPU (the config's
# encoder: "gpu": false; production still serves: off its --aux-cpus 65-127, niced, capped)
if [ -f "$LANES4/pics/pics.json" ]; then
    cp -r "$LANES4/pics" "$OUT/pics" && python3 - "$OUT/pics/pics.json" "$LANES4/pics" "$OUT/pics" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))
for v in p.values(): v["file"] = v["file"].replace(sys.argv[2], sys.argv[3])
json.dump(p, open(sys.argv[1], "w"))
PYEOF
else
    timeout 1200 taskset -c 0-63 nice -n 15 "$PY" -I "$DRV" --mode encode --config "$PRODSRC" --out "$OUT/pics" > "$OUT/encode.txt" 2>&1
    grep -aE '^==>' "$OUT/encode.txt" | sed 's/^/    [encode] /' | tee -a "$OUT/window.log" >/dev/null
fi
PICS=$OUT/pics/pics.json; [ -f "$PICS" ] || log "    [pics] no pictures: the L_vis group is skipped FAIL"

log "window start ($(( WINDOW_S / 60 )) min cap): stopping strata.service"
STOPPED=1; trap 'abort signal' INT TERM HUP     # from here on an abort ends the bench engines and restarts production
systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || abort "GPUs not free"

# ---- 0. phase-0 measurement: prompt reads by length, idle and beside 15 decoding slots --------------------------------
PF=(--mode prefill --sizes 170,512,1200 --reps 3 --phases idle,dec15 --env STRATA_PREFILL_TIMING=1)
step pf_PROD 300 900 drv "${PF[@]}" --config $CF/RD-PROD.json --label PF-PROD --log $OUT/engine-pf_PROD.log --out $OUT/pf_PROD.json
step pf_NU 300 900 drv "${PF[@]}" --config $CF/RD-NU.json --label PF-NU --log $OUT/engine-pf_NU.log --out $OUT/pf_NU.json
PFA=(PROD=$OUT/pf_PROD.json=$OUT/engine-pf_PROD.log); step_ran pf_NU && PFA+=(NU=$OUT/pf_NU.json=$OUT/engine-pf_NU.log)
judge prefsum pf_PROD "$PY" -I "$RT" --mode prefsum --arms "${PFA[@]}"
judge pf_eng pf_NU "$PY" -I "$RT" --mode engstat --arms PROD=$OUT/engine-pf_PROD.log NU=$OUT/engine-pf_NU.log \
    --want-on NU:WARM,LATE,NODRAIN --want-off-all PROD --want-clean PROD,NU --compare PROD,NU

# ---- 2. exactness ----------------------------------------------------------------------------------------------------
# 2a batch_test16 (greedy, 16 slots, a cold 40K read beside 15 decoding, production's env; STRATA_TRACE=1 in every arm
#    for the admission phases and the counters): production vs the new binary env unset, the umbrella, each switch
for arm in PROD N0 NU NA NB NC; do
    cfg=$(tiercfg $CF/RD-$arm.json bt_$arm); exe=$(cfg_exe "$cfg")
    step bt_$arm 150 900 scoped bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-bt_$arm.log env $(cfg_env $cfg) STRATA_TRACE=1 python3 tools/batch_test16.py --exe $exe --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --dump $OUT/bt_$arm.json"
done
judge bt_cmp bt_PROD "$PY" -I "$RT" --mode btcmp --dir $OUT --base bt_PROD --arms bt_N0,bt_NU,bt_NA,bt_NB,bt_NC --what "batch_test16 greedy"
BTE=(); for arm in PROD N0 NU NA NB NC; do BTE+=($arm=$OUT/engine-bt_$arm.log); done
judge bt_eng bt_PROD "$PY" -I "$RT" --mode engstat --arms "${BTE[@]}" --want-on NA:WARM --want-on NB:LATE --want-on NC:NODRAIN \
    --want-on NU:WARM,LATE,NODRAIN --want-off-all PROD,N0 --want-clean PROD,N0,NU,NA,NB,NC --compare N0,NA
judge bt_eng_b bt_N0,bt_NB "$PY" -I "$RT" --mode engstat --arms N0=$OUT/engine-bt_N0.log NB=$OUT/engine-bt_NB.log --compare N0,NB
# the default path's boot == production's; T0b's 40 MiB per card out of the expert caches (still 100 % resident?)
judge boot_N0 bt_PROD,bt_N0 "$PY" -I "$RT" --mode bootcmp --base $OUT/engine-bt_PROD.log --got $OUT/engine-bt_N0.log --what "new binary env unset vs production"
judge boot_NB bt_N0,bt_NB "$PY" -I "$RT" --mode bootcmp --base $OUT/engine-bt_N0.log --got $OUT/engine-bt_NB.log --what "T0b (STRATA_SR_CKPT_LATE) vs env unset"
judge boot_NU bt_N0,bt_NU "$PY" -I "$RT" --mode bootcmp --base $OUT/engine-bt_N0.log --got $OUT/engine-bt_NU.log --what "umbrella vs env unset"

# 2b HTTP conversations (ovkv_ident.py, greedy): 4 background streams of ~36.5K tokens (> --kv-resident: K/V streamed;
#    rounds 2-3 resume them from the conversation cache), cold joins of ~1.5K / 12K / 37K tokens beside them (T0c's
#    reads beside groups in flight), each join's second turn (a resume); STRATA_SLOT_PARK_VERIFY=1 both arms,
#    STRATA_SR_CKPT_LATE_VERIFY=1 in the umbrella arm: production vs the umbrella, every reply identical
step ov_PROD 300 1500 ov_arm PROD $CF/OV-PROD.json
step ov_NUV 300 1500 ov_arm NUV $CF/OV-NUV.json
judge ov_cmp ov_PROD,ov_NUV "$PY" -I "$L/ovkv_ident.py" --compare $OUT/ov_PROD.json $OUT/ov_NUV.json
judge ov_eng ov_NUV "$PY" -I "$RT" --mode engstat --arms PROD=$OUT/engine-ov_PROD.log NUV=$OUT/engine-ov_NUV.log \
    --want-on NUV:WARM,LATE,NODRAIN --want-verify NUV --want-park NUV --want-clean PROD,NUV --compare PROD,NUV

# 2c batch_test16 at T=0.7 seed 1234 (every slot's sampling seeded): production vs the umbrella + LATE_VERIFY
for arm in PROD NUV; do
    cfg=$(tiercfg $CF/RD-$arm.json bt7_$arm); exe=$(cfg_exe "$cfg")
    step bt7_$arm 150 900 scoped bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-bt7_$arm.log env $(cfg_env $cfg) STRATA_TRACE=1 python3 tools/batch_test16.py --exe $exe --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --keys 'temperature=0.7 seed=1234' --dump $OUT/bt7_$arm.json"
done
judge bt7_cmp bt7_PROD "$PY" -I "$RT" --mode btcmp --dir $OUT --base bt7_PROD --arms bt7_NUV --what "batch_test16 T=0.7 seed 1234"
judge bt7_eng bt7_NUV "$PY" -I "$RT" --mode engstat --arms NUV=$OUT/engine-bt7_NUV.log --want-verify NUV --want-clean NUV

# 2d the lanes with the umbrella == serial solo spec (the exactness settings, as the lanes4 window): 1 user pipelined
#    (< 32K, 49K, a > 32K resume), the switch 1-2-3-4-2-1, HOLD 2 / 3 / 4 (overlap off in the lane drives, as there)
if [ $REF_REUSED = 0 ]; then
    step ref 300 1800 drv --mode solo --seq ref4 --config $CF/LN-OFF.json --label REF --extra="$EXACT" "${XENV[@]}" \
        --log $OUT/engine-ref.log --out $OUT/ref.json
fi
LX=(--env STRATA_SPEC_LANES_LOG=1 --env STRATA_SOLO_PIPELINE_LOG=1)
if step_ran ref; then
  step solo_lc 180 1200 drv --mode solo --seq m1a --config $CF/LN-NU.json --label NU-SOLO --extra="$EXACT" "${XENV[@]}" \
      --log $OUT/engine-solo_lc.log --out $OUT/solo_lc.json
  judge cmp_solo_lc ref,solo_lc "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/solo_lc.json --seq m1a \
      --what "1 user pipelined (umbrella)" --need-pipelined --need-resume
  # the lanes the umbrella's engine fitted (T0b's staging takes VRAM): as many as production's (the lanes4 boot at 28672)
  NL=$(python3 -c "import json,sys
try: print(int(json.load(open(sys.argv[1]))['info'].get('spec_lanes') or 0))
except Exception: print(0)" $OUT/solo_lc.json)
  NLP=$(python3 -c "import json,sys
try: print(int(json.load(open(sys.argv[1])).get('lanes') or 4))
except Exception: print(4)" $LANES4/boot_LC4_28.json)
  step_ran solo_lc && log "    [lanes] ==> lanes fitted with the umbrella: $NL (production's config fitted $NLP in the lanes4 window) $([ "$NL" -ge "$NLP" ] && echo PASS || echo FAIL)"
  [ "$NL" -ge 2 ] || NL=$NLP     # (solo_lc did not run: try the drives anyway, they fail visibly)
  step switch 180 1200 drv --mode switch --config $CF/LN-NU.json --label NU-SWITCH --extra="$EXACT" "${XENV[@]}" "${LX[@]}" \
      --env STRATA_SPEC_LANES_HOLD=2 "${NO_OV[@]}" --log $OUT/engine-switch.log --out $OUT/switch.json
  judge cmp_switch ref,switch "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/switch.json --what "switch 1-2-3-4-2-1 (umbrella)"
  step hold2 180 1200 drv --mode lanes --pairs g2s --pics "$PICS" --config $CF/LN-NU.json --label NU-HOLD2 --extra="$EXACT" \
      "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=2 "${NO_OV[@]}" --log $OUT/engine-hold2.log --out $OUT/hold2.json
  judge cmp_hold2 ref,hold2 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold2.json --pairs g2s \
      --what "2 lanes (HOLD=2, umbrella)" --hold --full
  if [ "$NL" -ge 3 ]; then
    step hold3 180 1200 drv --mode lanes --pairs g3 --config $CF/LN-NU.json --label NU-HOLD3 --extra="$EXACT" \
        "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=3 "${NO_OV[@]}" --log $OUT/engine-hold3.log --out $OUT/hold3.json
    judge cmp_hold3 ref,hold3 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold3.json --pairs g3 \
        --what "3 lanes (HOLD=3, umbrella)" --hold --full
  else noskip hold3 "the umbrella's engine fitted $NL lanes"; fi
  if [ "$NL" -ge 4 ]; then
    step hold4 180 1200 drv --mode lanes --pairs g4 --config $CF/LN-NU.json --label NU-HOLD4 --extra="$EXACT" \
        "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=4 "${NO_OV[@]}" --log $OUT/engine-hold4.log --out $OUT/hold4.json
    judge cmp_hold4 ref,hold4 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold4.json --pairs g4 \
        --what "4 lanes (HOLD=4, umbrella)" --hold --full
  else noskip hold4 "the umbrella's engine fitted $NL lanes"; fi
  LNE=(); for s in solo_lc switch hold2 hold3 hold4; do LNE+=($s=$OUT/engine-$s.log); done
  judge lanes_eng solo_lc "$PY" -I "$RT" --mode engstat --arms "${LNE[@]}" --want-clean solo_lc,switch,hold2,hold3,hold4
else
  for s in solo_lc switch hold2 hold3 hold4; do noskip $s "no serial solo reference"; done
fi

# 2e the late copy's fault path over HTTP (every late host copy fails: the checkpoint disarmed and dropped at the next
#    safe point; parks follow): no engine exit, no ERR, every reply identical to production's
step ov_NF 300 1500 ov_arm NF $CF/OV-NF.json
judge ov_cmp_nf ov_PROD,ov_NF "$PY" -I "$L/ovkv_ident.py" --compare $OUT/ov_PROD.json $OUT/ov_NF.json
judge ov_eng_nf ov_NF "$PY" -I "$RT" --mode engstat --arms NF=$OUT/engine-ov_NF.log --want-fault NF

# T0c must have taken its path somewhere (else its exactness above proves nothing about it): 'reads beside groups' > 0
# in an arm whose replies were compared above (the prefill drive's NU-pf is listed for information only: its 4-token
# replies are compared with nothing, so a read beside groups there proves no exactness)
judge nd_path "" "$PY" -I "$RT" --mode engstat --arms NU-pf=$OUT/engine-pf_NU.log NC-bt=$OUT/engine-bt_NC.log \
    NU-bt=$OUT/engine-bt_NU.log NUV-bt7=$OUT/engine-bt7_NUV.log NUV-ov=$OUT/engine-ov_NUV.log NF-ov=$OUT/engine-ov_NF.log \
    --want-beside-any NC-bt,NU-bt,NUV-bt7,NUV-ov,NF-ov

# ---- 3. speed over HTTP, production settings: production vs the umbrella, ABBA x2 -------------------------------------
# 3a 16 users, mixed reply lengths (MIX=128): identical replies, agg (no loss beyond 1.5 %), TTFT p50 / p95 / max
step spd16 900 2400 ab env LEVELS=16 REPS=2 ORDER=abba ROUNDS=2 MIX=128 ./ab_quick.sh rd16 $CF/SP-PROD.json $CF/SP-NU.json
judge spd16_ident spd16 "$PY" -I "$DRV" --mode abcheck --file $OUT/spd16.txt --what "16-user replies, umbrella vs production" --expect identical
judge spd16_agg spd16 "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag rd16 --arms SP-PROD,SP-NU --reps 2 --flat-users 16 --flat-tol 1.5
judge spd16_ttft spd16 "$PY" -I "$RT" --mode abttft --since-line "$SWEEP0" --tag rd16 --arms-list SP-PROD,SP-NU --reps 2
# 3b 1-4 users (the lanes must not regress): decode p50 per user, 1 user strictly identical, lane starts by lane count
step spdlow 1000 1800 ab env LEVELS=1,2,3,4 REPS=2 ORDER=abba ROUNDS=2 ./ab_quick.sh rdlow $CF/SP-PROD.json $CF/SP-NU.json
judge spdlow_ident spdlow "$PY" -I "$DRV" --mode abcheck --file $OUT/spdlow.txt \
    --what "1-4 user replies (strict at 1 user; INFO beyond: lanes / pump windows by timing)" --expect identical --strict-users 1
judge spdlow_dec spdlow "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag rdlow --arms SP-PROD,SP-NU --reps 2 \
    --dec-min "1:-1.5,2:-1.5,3:-2,4:-2"
judge spdlow_ttft spdlow "$PY" -I "$RT" --mode abttft --since-line "$SWEEP0" --tag rdlow --arms-list SP-PROD,SP-NU --reps 2
# (the -long run log holds the -short run's lines too: one engine start, run_strata_variant cuts from its 'engine
#  started' line both times - so only the -long logs)
judge spdlow_lanes spdlow "$PY" -I "$DRV" --mode lanelog --what "umbrella HTTP engines, 1-4 users" --want-sizes 2,3,4 \
    "$L/results/engine-ab-rdlow-SP-NU-r*-long.run.log"
# the umbrella's HTTP engines of the speed arms: every switch on at boot (a switch that turned itself off there, e.g.
# T0b's staging without the VRAM, would make the speed A/B one of the default path) - the -long logs hold the boot
judge spd16_on spd16 "$PY" -I "$RT" --mode engstat --arms "NU16=$L/results/engine-ab-rd16-SP-NU-r*-long.run.log" \
    --want-on NU16:WARM,LATE,NODRAIN
judge spdlow_on spdlow "$PY" -I "$RT" --mode engstat --arms "NULOW=$L/results/engine-ab-rdlow-SP-NU-r*-long.run.log" \
    --want-on NULOW:WARM,LATE,NODRAIN
for s in spd16 spdlow; do step_ran $s && grep -hE '^ *(short|long) +[0-9]+ users' "$OUT/$s.txt" | sed 's/^/    /' | tee -a "$OUT/window.log" >/dev/null; done

# ---- 1. zero-code A/B: --short-read 16 and 32 vs 64 (production's binary and config; INFO only, so after the
#      exactness and speed steps this window is for: when time runs short, these arms are what is skipped) ------------
for arm in SR64 SR32 SR16; do step sr_$arm 420 1200 sr_arm srd $CF/RD-$arm.json; done
judge srab sr_SR64 "$PY" -I "$RT" --mode srab --tag srd --arms-list RD-SR64,RD-SR32,RD-SR16 --since-line "$SWEEP0" \
    --mt $OUT/multiturn.jsonl

# ---- 4. production back, unchanged ------------------------------------------------------------------------------------
# (the abort trap stays armed through bench_down: a signal there still restarts production; then signals are ignored
#  until production is up)
bench_down end
trap '' INT TERM HUP
start_prod || { log "production start failed: one more try"; systemctl --user restart strata.service; start_prod; } \
    || log "PRODUCTION DID NOT COME BACK - check strata.service now"
trap - INT TERM HUP
P=$(systemctl --user show -p MainPID --value strata.service)
PE=$(for c in $(pgrep -P "$P" 2>/dev/null); do readlink /proc/$c/exe; done | tr '\n' ' ')
log "production engine: $PE"
NOWCFG=$(sha256sum "$(readlink -f "$C/production.json")" | cut -c1-16)
log "    [prod] ==> production.json $(readlink $C/production.json) sha256 $NOWCFG (at the start $PRODCFG_SHA), engine $(basename "$PEXE") running: $([ "$NOWCFG" = "$PRODCFG_SHA" ] && echo "$PE" | grep -q "$(basename "$PEXE")" && echo PASS || echo FAIL)"
rm -rf "$TIER"
log "window done after $(( ($(date +%s) - T0) / 60 )) min"
grep -hE 'PASS|FAIL|SKIP' "$OUT/window.log" | grep -v 'grep' | tail -120
