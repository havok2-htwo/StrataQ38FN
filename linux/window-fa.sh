#!/usr/bin/env bash
# GPU window (~60 min of production outage, cap WINDOW_S 4200 s): fair reads stage 1 (engine STRATA_READ_YIELD=1 +
# server STRATA_YIELD_SHORT) alone and together with the admission order by estimated new tokens (STRATA_ADMIT_BY_NEW=1),
# in the agent mix - the one place stage 1 was not measured yet (TESTLOG #88: stage 2 failed there).
#   PROD = production as it runs (production.json read now, server Strata-143-prod, the drop-in's STRATA_* env)
#   F    = binary strata-g3-1441j-reads2 (sha256 b403479110173fcf, the one #88 measured stage 1 with; STRATA_READS unset)
#          + STRATA_READ_YIELD=1, server Strata-ll8-fairadmit (ll6-reads serve + ll6-admit, merge 2cf7434d + review fix 90f73293) with
#          STRATA_YIELD_SHORT=$YS STRATA_YIELDS_MAX=$YM
#   FE   = F + STRATA_ADMIT_BY_NEW=1 on the server
# Steps: 1 agent mix PROD F FE FE F PROD (one server start each: bench_multiturn 16x3 greedy --doc $MT_DOC,
#   bench_agentic greedy, 16 users short + 1.2K), judged F vs PROD and FE vs PROD (multiturn replies per (uid, turn)
#   and 16-user replies identical; follow-up / agentic TTFT)                                                    ~45 min
#   2 the user-visible win with the merged server: scen win reps 1,2 for FE (w1: shorts during a cold 125K read; w2:
#   a burst of 16), compared with #88's PROD and FAIR runs (results/reads2)                                       ~8 min
#   3 production restarted unchanged, health checked.
# Test servers only on 127.0.0.1:8091, tier dirs under /media/raid0/strata-kvspill-bench/fa, results in results/fa.
#   run: cd /media/raid0/StrataQ38FN/linux && ./window-fa.sh
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/fa; CF=$OUT/cfg
URL=http://127.0.0.1:8090; TPORT=8091; PY=$R/Strata/.venv/bin/python; DRV=$L/lanes_stdin.py; RT=$L/reads_tools.py
FT=$L/fair_tools.py
WT=$R/Strata-ll8-fairadmit; NEW=$R/Strata/engine/strata-g3-1441j-reads2
NEW_SHA=b403479110173fcf; WANT_COMMIT=90f73293
YS=${YS:-2048}                                              # STRATA_YIELD_SHORT of the FAIR arms
# STRATA_YIELDS_MAX of the FAIR arms: the server's default (2 stops per read) lets only the first two of w1's four
# spaced shorts past the 125K read (the 3rd and 4th wait it out) and x1's third short; 4 = one stop per short there
# (YM=2: the server's default as it is)
YM=${YM:-4}
RN=${RN:-2}                                                 # STRATA_READS of the READS arms
BGMIN=${BGMIN:-8192}                                        # STRATA_READS_BG_MIN of the READS servers
PROD_EXE_WANT=${PROD_EXE_WANT:-strata-g3-1441f-lanes4}       # what production must run for this window's comparisons
PRODSRC=$(readlink -f "$C/production.json")                 # what production runs now (its config is only read)
SRV=$R/Strata-143-prod                                      # production's server checkout
TIER=/media/raid0/strata-kvspill-bench/fa
EXACT="--pcie-frac 0 --adapt-every 0 --suffix-draft 0"
XENV=(--env STRATA_IQ_MT_MIN=1 --env STRATA_BATCH_DECODE_SHARE=0 --env STRATA_DECODE_TIMING=1)
NO_OV=(--drop-env STRATA_PIPE_OVERLAP --drop-env STRATA_PIPE_OVERLAP_KVS)   # (as the lanes4 / reads windows' lane drives)
MIX_BGMIN=${MIX_BGMIN:-4096}                                # STRATA_READS_BG_MIN of the agent mix's READS arm
MT_DOC=${MT_DOC:-8000}                                      # bench_multiturn's first-turn document (as TESTLOG #87)
WINDOW_S=${WINDOW_S:-4200}; RESERVE=480; T0=$(date +%s)
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
noskip() { log "step $1: SKIP ($2)"; echo skip > "$OUT/rc_$1"; }
# judge <name> <steps> <cmd...>: a check without the GPU of what <steps> produced (a comma list: every one must have
# run, else SKIP - no verdict from stale files; empty: no gate)
step_ran() { [ -f "$OUT/rc_$1" ] && [ "$(cat "$OUT/rc_$1")" != skip ]; }
judge() { local name=$1 from=$2 s; shift 2
    for s in ${from//,/ }; do step_ran "$s" || { log "    [$name] SKIP (step $s skipped)"; return 0; }; done
    timeout 600 "$@" > "$OUT/$name.txt" 2>&1 < /dev/null &
    wait $!
    grep -aE '^==>|^    ' "$OUT/$name.txt" | cut -c1-700 | sed "s/^/    [$name] /" | tee -a "$OUT/window.log" >/dev/null
    grep -qaE '^==>' "$OUT/$name.txt" || log "    [$name] no verdict (see $OUT/$name.txt) FAIL"; }
scoped() { systemctl --user stop strata-bench.scope 2>/dev/null; systemctl --user reset-failed strata-bench.scope 2>/dev/null
    systemd-run --user --scope -q --unit=strata-bench -p MemoryMax=200G -p MemorySwapMax=0 "$@"; }
drv() { scoped "$PY" -I "$DRV" "$@"; }
tup() { curl -s -m 3 "http://127.0.0.1:$TPORT/health" | grep -q '"loaded": *true'; }
# runcfg <NAME> <run>: a copy of $CF/NAME.json for one server start - its own engine log $OUT/engine-<run>.log and a
# fresh, empty tier dir (STRATA_CONV_TIER_WARM would otherwise preload an earlier start's conversations: every read of
# the scenarios must be cold); the server sidecar (.srv: SERVER_DIR + the server env) goes with it
runcfg() { local d=$TIER/$2 f=$CF/run-$2.json; rm -rf "$d"; mkdir -p "$d"; rm -f "$OUT/engine-$2.log"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; c['log']=sys.argv[3]; json.dump(c,open(sys.argv[4],'w'),indent=1)" \
        "$CF/$1.json" "$d" "$OUT/engine-$2.log" "$f" && cp "$CF/$1.srv" "$CF/run-$2.srv" && echo "$f"; }
# srv <cfg> <cmd...>: the command with the config's server checkout and server env, on 127.0.0.1:$TPORT
srv() { local s=${1%.json}.srv; shift; ( cd "$L" && env $(cat "$s") PORT=$TPORT BIND=127.0.0.1 "$@" ); }
# up <label> <cfg>: a test server with a 1-request warm-up (KEEP=1 leaves it running)
up() { srv "$2" env KEEP=1 ./run_strata_variant.sh "$1" "$2" 1 --rounds 1 --max-tok 16 --prompt-tok 100 --temperature 0 2>&1 | tail -3
    tup || { echo "==> $1: the test server did not come up FAIL"; systemctl --user stop strata-bench.scope; return 1; }
    echo "server $(head -1 "$L/results/$1.txt" 2>/dev/null | cut -c1-250)"; }
# rss_peak: the bench scope's peak memory (cgroup memory.peak: engine + server; else the largest VmHWM in it), GiB
rss_peak() { local cg; cg=$(systemctl --user show -p ControlGroup --value strata-bench.scope 2>/dev/null)
    if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ]; then
        awk '{printf "%.2f GiB (memory.peak)", $1 / 1073741824}' "/sys/fs/cgroup$cg/memory.peak"; return; fi
    local m=0 p k; for p in $(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null); do
        k=$(awk '/^VmHWM/ {print $2}' /proc/$p/status 2>/dev/null); [ -n "$k" ] && [ "$k" -gt "$m" ] && m=$k; done
    awk -v k="$m" 'BEGIN {printf "%.2f GiB (VmHWM)", k / 1048576}'; }
# scen_arm <run> <NAME> <scenario> <rep> [rounds]: one server start, one fair_tools scenario, stop
scen_arm() { local run=$1 cfg rc; cfg=$(runcfg "$2" "$run") || return 1
    up "fa-$run" "$cfg" || return 1
    timeout -s INT 1300 "$PY" -I "$FT" --mode scen --config "$cfg" --url "http://127.0.0.1:$TPORT" --scenario "$3" \
        --rep "$4" --label "$run" --out "$OUT/$run.json" ${5:+--rounds "$5"}; rc=$?
    echo "==> rss $run: peak $(rss_peak) INFO"
    sleep 2; systemctl --user stop strata-bench.scope; return $rc; }
# scen_reps <run> <NAME> <scenario> <reps>: one server start, the scenario once per repetition (comma list: k ->
# $OUT/<run>_k.json; each repetition has its own prompts, so every read stays cold), stop
scen_reps() { local run=$1 cfg rc=0 k; cfg=$(runcfg "$2" "$run") || return 1
    up "fa-$run" "$cfg" || return 1
    for k in ${4//,/ }; do
        timeout -s INT 1300 "$PY" -I "$FT" --mode scen --config "$cfg" --url "http://127.0.0.1:$TPORT" --scenario "$3" \
            --rep "$k" --label "${run}_$k" --out "$OUT/${run}_$k.json" || rc=1
    done
    echo "==> rss $run: peak $(rss_peak) INFO"
    sleep 2; systemctl --user stop strata-bench.scope; return $rc; }
# mix_arm <NAME> <rep>: one server start: bench_multiturn 16 x 3 (greedy), bench_agentic (greedy), then 16 users short
# and 1.2K on the same server (labels ab-mix-<NAME>-r<rep>-short|long, as abq's, so absum / abcheck / sweep read them)
mix_arm() { local n=$1 rep=$2 cfg rc=0 lab=ab-fam-$1-r$2 ml=mix-$1-r$2 u=http://127.0.0.1:$TPORT
    cfg=$(runcfg "$n" "mix-$n-r$rep") || return 1
    up "fa-$ml" "$cfg" || return 1
    timeout -s INT 900 "$PY" -I "$L/bench_multiturn.py" --url "$u" --users 16 --turns 3 --doc "$MT_DOC" --temperature 0 \
        --label "$ml-mt" --out "$OUT/multiturn.jsonl" > "$OUT/$ml-mt.txt" 2>&1 || rc=1
    echo "==> $ml multiturn: $(tail -1 "$OUT/$ml-mt.txt" | cut -c1-200) INFO"
    timeout -s INT 1500 "$PY" -I "$L/bench_agentic.py" --url "$u" --temperature 0 --label "$ml-agentic" --out "$OUT/agentic.jsonl" \
        > "$OUT/$ml-agentic.txt" 2>&1 || rc=1
    echo "==> $ml agentic: $(tail -1 "$OUT/$ml-agentic.txt" | cut -c1-200) INFO"
    if tup; then
        srv "$cfg" env ATTACH=1 KEEP=1 ./run_strata_variant.sh "$lab-short" "$cfg" 16 --rounds 2 --max-tok 512 \
            --prompt-tok 100 --temperature 0 --max-tok-min 128 2>&1 | tail -3
        srv "$cfg" env ATTACH=1 KEEP=1 ./run_strata_variant.sh "$lab-long" "$cfg" 16 --rounds 2 --max-tok 512 \
            --temperature 0 --warmup 0 --max-tok-min 128 2>&1 | tail -2
    else echo "==> $ml: the server is gone after the agent benches FAIL"; rc=1; fi
    echo "==> rss $ml: peak $(rss_peak) INFO"
    sleep 2; systemctl --user stop strata-bench.scope; return $rc; }
# abq <tag> <reps> <levels> <NAME...>: ab_quick.sh with every arm on its own server checkout (ORDER abba, MIX=128,
# ROUNDS=2, 16-user short then 1.2K prompts on the same server; labels ab-<tag>-<NAME>-r<k>-short|long as ab_quick's)
abq() { local tag=$1 reps=$2 levels=$3; shift 3; local rep k cfg lab arms
    for rep in $(seq 1 "$reps"); do
        arms=("$@"); if [ $(( rep % 2 )) -eq 0 ]; then arms=(); for (( k=$#; k>=1; k-- )); do arms+=("${!k}"); done; fi
        for n in "${arms[@]}"; do
            lab=ab-$tag-$n; [ "$reps" -gt 1 ] && lab=$lab-r$rep
            cfg=$(runcfg "$n" "$tag-$n-r$rep") || return 1
            srv "$cfg" env KEEP=1 ./run_strata_variant.sh "$lab-short" "$cfg" "$levels" --rounds 2 --max-tok 512 \
                --prompt-tok 100 --temperature 0 --max-tok-min 128 2>&1 | tail -3
            if ! tup; then echo "==> $lab: the test server did not come up FAIL"; systemctl --user stop strata-bench.scope; continue; fi
            srv "$cfg" env ATTACH=1 KEEP=1 ./run_strata_variant.sh "$lab-long" "$cfg" "$levels" --rounds 2 --max-tok 512 \
                --temperature 0 --warmup 0 --max-tok-min 128 2>&1 | tail -2
            systemctl --user stop strata-bench.scope
        done
    done; }
LANES_TIER_ROOT=$TIER
export -f scoped drv tup runcfg srv up scen_arm scen_reps mix_arm abq rss_peak
export PY DRV FT L CF OUT TIER TPORT LANES_TIER_ROOT MT_DOC   # (step runs them under timeout through bash -c)
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }
cfg_exe() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" "$1"; }

# ---- prep (no GPU; production still serving) ------------------------------------------------------------------------
[ -x "$NEW" ] || { echo "missing $NEW"; exit 1; }
[ "$(sha256sum "$NEW" | cut -c1-16)" = "$NEW_SHA" ] || { echo "$NEW is not the binary #88 measured ($NEW_SHA)"; exit 1; }
git -C "$WT" merge-base --is-ancestor $WANT_COMMIT HEAD || { echo "$WT HEAD lacks $WANT_COMMIT (the merge)"; exit 1; }
git -C "$WT" diff --quiet HEAD -- serve || { echo "$WT has uncommitted changes under serve/: the server is not HEAD"; exit 1; }
grep -q 'STRATA_ADMIT_BY_NEW' "$WT/serve/server.py" && grep -q 'STRATA_YIELD_SHORT' "$WT/serve/server.py" \
    || { echo "$WT/serve/server.py lacks STRATA_ADMIT_BY_NEW or STRATA_YIELD_SHORT"; exit 1; }
"$PY" -m py_compile "$WT/serve/server.py" || { echo "$WT/serve/server.py does not compile"; exit 1; }
for f in "$DRV" "$RT" "$FT" "$L/run_strata_variant.sh" "$L/compare_replies.py" "$L/ovkv_ident.py" \
         "$R/Strata-1441/tools/batch_test16.py" "$SRV/serve/server.py" "$PRODSRC" "$L/bench_multiturn.py" \
         "$L/bench_agentic.py" "$L/bench_concurrency.py" "$L/vision_check.py"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
for f in "$DRV" "$RT" "$FT" "$L/run_strata_variant.sh"; do [ -x "$f" ] || { echo "$f is not executable (chmod +x)"; exit 1; }; done
for f in "$DRV" "$RT" "$FT" "$L/bench_multiturn.py" "$L/bench_agentic.py"; do "$PY" -I "$f" --help > /dev/null || { echo "$f does not load"; exit 1; }; done
"$PY" -c 'import PIL' || { echo "the venv has no PIL (the vis scenario's pictures)"; exit 1; }
curl -s -m 2 "http://127.0.0.1:$TPORT/health" > /dev/null && { echo "port $TPORT already serves something: the test servers need it"; exit 1; }
# (scoped / run_strata_variant stop strata-bench.scope unconditionally: another window's bench must not be running)
systemctl --user is-active -q strata-bench.scope && { echo "strata-bench.scope is active: another bench is running"; exit 1; }
PEXE=$(cfg_exe "$PRODSRC"); [ -x "$PEXE" ] || { echo "production's exe $PEXE missing"; exit 1; }
python3 - "$PRODSRC" "$PROD_EXE_WANT" <<'PYEOF' || exit 1
import json, os, sys
c = json.load(open(sys.argv[1])); e = c["env"]; a = c["args"]
want = {"STRATA_SPEC_LANES": "4", "STRATA_SOLO_PIPELINE": "1", "STRATA_PIPE_OVERLAP": "1", "STRATA_PIPE_OVERLAP_KVS": "1",
        "STRATA_NAN_GUARD": "0", "STRATA_CKPT_POOL": "1"}
bad = {k: e.get(k) for k, v in want.items() if e.get(k) != v}
kv = a[a.index("--kv-resident") + 1] if "--kv-resident" in a else None
pf = a[a.index("--prefill") + 1] if "--prefill" in a else None
extra = [k for k in e if k.startswith("STRATA_READ") or k.startswith("STRATA_YIELD")]
why = []
if bad: why.append(f"env differs: {bad}")
if kv != "28672": why.append(f"--kv-resident {kv}")
if pf != "512": why.append(f"--prefill {pf} (STRATA_READ_YIELD_TEST_AT's chunks are 512 tokens here)")
if extra: why.append(f"already has {extra}")
if int(c.get("parallel") or 0) != 16: why.append(f"parallel {c.get('parallel')}")
if c.get("layer_split") != "12,25,37": why.append(f"layer_split {c.get('layer_split')}")
if os.path.basename(c["exe"]) != sys.argv[2]: why.append(f"exe {os.path.basename(c['exe'])} (want {sys.argv[2]})")
if why:
    print(f"production's config {sys.argv[1]} is not the one this window is written for: {'; '.join(why)}"); sys.exit(1)
PYEOF
[ "$(readlink -f "$PEXE")" != "$(readlink -f "$NEW")" ] || { echo "production already runs $NEW: nothing to compare against"; exit 1; }
# every GPU process now is production's engine (free_gpus later kills what is left of $R/Strata* only)
for p in $(gpu_apps); do [ "$(readlink -f /proc/$p/exe 2>/dev/null)" = "$(readlink -f "$PEXE")" ] \
    || { echo "GPU process $p ($(readlink -f /proc/$p/exe 2>/dev/null)) is not production's engine: not starting"; exit 1; }; done
systemctl --user is-active -q strata.service || { echo "strata.service is not active: start it first"; exit 1; }
# production's server env (the drop-in's STRATA_* keys, read now): every test server gets it, the FAIR arms add
# STRATA_YIELD_SHORT
SRVENV=$(systemctl --user show strata.service -p Environment --value | tr ' ' '\n' | grep '^STRATA_' | tr '\n' ' ')
case "$SRVENV" in *STRATA_YIELD*|*STRATA_READ*|*STRATA_SERVER_READS*|*STRATA_ADMIT_BY_NEW*) echo "production's server env already has $SRVENV"; exit 1 ;; esac
PROD_SHA=$(sha256sum "$PEXE" | cut -c1-16); PRODCFG_SHA=$(sha256sum "$PRODSRC" | cut -c1-16)
TOKCHK=$(taskset -c 0-63 nice -n 15 "$PY" -I "$FT" --mode tokcheck --config "$PRODSRC" --short "$YS" --reps-list 1,2 --two 2>&1) \
    || { echo "$TOKCHK"; echo "the scenarios do not fit (see above)"; exit 1; }
RUNLOGS=("$L"/results/engine-ab-fam-*.run.log "$L"/results/engine-fa-*.run.log)
if [ -e "$OUT" ]; then PREV=$OUT.prev-$(date +%Y%m%d-%H%M%S); mv "$OUT" "$PREV" || { echo "cannot move $OUT aside"; exit 1; }
    for f in "${RUNLOGS[@]}"; do [ -e "$f" ] && mv "$f" "$PREV"/; done; fi
for f in "${RUNLOGS[@]}"; do rm -f "$f"; done
rm -rf "$TIER"; mkdir -p "$OUT" "$CF" "$TIER"
echo "$TOKCHK" | sed 's/^/    [tokcheck] /' | tee -a "$OUT/window.log" >/dev/null
SWEEP0=$(cat "$L/results/sweep.jsonl" 2>/dev/null | wc -l)
log "window prep: $(basename $NEW) sha256 $(sha256sum $NEW | cut -c1-16) = $WT $(git -C $WT rev-parse --short HEAD) ($(git -C $WT log -1 --format=%s HEAD | cut -c1-80)); production: $(basename $PRODSRC) sha256 $PRODCFG_SHA, exe $(basename $PEXE) sha256 $PROD_SHA, server $SRV @$(git -C $SRV rev-parse --short HEAD), server env: $SRVENV; F/FE server $WT/serve, FAIR STRATA_YIELD_SHORT=$YS STRATA_YIELDS_MAX=$YM, FE + STRATA_ADMIT_BY_NEW=1; bench_multiturn --doc $MT_DOC"

# configs, all from production's config: exe, then the arm's changes (K=V sets an env key, -K drops one); the sidecar
# <name>.srv: the server checkout + the server env (prod: Strata-143-prod; new0: Strata-ll6-reads, env as production's;
# fair: Strata-ll6-reads + STRATA_YIELD_SHORT)
mkcfg() {   # $1 name, $2 exe, $3 prod|new0|fair, then changes ...
    local name=$1 exe=$2 kind=$3; shift 3
    python3 - "$PRODSRC" "$CF/$name.json" "$exe" "$@" <<'PYEOF' || exit 1
import json, sys
src, dst, exe = sys.argv[1:4]
c = json.load(open(src)); c["exe"] = exe; e = c["env"]
for kv in sys.argv[4:]:
    if kv.startswith("-"):
        e.pop(kv[1:], None)
    else:
        k, v = kv.split("=", 1); e[k] = v
json.dump(c, open(dst, "w"), indent=1)
PYEOF
    case $kind in
        prod) echo "SERVER_DIR=$SRV $SRVENV" > "$CF/$name.srv" ;;
        new0) echo "SERVER_DIR=$WT $SRVENV" > "$CF/$name.srv" ;;
        fair) echo "SERVER_DIR=$WT $SRVENV STRATA_YIELD_SHORT=$YS STRATA_YIELDS_MAX=$YM" > "$CF/$name.srv" ;;
        fairadmit) echo "SERVER_DIR=$WT $SRVENV STRATA_YIELD_SHORT=$YS STRATA_YIELDS_MAX=$YM STRATA_ADMIT_BY_NEW=1" > "$CF/$name.srv" ;;
    esac; }
FY=(STRATA_READ_YIELD=1)
LOGS=(STRATA_SPEC_LANES_LOG=1 STRATA_SOLO_PIPELINE_LOG=1)
mkcfg SP-PROD "$PEXE" prod "${LOGS[@]}"                                  # production as it runs (+ lane log lines)
mkcfg MX-F    "$NEW"  fair "${FY[@]}" "${LOGS[@]}"                       # stage 1
mkcfg MX-FE   "$NEW"  fairadmit "${FY[@]}" "${LOGS[@]}"                  # stage 1 + order by new tokens
mkcfg W-FE    "$NEW"  fairadmit "${FY[@]}"                               # the win scenarios, as #88's W-FAIR
# a copy of a config with a fresh, empty tier dir of its own (batch_test16 / the lane drives)
tiercfg() { local d=$TIER/$2; rm -rf "$d"; mkdir -p "$d"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; c['log']=sys.argv[4]; json.dump(c,open(sys.argv[3],'w'),indent=1)" \
        "$1" "$d" "$CF/$2.json" "$OUT/engine-$2.log"; echo "$CF/$2.json"; }

[ -n "${DRY:-}" ] && { log "dry run: prep and configs ok, production untouched"; ls "$CF"; exit 0; }
for i in $(seq 180); do curl -s -m 3 $URL/health | grep -q '"loaded": *true' && break; sleep 5; done
curl -s -m 3 $URL/health | grep -q '"loaded": *true' || { echo "production does not report loaded: not stopping it"; exit 1; }
log "window start ($(( WINDOW_S / 60 )) min cap): production loaded; stopping strata.service"
trap '' INT TERM HUP                             # (not while it stops)
systemctl --user stop strata.service
STOPPED=1; trap 'abort signal' INT TERM HUP     # from here on an abort ends the bench engines and restarts production
T0=$(date +%s)                                  # (WINDOW_S: production's outage, from its stop to its restart)
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || abort "GPUs not free"

# ---- 1. agent mix + 16 users: PROD F FE FE F PROD ----------------------------------------------------------------------
step mix_PROD_1 450 1500 mix_arm SP-PROD 1
step mix_F_1 450 1500 mix_arm MX-F 1
step mix_FE_1 450 1500 mix_arm MX-FE 1
step mix_FE_2 450 1500 mix_arm MX-FE 2
step mix_F_2 450 1500 mix_arm MX-F 2
step mix_PROD_2 450 1500 mix_arm SP-PROD 2
for A in F FE; do
    judge mix_$A mix_PROD_1,mix_${A}_1 "$PY" -I "$FT" --mode mix --dir $OUT --arms-list SP-PROD,MX-$A \
        --reps $(step_ran mix_PROD_2 && step_ran mix_${A}_2 && echo 2 || echo 1) --gain 1.10
    judge mix_sum_$A mix_PROD_1,mix_${A}_1,mix_${A}_2,mix_PROD_2 "$PY" -I "$FT" --mode absum --tag fam --reps 2 --arms-list SP-PROD,MX-$A --since-line "$SWEEP0"
    judge mix_ident_$A mix_PROD_1,mix_${A}_1,mix_${A}_2,mix_PROD_2 "$PY" -I "$DRV" --mode abcheck --file $OUT/mix_sum_$A.txt --what "16-user replies, MX-$A vs production" --expect identical
    judge mix_ttft_$A mix_PROD_1,mix_${A}_1,mix_${A}_2,mix_PROD_2 "$PY" -I "$RT" --mode abttft --since-line "$SWEEP0" --tag fam \
        --arms-list SP-PROD,MX-$A --reps 2
done
for f in "$OUT"/engine-mix-MX-F*-r*.log; do [ -f "$f" ] && log "    [yields] ==> $(basename $f): $(LC_ALL=C grep -ac 'gives way at' "$f") give-ways, $(grep -c 'YIELDNO' "$f") YIELDNO, $(grep -c '^ERR\| ERR ' "$f") ERR INFO"; done
for f in "$OUT"/fa-mix-MX-FE*.txt "$L"/results/server-fa-mix-MX-FE*.out; do [ -f "$f" ] && log "    [admit] ==> $(basename $f): $(grep -c '\[strata\] admit: total' "$f") admit lines INFO"; done

# ---- 2. the win with the merged server (vs #88's PROD / FAIR runs) -----------------------------------------------------
step w_FE 480 1800 scen_reps w_FE W-FE win 1,2
P88=$L/results/reads2
judge win_FE w_FE "$PY" -I "$FT" --mode win --arms "PROD=$P88/w_PROD_[12].json" "FE=$OUT/w_FE_[12].json"
judge win_FE_vs_FAIR w_FE "$PY" -I "$FT" --mode win --gain 1.0 --long-tol 0.05 --arms "FAIR=$P88/w_FAIR_[12].json" "FE=$OUT/w_FE_[12].json"
for k in 1 2; do
    judge w_cmp_$k w_FE "$PY" -I "$FT" --mode cmp --a $P88/w_PROD_$k.json --b $OUT/w_FE_$k.json --what "win rep $k (greedy): FE vs #88 production"
done
for s in mix_sum_F mix_sum_FE; do [ -f "$OUT/$s.txt" ] && sed 's/^/    /' "$OUT/$s.txt" | tee -a "$OUT/window.log" >/dev/null; done

# ---- 3. production back, unchanged ------------------------------------------------------------------------------------
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
grep -hE 'PASS|FAIL|SKIP' "$OUT/window.log" | grep -v 'grep' | tail -150

