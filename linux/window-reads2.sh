#!/usr/bin/env bash
# GPU window for branch ll7-reads2 (Strata-ll7-reads2, HEAD >= 28de225d): stage 2 step A - background prompt reads
# (engine STRATA_READS=N, server STRATA_SERVER_READS=1 from Strata-ll7-reads2/serve): a long admission's batched parts are
# read into a reserved top-row slot between the engine's lines, beside the decoding groups, up to N held at once (one
# reading at a time, round-robin by STRATA_READS_QUANTUM chunks), and each finishes on the main sessions and the main
# verifier (stage 1's pin).  Against production as it runs now (production.json read at run time: strata-g3-1441f-lanes4,
# overlap + KVS, solo pipeline, STRATA_SPEC_LANES=4, --kv-resident 28672, split 12,25,37, 16 slots, server Strata-143-prod
# with the drop-in's STRATA_* env; the window refuses a production config of another shape) and against stage 1 (the
# same binary with STRATA_READ_YIELD=1 + STRATA_YIELD_SHORT).  Binary: strata-g3-1441j-reads2 (a copy of
# Strata-ll7-reads2/build/strata).
# Steps ~2 h (hard cap WINDOW_S, default 7800 s = 2 h 10 min of production outage, counted from its stop, incl. its restart): a step that no longer
# fits is skipped (logged), steps are ordered by value; the last ones (marked "if time") run only when the earlier
# ones were faster than expected.  Production is stopped for the window (only once it reports
# loaded) and started again unchanged (production.json, the unit and its drop-in are never written), also on abort (the
# trap is armed once it is stopped; signals are ignored while it stops and while it starts again).  Every test engine
# logs into results/reads2 (never production's log) and spills into /media/raid0/strata-kvspill-bench/reads2/<run> (a
# fresh dir per server start, never production's tier dir); every test HTTP server listens on 127.0.0.1:8091 (never
# production's port 8090).  An existing results/reads2 moves to results/reads2.prev-<time>.
#   run:      cd /media/raid0/StrataQ38FN/linux && ./window-reads2.sh
#   verdicts: grep -E 'PASS|FAIL|SKIP' results/reads2/window.log
# Before the window (no GPU, production serving): every scenario prompt measured with the tokenizer (fair_tools.py
# tokcheck --two: each request incl. its output and history < 262,144), the server checkout clean and compiling.
# Steps (PASS/FAIL/INFO lines, an expected time each; in this order):
#   1 exactness over HTTP, concurrent background reads vs production's sequential reads (greedy + seeded T=0.7,
#     STRATA_SLOT_PARK_VERIFY=1 + STRATA_TRACE=1 in every arm; follow-up turns resume each prompt's slot / cached state,
#     so a slot state that differs gives another reply): fair_tools scen exact,vis (x0-x3 reads beside decoding streams
#     and shorts; v1/v2 pictures arriving during a background read and a long read with a picture, which reads
#     serially) - production vs READS N=2, READS with a pause every 3 chunks and a quantum of 2 (every pause / restart
#     path); every reply identical, reads went to the background and all came back ready, every finish pinned, park
#     verify identical, no ERR / watchdog / pad guard                                                      ~21 min
#   2 concurrency: fair_tools scen two (two and three cold long reads at once beside 2-3 streams, shorts between):
#     production vs READS N=4 - replies identical, background reads held at once                          ~10 min
#   3 the user-visible win, one server start per arm, reps 1 and 2 each (w1: a cold ~125K read beside 2 streams, 4
#     shorts at +3/6/9/12 s; w2: a cold ~105K read, a burst of 16 shorts at +4 s: time to all 16 first tokens):
#     PROD, READS, FAIR (stage 1) - shorts' TTFT and the long read's total; the plan's kill rule - READS ships if its
#     w1/w2 TTFT is at least as good as FAIR's and its long read not slower                                ~24 min
#   4 agent mix + 16 users, ABBA (PROD READS READS PROD, one server start each): bench_multiturn 16 users x 3 turns
#     greedy (follow-up TTFT, reply identity per (uid, turn)), bench_agentic (TTFT per role), then 16 users short +
#     1.2K (bench_concurrency, MIX=128: reply identity, agg within 1.5 %, TTFT); the READS arm with
#     STRATA_READS_BG_MIN=$MIX_BGMIN so the 8K first turns read in the background                         ~30 min
#   5 the default path: batch_test16 dumps (greedy, a cold 40K read beside 15 decoding) production vs the new binary
#     env unset, boot VRAM / experts; over HTTP (new binary + new server, env unset) scen exact,vis vs production ~12 min
#   6 1-4 users, PROD then READS (decode p50 per user, 1 user strictly identical; lanes start as before)  ~8 min
#   7 lone conversations and the decode after a background read (review R2-1/R2-2): fair_tools scen solo, production
#     vs READS with STRATA_READS_SOLO=1 - replies identical, decode tok/s and draft acceptance within 10 %, 2 background
#     reads per run, peak RSS                                                                              ~14 min
#   8 if time: the gate refused (STRATA_PIPE_OVERLAP_KVS dropped: drained turns beside 4 streams, x1), READS N=2 on
#     scen two, READS N=1 on scen exact,vis, batch_test16 with STRATA_READS=2 (no server: only the INFO line differs)
#   9 production restarted unchanged, health checked
# Not here (engine-level, no server: bg= is never sent there): the lane drives (STRATA_READS changes nothing without a
# server that sends bg=), STRATA_READS_FORCE_BG (its BREAD needs a driver that sends the finish).
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/reads2; CF=$OUT/cfg
URL=http://127.0.0.1:8090; TPORT=8091; PY=$R/Strata/.venv/bin/python; DRV=$L/lanes_stdin.py; RT=$L/reads_tools.py
FT=$L/fair_tools.py
WT=$R/Strata-ll7-reads2; B=$WT/build/strata; NEW=$R/Strata/engine/strata-g3-1441j-reads2
WANT_COMMIT=28de225d
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
TIER=/media/raid0/strata-kvspill-bench/reads2
EXACT="--pcie-frac 0 --adapt-every 0 --suffix-draft 0"
XENV=(--env STRATA_IQ_MT_MIN=1 --env STRATA_BATCH_DECODE_SHARE=0 --env STRATA_DECODE_TIMING=1)
NO_OV=(--drop-env STRATA_PIPE_OVERLAP --drop-env STRATA_PIPE_OVERLAP_KVS)   # (as the lanes4 / reads windows' lane drives)
MIX_BGMIN=${MIX_BGMIN:-4096}                                # STRATA_READS_BG_MIN of the agent mix's READS arm
MT_DOC=${MT_DOC:-8000}                                      # bench_multiturn's first-turn document (as TESTLOG #87)
WINDOW_S=${WINDOW_S:-7800}; RESERVE=480; T0=$(date +%s)
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
    up "reads2-$run" "$cfg" || return 1
    timeout -s INT 1300 "$PY" -I "$FT" --mode scen --config "$cfg" --url "http://127.0.0.1:$TPORT" --scenario "$3" \
        --rep "$4" --label "$run" --out "$OUT/$run.json" ${5:+--rounds "$5"}; rc=$?
    echo "==> rss $run: peak $(rss_peak) INFO"
    sleep 2; systemctl --user stop strata-bench.scope; return $rc; }
# scen_reps <run> <NAME> <scenario> <reps>: one server start, the scenario once per repetition (comma list: k ->
# $OUT/<run>_k.json; each repetition has its own prompts, so every read stays cold), stop
scen_reps() { local run=$1 cfg rc=0 k; cfg=$(runcfg "$2" "$run") || return 1
    up "reads2-$run" "$cfg" || return 1
    for k in ${4//,/ }; do
        timeout -s INT 1300 "$PY" -I "$FT" --mode scen --config "$cfg" --url "http://127.0.0.1:$TPORT" --scenario "$3" \
            --rep "$k" --label "${run}_$k" --out "$OUT/${run}_$k.json" || rc=1
    done
    echo "==> rss $run: peak $(rss_peak) INFO"
    sleep 2; systemctl --user stop strata-bench.scope; return $rc; }
# mix_arm <NAME> <rep>: one server start: bench_multiturn 16 x 3 (greedy), bench_agentic (greedy), then 16 users short
# and 1.2K on the same server (labels ab-mix-<NAME>-r<rep>-short|long, as abq's, so absum / abcheck / sweep read them)
mix_arm() { local n=$1 rep=$2 cfg rc=0 lab=ab-mix-$1-r$2 ml=mix-$1-r$2 u=http://127.0.0.1:$TPORT
    cfg=$(runcfg "$n" "mix-$n-r$rep") || return 1
    up "reads2-$ml" "$cfg" || return 1
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
[ -x "$B" ] || { echo "missing $B"; exit 1; }
git -C "$WT" merge-base --is-ancestor $WANT_COMMIT HEAD || { echo "$WT HEAD lacks $WANT_COMMIT (the review fixes)"; exit 1; }
SRCS=(); for d in src include CMakeLists.txt; do [ -e "$WT/$d" ] && SRCS+=("$WT/$d"); done
NEWER=$(find "${SRCS[@]}" -newer "$B" \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \
        -o -name CMakeLists.txt -o -name '*.cmake' \) | head -3)
[ -z "$NEWER" ] || { echo "$B is older than $NEWER: rebuild first"; exit 1; }
git -C "$WT" diff --quiet HEAD -- src include CMakeLists.txt serve \
    || { echo "$WT has uncommitted changes under src/, include/, CMakeLists.txt or serve/: the binary / server is not HEAD"; exit 1; }
grep -q 'STRATA_SERVER_READS' "$WT/serve/server.py" || { echo "$WT/serve/server.py has no STRATA_SERVER_READS"; exit 1; }
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
case "$SRVENV" in *STRATA_YIELD*|*STRATA_READ*|*STRATA_SERVER_READS*) echo "production's server env already has $SRVENV"; exit 1 ;; esac
PROD_SHA=$(sha256sum "$PEXE" | cut -c1-16); PRODCFG_SHA=$(sha256sum "$PRODSRC" | cut -c1-16)
TOKCHK=$(taskset -c 0-63 nice -n 15 "$PY" -I "$FT" --mode tokcheck --config "$PRODSRC" --short "$YS" --reps-list 1,2 --two 2>&1) \
    || { echo "$TOKCHK"; echo "the scenarios do not fit (see above)"; exit 1; }
cmp -s "$B" "$NEW" || { cp "$B" "$NEW" && chmod +x "$NEW"; } || { echo "cannot copy $B to $NEW"; exit 1; }
RUNLOGS=("$L"/results/engine-ab-mix-*.run.log "$L"/results/engine-ab-rslow-*.run.log "$L"/results/engine-reads2-*.run.log)
if [ -e "$OUT" ]; then PREV=$OUT.prev-$(date +%Y%m%d-%H%M%S); mv "$OUT" "$PREV" || { echo "cannot move $OUT aside"; exit 1; }
    for f in "${RUNLOGS[@]}"; do [ -e "$f" ] && mv "$f" "$PREV"/; done; fi
for f in "${RUNLOGS[@]}"; do rm -f "$f"; done
rm -rf "$TIER"; mkdir -p "$OUT" "$CF" "$TIER"
echo "$TOKCHK" | sed 's/^/    [tokcheck] /' | tee -a "$OUT/window.log" >/dev/null
SWEEP0=$(cat "$L/results/sweep.jsonl" 2>/dev/null | wc -l)
log "window prep: $(basename $NEW) sha256 $(sha256sum $NEW | cut -c1-16) = $WT $(git -C $WT rev-parse --short HEAD) ($(git -C $WT log -1 --format=%s HEAD | cut -c1-80)); production: $(basename $PRODSRC) sha256 $PRODCFG_SHA, exe $(basename $PEXE) sha256 $PROD_SHA, server $SRV @$(git -C $SRV rev-parse --short HEAD), server env: $SRVENV; READS server $WT/serve, STRATA_READS=$RN STRATA_READS_BG_MIN=$BGMIN (agent mix $MIX_BGMIN, bench_multiturn --doc $MT_DOC); FAIR STRATA_YIELD_SHORT=$YS STRATA_YIELDS_MAX=$YM"

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
        reads) echo "SERVER_DIR=$WT $SRVENV STRATA_SERVER_READS=1 STRATA_READS_BG_MIN=$BGMIN" > "$CF/$name.srv" ;;
        readsmix) echo "SERVER_DIR=$WT $SRVENV STRATA_SERVER_READS=1 STRATA_READS_BG_MIN=$MIX_BGMIN" > "$CF/$name.srv" ;;
        readssolo) echo "SERVER_DIR=$WT $SRVENV STRATA_SERVER_READS=1 STRATA_READS_BG_MIN=$BGMIN STRATA_READS_SOLO=1" > "$CF/$name.srv" ;;
    esac; }
FY=(STRATA_READ_YIELD=1)
RD=(STRATA_READS=$RN)
VER=(STRATA_SLOT_PARK_VERIFY=1 STRATA_TRACE=1)
LOGS=(STRATA_SPEC_LANES_LOG=1 STRATA_SOLO_PIPELINE_LOG=1)
mkcfg X-PROD  "$PEXE" prod "${VER[@]}"                                  # exactness: production as it runs (+ verify)
mkcfg X-RD    "$NEW"  reads "${RD[@]}" "${VER[@]}"                      # stage 2 step A, natural pauses (lines)
mkcfg X-RDP   "$NEW"  reads "${RD[@]}" "${VER[@]}" STRATA_READS_TEST_PAUSE_EVERY=3 STRATA_READS_QUANTUM=2   # pauses everywhere
mkcfg X-RD1   "$NEW"  reads STRATA_READS=1 "${VER[@]}"                  # one held read
mkcfg X-N0    "$NEW"  new0 "${VER[@]}"                                  # the default path: new binary + server, env unset
mkcfg T-RD4   "$NEW"  reads STRATA_READS=4 "${VER[@]}"                  # concurrency: up to 4 held
mkcfg X-RDG   "$NEW"  reads "${RD[@]}" "${VER[@]}" -STRATA_PIPE_OVERLAP_KVS   # the gate refused: drained turns only
mkcfg SO-PROD "$PEXE" prod                                              # lone conversations + decode after a read
mkcfg SO-RD   "$NEW"  readssolo "${RD[@]}"
mkcfg W-PROD  "$PEXE" prod                                              # the win: production as it runs
mkcfg W-FAIR  "$NEW"  fair "${FY[@]}"                                   # ... vs stage 1
mkcfg W-RD    "$NEW"  reads "${RD[@]}"                                  # ... vs stage 2 step A
mkcfg SP-PROD "$PEXE" prod "${LOGS[@]}"                                 # speed (lane log lines in both arms)
mkcfg SP-RD   "$NEW"  reads "${RD[@]}" "${LOGS[@]}"
mkcfg MX-RD   "$NEW"  readsmix "${RD[@]}" "${LOGS[@]}"                  # the agent mix: 8K first turns in the background
mkcfg RD-PROD "$PEXE" prod                                              # batch_test16
mkcfg RD-N0   "$NEW"  new0
mkcfg RD-RD   "$NEW"  new0 "${RD[@]}"
# a copy of a config with a fresh, empty tier dir of its own (batch_test16 / the lane drives)
tiercfg() { local d=$TIER/$2; rm -rf "$d"; mkdir -p "$d"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; c['log']=sys.argv[4]; json.dump(c,open(sys.argv[3],'w'),indent=1)" \
        "$1" "$d" "$CF/$2.json" "$OUT/engine-$2.log"; echo "$CF/$2.json"; }

for i in $(seq 180); do curl -s -m 3 $URL/health | grep -q '"loaded": *true' && break; sleep 5; done
curl -s -m 3 $URL/health | grep -q '"loaded": *true' || { echo "production does not report loaded: not stopping it"; exit 1; }
log "window start ($(( WINDOW_S / 60 )) min cap): production loaded; stopping strata.service"
trap '' INT TERM HUP                             # (not while it stops)
systemctl --user stop strata.service
STOPPED=1; trap 'abort signal' INT TERM HUP     # from here on an abort ends the bench engines and restarts production
T0=$(date +%s)                                  # (WINDOW_S: production's outage, from its stop to its restart)
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || abort "GPUs not free"

# ---- 1. exactness over HTTP: production (sequential reads) vs background reads; text and pictures ------------------
XS=exact,vis
step x_PROD 420 1500 scen_arm x_PROD X-PROD $XS 0
step x_RD 420 1500 scen_arm x_RD X-RD $XS 0
judge x_cmp_RD x_PROD,x_RD "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_RD.json --what "exact+vis: READS N=$RN vs production"
step x_RDP 420 1500 scen_arm x_RDP X-RDP $XS 0
judge x_cmp_RDP x_PROD,x_RDP "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_RDP.json --what "exact+vis: READS, a pause every 3 chunks, quantum 2 vs production"
judge x_cmp_vis x_PROD,x_RD "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_RD.json --rounds v1,v2 \
    --what "pictures beside background reads (v1/v2 only): READS vs production"
judge x_reads x_PROD "$PY" -I "$FT" --mode readstat --arms PROD=$OUT/engine-x_PROD.log RD=$OUT/engine-x_RD.log \
    RDP=$OUT/engine-x_RDP.log --want-none PROD --want-bg RD,RDP --want-clean PROD,RD,RDP

# ---- 2. concurrency: two and three long reads at once, up to 4 held ------------------------------------------------------
step t_PROD 300 1500 scen_arm t_PROD X-PROD two 1
step t_RD4 300 1500 scen_arm t_RD4 T-RD4 two 1
judge t_cmp_RD4 t_PROD,t_RD4 "$PY" -I "$FT" --mode cmp --a $OUT/t_PROD.json --b $OUT/t_RD4.json --what "two: READS N=4 vs production"
judge t_reads t_PROD "$PY" -I "$FT" --mode readstat --arms PROD=$OUT/engine-t_PROD.log RD4=$OUT/engine-t_RD4.log \
    --want-none PROD --want-bg RD4 --want-clean PROD,RD4
f=$OUT/engine-t_RD4.log
if step_ran t_RD4 && [ -f "$f" ]; then
    n=$(grep -o 'STRATA_READS, [0-9]* held' "$f" | awk '{print $2}' | sort -n | tail -1)
    log "    [t_reads] ==> RD4: most background reads held at once: ${n:-0} (want >= 2: else step 2 proves nothing about concurrent reads) $([ "${n:-0}" -ge 2 ] && echo PASS || echo FAIL)"
fi

# ---- 3. the user-visible win: shorts during a cold 100K-125K read, a burst of 16; PROD READS FAIR (reps 1,2 each) -------
step w_PROD 480 1800 scen_reps w_PROD W-PROD win 1,2
step w_RD 480 1800 scen_reps w_RD W-RD win 1,2
step w_FAIR 480 1800 scen_reps w_FAIR W-FAIR win 1,2
judge win w_PROD,w_RD "$PY" -I "$FT" --mode win --arms "PROD=$OUT/w_PROD_[12].json" "FAIR=$OUT/w_FAIR_[12].json" "READS=$OUT/w_RD_[12].json"
judge win_vs_fair w_FAIR,w_RD "$PY" -I "$FT" --mode win --gain 1.0 --long-tol 0.05 --arms "FAIR=$OUT/w_FAIR_[12].json" "READS=$OUT/w_RD_[12].json"
for k in 1 2; do
    judge w_cmp_$k w_PROD,w_RD "$PY" -I "$FT" --mode cmp --a $OUT/w_PROD_$k.json --b $OUT/w_RD_$k.json \
        --what "win rep $k (greedy): READS vs production"
    judge w_cmpf_$k w_PROD,w_FAIR "$PY" -I "$FT" --mode cmp --a $OUT/w_PROD_$k.json --b $OUT/w_FAIR_$k.json \
        --what "win rep $k (greedy): FAIR vs production"
done
judge w_reads w_RD "$PY" -I "$FT" --mode readstat --arms "PROD=$OUT/engine-w_PROD.log" "RD=$OUT/engine-w_RD.log" \
    --want-none PROD --want-bg RD --want-clean PROD,RD --max-starts 1

# ---- 4. agent mix + 16 users, ABBA: PROD READS READS PROD ----------------------------------------------------------------
step mix_PROD_1 450 1500 mix_arm SP-PROD 1
step mix_RD_1 450 1500 mix_arm MX-RD 1
step mix_RD_2 450 1500 mix_arm MX-RD 2
step mix_PROD_2 450 1500 mix_arm SP-PROD 2
judge mix mix_PROD_1,mix_RD_1 "$PY" -I "$FT" --mode mix --dir $OUT --arms-list SP-PROD,MX-RD \
    --reps $(step_ran mix_PROD_2 && step_ran mix_RD_2 && echo 2 || echo 1) --gain 1.10
judge mix_sum mix_PROD_1,mix_RD_1,mix_RD_2,mix_PROD_2 "$PY" -I "$FT" --mode absum --tag mix --reps 2 --arms-list SP-PROD,MX-RD --since-line "$SWEEP0"
judge mix_ident mix_PROD_1,mix_RD_1,mix_RD_2,mix_PROD_2 "$PY" -I "$DRV" --mode abcheck --file $OUT/mix_sum.txt --what "16-user replies, READS vs production" --expect identical
judge mix_agg mix_PROD_1,mix_RD_1,mix_RD_2,mix_PROD_2 "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag mix \
    --arms SP-PROD,MX-RD --reps 2 --flat-users 16 --flat-tol 1.5
judge mix_ttft mix_PROD_1,mix_RD_1,mix_RD_2,mix_PROD_2 "$PY" -I "$RT" --mode abttft --since-line "$SWEEP0" --tag mix \
    --arms-list SP-PROD,MX-RD --reps 2
judge mix_eng mix_PROD_1,mix_RD_1 "$PY" -I "$FT" --mode readstat --arms "PROD=$OUT/engine-mix-SP-PROD-r*.log" \
    "RD=$OUT/engine-mix-MX-RD-r*.log" --want-none PROD --want-bg RD --want-clean PROD,RD

# ---- 5. the default path: batch_test16 dumps + scen exact,vis over HTTP (new binary + new server, env unset) -------------
btarm() { local arm=$1 cfg exe; cfg=$(tiercfg $CF/RD-$arm.json bt_$arm); exe=$(cfg_exe "$cfg")
    step bt_$arm 150 900 scoped bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-bt_$arm.log env $(cfg_env $cfg) STRATA_TRACE=1 python3 tools/batch_test16.py --exe $exe --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --dump $OUT/bt_$arm.json"; }
btarm PROD; btarm N0
judge bt_cmp bt_PROD,bt_N0 "$PY" -I "$RT" --mode btcmp --dir $OUT --base bt_PROD --arms bt_N0 --what "batch_test16 greedy"
judge boot_N0 bt_PROD,bt_N0 "$PY" -I "$RT" --mode bootcmp --base $OUT/engine-bt_PROD.log --got $OUT/engine-bt_N0.log --what "new binary env unset vs production"
judge bt_eng bt_PROD,bt_N0 "$PY" -I "$FT" --mode readstat --arms PROD=$OUT/engine-bt_PROD.log N0=$OUT/engine-bt_N0.log \
    --want-none PROD,N0 --want-clean PROD,N0
step x_N0 420 1500 scen_arm x_N0 X-N0 $XS 0
judge x_cmp_N0 x_PROD,x_N0 "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_N0.json --what "exact+vis: default path (new binary + server, env unset) vs production"
judge x_reads_N0 x_N0 "$PY" -I "$FT" --mode readstat --arms N0=$OUT/engine-x_N0.log --want-none N0 --want-clean N0

# ---- 6. 1-4 users, PROD then READS ---------------------------------------------------------------------------------------
step spdlow 500 1200 abq rslow 1 1,2,3,4 SP-PROD SP-RD
judge spdlow_sum spdlow "$PY" -I "$FT" --mode absum --tag rslow --reps 1 --arms-list SP-PROD,SP-RD --since-line "$SWEEP0"
judge spdlow_ident spdlow "$PY" -I "$DRV" --mode abcheck --file $OUT/spdlow_sum.txt \
    --what "1-4 user replies (strict at 1 user; INFO beyond: lanes / pump windows by timing)" --expect identical --strict-users 1
judge spdlow_dec spdlow "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag rslow --arms SP-PROD,SP-RD --reps 1 \
    --dec-min "1:-1.5,2:-1.5,3:-2,4:-2"
judge spdlow_ttft spdlow "$PY" -I "$RT" --mode abttft --since-line "$SWEEP0" --tag rslow --arms-list SP-PROD,SP-RD --reps 1
judge spdlow_lanes spdlow "$PY" -I "$DRV" --mode lanelog --what "READS HTTP engines, 1-4 users" --want-sizes 2,3,4 \
    "$L/results/engine-ab-rslow-SP-RD-long.run.log"
for s in mix_sum spdlow_sum; do [ -f "$OUT/$s.txt" ] && sed 's/^/    /' "$OUT/$s.txt" | tee -a "$OUT/window.log" >/dev/null; done

# ---- 7. lone conversations, decode and drafts after a background read ----------------------------------------------------
step so_PROD 420 1500 scen_arm so_PROD SO-PROD solo 1
step so_RD 420 1500 scen_arm so_RD SO-RD solo 1
judge so_cmp so_PROD,so_RD "$PY" -I "$FT" --mode cmp --a $OUT/so_PROD.json --b $OUT/so_RD.json \
    --what "solo (greedy): READS (STRATA_READS_SOLO=1) vs production"
judge so_dec so_PROD,so_RD "$PY" -I "$FT" --mode dec --arms "PROD=$OUT/so_PROD.json" "READS=$OUT/so_RD.json"
judge so_reads so_RD "$PY" -I "$FT" --mode readstat --arms "PROD=$OUT/engine-so_PROD.log" "RD=$OUT/engine-so_RD.log" \
    --want-none PROD --want-bg RD --want-clean PROD,RD
f=$OUT/engine-so_RD.log   # o1's first turn and o2 in the background, o1's two follow-ups solo (R2-1): 2 per run
step_ran so_RD && [ -f "$f" ] && n=$(grep -c 'the prompt read goes to the background at' "$f"; true) && \
    log "    [so_reads] ==> so_RD: $n background reads (want 2: the follow-ups of a solo conversation stay solo) $([ "$n" = 2 ] && echo PASS || echo FAIL)"

# ---- 8. if time: the gate refused, N=2 on two, N=1, batch_test16 with STRATA_READS ----------------------------------------
step x_RDG 300 1500 scen_arm x_RDG X-RDG exact 0 x1
judge x_cmp_RDG x_PROD,x_RDG "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_RDG.json --rounds x1 \
    --what "exact x1: READS with the gate refused (drained turns) vs production"
judge x_reads_RDG x_RDG "$PY" -I "$FT" --mode readstat --arms RDG=$OUT/engine-x_RDG.log --want-bg RDG --want-clean RDG
judge x_stall_RDG x_PROD,x_RDG "$PY" -I "$FT" --mode dec --tol 1.0 --rounds x1 --arms "PROD=$OUT/x_PROD.json" "RDG=$OUT/x_RDG.json"
step t_RD 300 1500 scen_arm t_RD X-RD two 1
judge t_cmp_RD t_PROD,t_RD "$PY" -I "$FT" --mode cmp --a $OUT/t_PROD.json --b $OUT/t_RD.json --what "two: READS N=$RN vs production"
judge t_reads_RD t_RD "$PY" -I "$FT" --mode readstat --arms RD=$OUT/engine-t_RD.log --want-bg RD --want-clean RD
step x_RD1 420 1500 scen_arm x_RD1 X-RD1 $XS 0
judge x_cmp_RD1 x_PROD,x_RD1 "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_RD1.json --what "exact+vis: READS N=1 vs production"
judge x_reads_RD1 x_RD1 "$PY" -I "$FT" --mode readstat --arms RD1=$OUT/engine-x_RD1.log --want-bg RD1 --want-clean RD1
btarm RD
judge bt_cmp_RD bt_PROD,bt_RD "$PY" -I "$RT" --mode btcmp --dir $OUT --base bt_PROD --arms bt_RD --what "batch_test16 greedy, STRATA_READS=$RN"
judge boot_RD bt_N0,bt_RD "$PY" -I "$RT" --mode bootcmp --base $OUT/engine-bt_N0.log --got $OUT/engine-bt_RD.log --what "STRATA_READS=$RN vs env unset"

# ---- 9. production back, unchanged ------------------------------------------------------------------------------------
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
