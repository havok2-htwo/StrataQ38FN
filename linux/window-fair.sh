#!/usr/bin/env bash
# GPU window for branch ll6-reads (Strata-ll6-reads, HEAD >= bda03b22): stage 1 FAIR - a prompt read in progress gives
# way at a chunk boundary to short requests (engine STRATA_READ_YIELD=1, server STRATA_YIELD_SHORT=2048 from
# Strata-ll6-reads/serve) - against production as it runs now (production.json read at run time: strata-g3-1441f-lanes4,
# overlap + KVS, solo pipeline, STRATA_SPEC_LANES=4, --kv-resident 28672, split 12,25,37, 16 slots, server
# Strata-143-prod with the drop-in's STRATA_* env; the window refuses a production config of another shape).
# Binary: strata-g3-1441i-fair (a copy of Strata-ll6-reads/build/strata).  Stage 2 (STRATA_READS=N) is not built: its
# arm is logged as SKIP.
# ~1 h 50 min expected, hard cap WINDOW_S (default 7200 s = 2 h, production restart included): a step that no longer
# fits is skipped (logged), steps are ordered by value.  Production is stopped for the window and started again
# unchanged (production.json, the unit and its drop-in are never written), also on abort (trap).  Every test engine
# logs into results/fair (never production's log) and spills into /media/raid0/strata-kvspill-bench/fair/<run> (a fresh
# dir per server start, never production's tier dir); every test HTTP server listens on 127.0.0.1:8091 (never
# production's port 8090).  The test servers run serve/server.py from Strata-143-prod (production's arms) or from
# Strata-ll6-reads (the new arms; SERVER_DIR), each with production's server env.  A signal (INT/TERM/HUP) before
# production is stopped only exits; after it, it interrupts the running step at once, ends the bench engines and starts
# production again (signals ignored during that restart).  An existing results/fair moves to results/fair.prev-<time>.
#   run:      cd /media/raid0/StrataQ38FN/linux && ./window-fair.sh
#   verdicts: grep -E 'PASS|FAIL|SKIP' results/fair/window.log
# Before the window (no GPU, production serving): every scenario prompt measured with the tokenizer (fair_tools.py
# tokcheck: each request incl. its output and history < 262,144), the lanes4 window's serial solo reference checked for
# reuse, pictures, the server checkout clean and compiling.
# Steps (PASS/FAIL/INFO lines, a time budget each; in this order):
#   1 exactness over HTTP (greedy + seeded T=0.7, STRATA_SLOT_PARK_VERIFY=1 + STRATA_TRACE=1 in every arm): fair_tools
#     scen exact - a lone 48K read + a short (the solo path), a 100K read beside 4 decoding streams + 3 shorts (one
#     T=0.7 seeded), a 40K T=0.7 seeded read + a seeded short beside 2 streams, a 64K read + a burst of 8 shorts; then
#     the long reads' and two shorts' second turns (resumes: a slot state that differs gives another reply).
#     Production vs FAIR (natural stops) and FAIR with the stop placed by STRATA_READ_YIELD_TEST_AT at ~30K (just past
#     --kv-resident), ~90K and ~8K of the read: every reply identical; reads gave way (else no claim), every resume
#     pinned, park verify identical, no ERR / watchdog
#   2 the user-visible win, ABBA (PROD FAIR FAIR PROD): w1 a cold ~125K read beside 2 decoding streams, 4 shorts at
#     +3/6/9/12 s (every one's TTFT: judged on the max); w2 a cold ~105K read, a burst of 16 shorts at +4 s (time to all 16 first tokens); the
#     long read's own TTFT / total (its cost); replies identical per repetition.  Stage 2: SKIP (not built)
#   3 no regression, 16 users short + 1.2K ABBA x2 MIX=128 (reply identity, agg within 1.5 %, TTFT)
#   4 batch_test16 dumps (greedy and T=0.7 seed 1234, a cold 40K read beside 15 decoding): production vs the new binary
#     env unset (default path) and with STRATA_READ_YIELD=1; boot VRAM / experts; HTTP exactness of the default path
#     (new binary + new server, env unset) vs production
#   5 lanes with STRATA_READ_YIELD=1 == serial solo spec (1 user pipelined + resume, switch 1-2-3-4-2-1, HOLD 2/3/4)
#   6 1-4 users ABBA x2 (decode p50 per user, 1 user strictly identical, lane starts by lane count)
#   7 production restarted unchanged, health checked
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/fair; CF=$OUT/cfg
URL=http://127.0.0.1:8090; TPORT=8091; PY=$R/Strata/.venv/bin/python; DRV=$L/lanes_stdin.py; RT=$L/reads_tools.py
FT=$L/fair_tools.py
WT=$R/Strata-ll6-reads; B=$WT/build/strata; NEW=$R/Strata/engine/strata-g3-1441i-fair
WANT_COMMIT=bda03b22
YS=${YS:-2048}                                              # STRATA_YIELD_SHORT of the FAIR arms
# STRATA_YIELDS_MAX of the FAIR arms: the server's default (2 stops per read) lets only the first two of w1's four
# spaced shorts past the 125K read (the 3rd and 4th wait it out) and x1's third short; 4 = one stop per short there
# (YM=2: the server's default as it is)
YM=${YM:-4}
PROD_EXE_WANT=${PROD_EXE_WANT:-strata-g3-1441f-lanes4}       # what production must run for this window's comparisons
PRODSRC=$(readlink -f "$C/production.json")                 # what production runs now (its config is only read)
SRV=$R/Strata-143-prod                                      # production's server checkout
TIER=/media/raid0/strata-kvspill-bench/fair
LANES4=$L/results/lanes4                                    # the lanes4 window: its serial solo reference, its pictures
EXACT="--pcie-frac 0 --adapt-every 0 --suffix-draft 0"
XENV=(--env STRATA_IQ_MT_MIN=1 --env STRATA_BATCH_DECODE_SHARE=0 --env STRATA_DECODE_TIMING=1)
NO_OV=(--drop-env STRATA_PIPE_OVERLAP --drop-env STRATA_PIPE_OVERLAP_KVS)   # (as the lanes4 / reads windows' lane drives)
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
# scen_arm <run> <NAME> <scenario> <rep>: one server start, one fair_tools scenario, stop
scen_arm() { local run=$1 cfg rc; cfg=$(runcfg "$2" "$run") || return 1
    up "fair-$run" "$cfg" || return 1
    timeout -s INT 1300 "$PY" -I "$FT" --mode scen --config "$cfg" --url "http://127.0.0.1:$TPORT" --scenario "$3" \
        --rep "$4" --label "$run" --out "$OUT/$run.json"; rc=$?
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
export -f scoped drv tup runcfg srv up scen_arm abq
export PY DRV FT L CF OUT TIER TPORT LANES_TIER_ROOT   # (step runs them under timeout through bash -c)
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
grep -q 'STRATA_YIELD_SHORT' "$WT/serve/server.py" || { echo "$WT/serve/server.py has no STRATA_YIELD_SHORT"; exit 1; }
"$PY" -m py_compile "$WT/serve/server.py" || { echo "$WT/serve/server.py does not compile"; exit 1; }
for f in "$DRV" "$RT" "$FT" "$L/run_strata_variant.sh" "$L/compare_replies.py" "$L/ovkv_ident.py" \
         "$R/Strata-1441/tools/batch_test16.py" "$SRV/serve/server.py" "$PRODSRC"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
for f in "$DRV" "$RT" "$FT" "$L/run_strata_variant.sh"; do [ -x "$f" ] || { echo "$f is not executable (chmod +x)"; exit 1; }; done
for f in "$DRV" "$RT" "$FT"; do "$PY" -I "$f" --help > /dev/null || { echo "$f does not load"; exit 1; }; done
curl -s -m 2 "http://127.0.0.1:$TPORT/health" > /dev/null && { echo "port $TPORT already serves something: the test servers need it"; exit 1; }
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
systemctl --user is-active -q strata.service || { echo "strata.service is not active: start it first"; exit 1; }
# production's server env (the drop-in's STRATA_* keys, read now): every test server gets it, the FAIR arms add
# STRATA_YIELD_SHORT
SRVENV=$(systemctl --user show strata.service -p Environment --value | tr ' ' '\n' | grep '^STRATA_' | tr '\n' ' ')
case "$SRVENV" in *STRATA_YIELD*|*STRATA_READ*) echo "production's server env already has $SRVENV"; exit 1 ;; esac
PROD_SHA=$(sha256sum "$PEXE" | cut -c1-16); PRODCFG_SHA=$(sha256sum "$PRODSRC" | cut -c1-16)
TOKCHK=$(taskset -c 0-63 nice -n 15 "$PY" -I "$FT" --mode tokcheck --config "$PRODSRC" --short "$YS" --reps-list 1,2 2>&1) \
    || { echo "$TOKCHK"; echo "the scenarios do not fit (see above)"; exit 1; }
cmp -s "$B" "$NEW" || { cp "$B" "$NEW" && chmod +x "$NEW"; } || { echo "cannot copy $B to $NEW"; exit 1; }
RUNLOGS=("$L"/results/engine-ab-fs16-*.run.log "$L"/results/engine-ab-fslow-*.run.log "$L"/results/engine-fair-*.run.log)
if [ -e "$OUT" ]; then PREV=$OUT.prev-$(date +%Y%m%d-%H%M%S); mv "$OUT" "$PREV" || { echo "cannot move $OUT aside"; exit 1; }
    for f in "${RUNLOGS[@]}"; do [ -e "$f" ] && mv "$f" "$PREV"/; done; fi
for f in "${RUNLOGS[@]}"; do rm -f "$f"; done
rm -rf "$TIER"; mkdir -p "$OUT" "$CF" "$TIER"
echo "$TOKCHK" | sed 's/^/    [tokcheck] /' | tee -a "$OUT/window.log" >/dev/null
SWEEP0=$(cat "$L/results/sweep.jsonl" 2>/dev/null | wc -l)
log "window prep: $(basename $NEW) sha256 $(sha256sum $NEW | cut -c1-16) = $WT $(git -C $WT rev-parse --short HEAD) ($(git -C $WT log -1 --format=%s HEAD | cut -c1-80)); production: $(basename $PRODSRC) sha256 $PRODCFG_SHA, exe $(basename $PEXE) sha256 $PROD_SHA, server $SRV @$(git -C $SRV rev-parse --short HEAD), server env: $SRVENV; FAIR server $WT/serve, STRATA_YIELD_SHORT=$YS STRATA_YIELDS_MAX=$YM"

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
    esac; }
FY=(STRATA_READ_YIELD=1)
VER=(STRATA_SLOT_PARK_VERIFY=1 STRATA_TRACE=1)
LOGS=(STRATA_SPEC_LANES_LOG=1 STRATA_SOLO_PIPELINE_LOG=1)
mkcfg X-PROD  "$PEXE" prod "${VER[@]}"                                  # exactness: production as it runs (+ verify)
mkcfg X-FAIR  "$NEW"  fair "${FY[@]}" "${VER[@]}"                       # stage 1, natural stops
mkcfg X-Q30   "$NEW"  fair "${FY[@]}" "${VER[@]}" STRATA_READ_YIELD_TEST_AT=59    # stop from ~30.2K (past --kv-resident)
mkcfg X-Q90   "$NEW"  fair "${FY[@]}" "${VER[@]}" STRATA_READ_YIELD_TEST_AT=176   # from ~90.1K
mkcfg X-Q8    "$NEW"  fair "${FY[@]}" "${VER[@]}" STRATA_READ_YIELD_TEST_AT=16    # from ~8.2K
mkcfg X-N0    "$NEW"  new0 "${VER[@]}"                                  # the default path: new binary + server, env unset
mkcfg W-PROD  "$PEXE" prod                                              # the win: production as it runs
mkcfg W-FAIR  "$NEW"  fair "${FY[@]}"                                   # ... vs stage 1
mkcfg SP-PROD "$PEXE" prod "${LOGS[@]}"                                 # speed (lane log lines in both arms)
mkcfg SP-FAIR "$NEW"  fair "${FY[@]}" "${LOGS[@]}"
mkcfg RD-PROD "$PEXE" prod                                              # batch_test16
mkcfg RD-N0   "$NEW"  new0
mkcfg RD-FY   "$NEW"  fair "${FY[@]}"
mkcfg LN-FY   "$NEW"  fair "${FY[@]}" "${LOGS[@]}"                      # the lane drives
mkcfg LN-OFF  "$PEXE" prod -STRATA_SPEC_LANES -STRATA_SOLO_PIPELINE     # the serial solo reference (if it must be run)
# a copy of a config with a fresh, empty tier dir of its own (batch_test16 / the lane drives)
tiercfg() { local d=$TIER/$2; rm -rf "$d"; mkdir -p "$d"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; c['log']=sys.argv[4]; json.dump(c,open(sys.argv[3],'w'),indent=1)" \
        "$1" "$d" "$CF/$2.json" "$OUT/engine-$2.log"; echo "$CF/$2.json"; }

# the serial solo reference of the lane drives: the lanes4 window's when it still matches, else this window runs it
REF_REUSED=0
if "$PY" -I "$RT" --mode refok --ref "$LANES4/ref.json" --ref-txt "$LANES4/ref.txt" --ref-cfg "$LANES4/cfg/LC-OFF-28672.json" \
        --window-log "$LANES4/window.log" --prod "$PRODSRC" --prod-sha "$PROD_SHA" --kvres 28672 > "$OUT/refok.txt" 2>&1; then
    cp "$LANES4/ref.json" "$OUT/ref.json" && REF_REUSED=1 && echo 0 > "$OUT/rc_ref"; fi
log "reference: $(tail -1 "$OUT/refok.txt" | cut -c1-300)"
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

# ---- 1. exactness over HTTP: production vs stage 1 (natural stops, then stops placed at ~30K / ~90K / ~8K) ------------
step x_PROD 300 1500 scen_arm x_PROD X-PROD exact 0
step x_FAIR 300 1500 scen_arm x_FAIR X-FAIR exact 0
judge x_cmp_FAIR x_PROD,x_FAIR "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_FAIR.json --what "exact: FAIR (natural stops) vs production"
step x_Q30 300 1500 scen_arm x_Q30 X-Q30 exact 0
judge x_cmp_Q30 x_PROD,x_Q30 "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_Q30.json --what "exact: FAIR stop >= ~30K vs production"
step x_Q90 300 1500 scen_arm x_Q90 X-Q90 exact 0
judge x_cmp_Q90 x_PROD,x_Q90 "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_Q90.json --what "exact: FAIR stop >= ~90K vs production"
step x_Q8 300 1500 scen_arm x_Q8 X-Q8 exact 0
judge x_cmp_Q8 x_PROD,x_Q8 "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_Q8.json --what "exact: FAIR stop >= ~8K vs production"
judge x_yield x_PROD "$PY" -I "$FT" --mode yieldstat --arms PROD=$OUT/engine-x_PROD.log FAIR=$OUT/engine-x_FAIR.log \
    Q30=$OUT/engine-x_Q30.log Q90=$OUT/engine-x_Q90.log Q8=$OUT/engine-x_Q8.log \
    --want-none PROD --want-yield FAIR,Q30,Q90,Q8 --want-clean PROD,FAIR,Q30,Q90,Q8 --test-at Q30:59 --test-at Q90:176 --test-at Q8:16

# ---- 2. the user-visible win: TTFT of shorts during a cold 100K-150K read, a burst of 16; ABBA -------------------------
step w_PROD_1 300 1200 scen_arm w_PROD_1 W-PROD win 1
step w_FAIR_1 300 1200 scen_arm w_FAIR_1 W-FAIR win 1
step w_FAIR_2 300 1200 scen_arm w_FAIR_2 W-FAIR win 2
step w_PROD_2 300 1200 scen_arm w_PROD_2 W-PROD win 2
judge win w_PROD_1,w_FAIR_1 "$PY" -I "$FT" --mode win --arms "PROD=$OUT/w_PROD_[12].json" "FAIR=$OUT/w_FAIR_[12].json"
log "    [win] ==> stage 2 (STRATA_READS=N, up to 4 reads at once): not built (only include/strata/program/read_sched.hpp) SKIP"
for k in 1 2; do
    judge w_cmp_$k w_PROD_$k,w_FAIR_$k "$PY" -I "$FT" --mode cmp --a $OUT/w_PROD_$k.json --b $OUT/w_FAIR_$k.json \
        --what "win rep $k (greedy): FAIR vs production"
done
judge w_yield w_FAIR_1 "$PY" -I "$FT" --mode yieldstat --arms "PROD=$OUT/engine-w_PROD_*.log" "FAIR=$OUT/engine-w_FAIR_*.log" \
    --want-none PROD --want-yield FAIR --want-clean PROD,FAIR

# ---- 3. no regression: 16 users ABBA x2 --------------------------------------------------------------------------------
step spd16 900 2400 abq fs16 2 16 SP-PROD SP-FAIR
judge spd16_sum spd16 "$PY" -I "$FT" --mode absum --tag fs16 --reps 2 --arms-list SP-PROD,SP-FAIR --since-line "$SWEEP0"
judge spd16_ident spd16 "$PY" -I "$DRV" --mode abcheck --file $OUT/spd16_sum.txt --what "16-user replies, FAIR vs production" --expect identical
judge spd16_agg spd16 "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag fs16 --arms SP-PROD,SP-FAIR --reps 2 --flat-users 16 --flat-tol 1.5
judge spd16_ttft spd16 "$PY" -I "$RT" --mode abttft --since-line "$SWEEP0" --tag fs16 --arms-list SP-PROD,SP-FAIR --reps 2
judge spd16_eng spd16 "$PY" -I "$FT" --mode yieldstat --arms "PROD=$OUT/engine-fs16-SP-PROD-r*.log" "FAIR=$OUT/engine-fs16-SP-FAIR-r*.log" \
    --want-clean PROD,FAIR

# ---- 4. batch_test16 dumps + the default path over HTTP ----------------------------------------------------------------
for arm in PROD N0 FY; do
    cfg=$(tiercfg $CF/RD-$arm.json bt_$arm); exe=$(cfg_exe "$cfg")
    step bt_$arm 150 900 scoped bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-bt_$arm.log env $(cfg_env $cfg) STRATA_TRACE=1 python3 tools/batch_test16.py --exe $exe --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --dump $OUT/bt_$arm.json"
done
judge bt_cmp bt_PROD "$PY" -I "$RT" --mode btcmp --dir $OUT --base bt_PROD --arms bt_N0,bt_FY --what "batch_test16 greedy"
judge boot_N0 bt_PROD,bt_N0 "$PY" -I "$RT" --mode bootcmp --base $OUT/engine-bt_PROD.log --got $OUT/engine-bt_N0.log --what "new binary env unset vs production"
judge boot_FY bt_N0,bt_FY "$PY" -I "$RT" --mode bootcmp --base $OUT/engine-bt_N0.log --got $OUT/engine-bt_FY.log --what "STRATA_READ_YIELD=1 vs env unset"
for arm in PROD FY; do
    cfg=$(tiercfg $CF/RD-$arm.json bt7_$arm); exe=$(cfg_exe "$cfg")
    step bt7_$arm 150 900 scoped bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-bt7_$arm.log env $(cfg_env $cfg) STRATA_TRACE=1 python3 tools/batch_test16.py --exe $exe --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --keys 'temperature=0.7 seed=1234' --dump $OUT/bt7_$arm.json"
done
judge bt7_cmp bt7_PROD "$PY" -I "$RT" --mode btcmp --dir $OUT --base bt7_PROD --arms bt7_FY --what "batch_test16 T=0.7 seed 1234"
judge bt_eng bt_PROD "$PY" -I "$FT" --mode yieldstat --arms PROD=$OUT/engine-bt_PROD.log N0=$OUT/engine-bt_N0.log \
    FY=$OUT/engine-bt_FY.log PROD7=$OUT/engine-bt7_PROD.log FY7=$OUT/engine-bt7_FY.log --want-none PROD,N0,FY,PROD7,FY7 \
    --want-clean PROD,N0,FY,PROD7,FY7
step x_N0 300 1500 scen_arm x_N0 X-N0 exact 0
judge x_cmp_N0 x_PROD,x_N0 "$PY" -I "$FT" --mode cmp --a $OUT/x_PROD.json --b $OUT/x_N0.json --what "exact: default path (new binary + server, env unset) vs production"
judge x_yield_N0 x_N0 "$PY" -I "$FT" --mode yieldstat --arms N0=$OUT/engine-x_N0.log --want-none N0 --want-clean N0

# ---- 5. the lanes with STRATA_READ_YIELD=1 == serial solo spec ---------------------------------------------------------
if [ $REF_REUSED = 0 ]; then
    rcfg=$(tiercfg $CF/LN-OFF.json ln_ref)
    step ref 1500 1800 drv --mode solo --seq ref4 --config $rcfg --label REF --extra="$EXACT" "${XENV[@]}" \
        --log $OUT/engine-ref.log --out $OUT/ref.json
fi
LX=(--env STRATA_SPEC_LANES_LOG=1 --env STRATA_SOLO_PIPELINE_LOG=1)
if step_ran ref; then
  LCF=$(tiercfg $CF/LN-FY.json ln_fy)
  step solo_lc 180 1200 drv --mode solo --seq m1a --config $LCF --label FY-SOLO --extra="$EXACT" "${XENV[@]}" \
      --log $OUT/engine-solo_lc.log --out $OUT/solo_lc.json
  judge cmp_solo_lc ref,solo_lc "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/solo_lc.json --seq m1a \
      --what "1 user pipelined (STRATA_READ_YIELD=1)" --need-pipelined --need-resume
  NL=$(python3 -c "import json,sys
try: print(int(json.load(open(sys.argv[1]))['info'].get('spec_lanes') or 0))
except Exception: print(0)" $OUT/solo_lc.json)
  NLP=$(python3 -c "import json,sys
try: print(int(json.load(open(sys.argv[1])).get('lanes') or 4))
except Exception: print(4)" $LANES4/boot_LC4_28.json)
  step_ran solo_lc && log "    [lanes] ==> lanes fitted with STRATA_READ_YIELD=1: $NL (production's config fitted $NLP in the lanes4 window) $([ "$NL" -ge "$NLP" ] && echo PASS || echo FAIL)"
  [ "$NL" -ge 2 ] || NL=$NLP
  step switch 180 1200 drv --mode switch --config $LCF --label FY-SWITCH --extra="$EXACT" "${XENV[@]}" "${LX[@]}" \
      --env STRATA_SPEC_LANES_HOLD=2 "${NO_OV[@]}" --log $OUT/engine-switch.log --out $OUT/switch.json
  judge cmp_switch ref,switch "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/switch.json --what "switch 1-2-3-4-2-1 (STRATA_READ_YIELD=1)"
  step hold2 180 1200 drv --mode lanes --pairs g2s --pics "$PICS" --config $LCF --label FY-HOLD2 --extra="$EXACT" \
      "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=2 "${NO_OV[@]}" --log $OUT/engine-hold2.log --out $OUT/hold2.json
  judge cmp_hold2 ref,hold2 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold2.json --pairs g2s \
      --what "2 lanes (HOLD=2, STRATA_READ_YIELD=1)" --hold --full
  if [ "$NL" -ge 3 ]; then
    step hold3 180 1200 drv --mode lanes --pairs g3 --config $LCF --label FY-HOLD3 --extra="$EXACT" \
        "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=3 "${NO_OV[@]}" --log $OUT/engine-hold3.log --out $OUT/hold3.json
    judge cmp_hold3 ref,hold3 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold3.json --pairs g3 \
        --what "3 lanes (HOLD=3, STRATA_READ_YIELD=1)" --hold --full
  else noskip hold3 "the engine fitted $NL lanes"; fi
  if [ "$NL" -ge 4 ]; then
    step hold4 180 1200 drv --mode lanes --pairs g4 --config $LCF --label FY-HOLD4 --extra="$EXACT" \
        "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=4 "${NO_OV[@]}" --log $OUT/engine-hold4.log --out $OUT/hold4.json
    judge cmp_hold4 ref,hold4 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold4.json --pairs g4 \
        --what "4 lanes (HOLD=4, STRATA_READ_YIELD=1)" --hold --full
  else noskip hold4 "the engine fitted $NL lanes"; fi
  judge lanes_eng solo_lc "$PY" -I "$FT" --mode yieldstat --arms solo_lc=$OUT/engine-solo_lc.log switch=$OUT/engine-switch.log \
      hold2=$OUT/engine-hold2.log hold3=$OUT/engine-hold3.log hold4=$OUT/engine-hold4.log \
      --want-none solo_lc,switch,hold2,hold3,hold4 --want-clean solo_lc,switch,hold2,hold3,hold4
else
  for s in solo_lc switch hold2 hold3 hold4; do noskip $s "no serial solo reference"; done
fi

# ---- 6. 1-4 users ABBA x2 ------------------------------------------------------------------------------------------------
step spdlow 1000 1800 abq fslow 2 1,2,3,4 SP-PROD SP-FAIR
judge spdlow_sum spdlow "$PY" -I "$FT" --mode absum --tag fslow --reps 2 --arms-list SP-PROD,SP-FAIR --since-line "$SWEEP0"
judge spdlow_ident spdlow "$PY" -I "$DRV" --mode abcheck --file $OUT/spdlow_sum.txt \
    --what "1-4 user replies (strict at 1 user; INFO beyond: lanes / pump windows by timing)" --expect identical --strict-users 1
judge spdlow_dec spdlow "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag fslow --arms SP-PROD,SP-FAIR --reps 2 \
    --dec-min "1:-1.5,2:-1.5,3:-2,4:-2"
judge spdlow_ttft spdlow "$PY" -I "$RT" --mode abttft --since-line "$SWEEP0" --tag fslow --arms-list SP-PROD,SP-FAIR --reps 2
judge spdlow_lanes spdlow "$PY" -I "$DRV" --mode lanelog --what "FAIR HTTP engines, 1-4 users" --want-sizes 2,3,4 \
    "$L/results/engine-ab-fslow-SP-FAIR-r*-long.run.log"
for s in spd16_sum spdlow_sum; do [ -f "$OUT/$s.txt" ] && sed 's/^/    /' "$OUT/$s.txt" | tee -a "$OUT/window.log" >/dev/null; done

# ---- 7. production back, unchanged ------------------------------------------------------------------------------------
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
