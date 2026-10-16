#!/usr/bin/env bash
# GPU window for branch ll4-lanes (Strata-ll4-lanes, HEAD >= f60f4ea5): one mechanism for 1-4 users -
# STRATA_SPEC_LANES=4 (up to four spec lanes on slots 0/4/8/12, the solo pipeline folded in, the shared-VRAM profile:
# STRATA_LC_SHARE on, KEEP on) - against production as it runs now (production.json: strata-g3-1441e-lanes = ee403737,
# --kv-resident 28672, STRATA_SPEC_LANES=2 + STRATA_SOLO_PIPELINE=1).  Binary: strata-g3-1441f-lanes4.
# ~1 h 40 min expected, hard cap WINDOW_S (default 9000 s = 2.5 h, production restart included): a step that no
# longer fits is skipped (logged), steps are ordered by value.  Production is stopped for the window and started again
# unchanged (production.json is never touched), also on abort (trap).  Every test engine logs to results/ (never
# production's log) and spills into /media/raid0/strata-kvspill-bench/lanes4 (never production's tier dir); every test
# HTTP server listens on 127.0.0.1:8091 (never production's port 8090).  A signal (INT/TERM/HUP) before production is
# stopped only exits; after it, it interrupts the running step at once, ends the bench engines and starts production
# again.  An existing results/lanes4 is moved aside to results/lanes4.prev-<time> at the start.
#   run:      cd /media/raid0/StrataQ38FN/linux && ./window-lanes4.sh
#   verdicts: grep -E 'PASS|FAIL|SKIP' results/lanes4/window.log
# Steps (each with PASS/FAIL lines and a time budget):
#   1 boots on production's config: production itself (base), the new binary under production's env (M1b profile:
#     VRAM within 64 MiB of production's), STRATA_SPEC_LANES=4 at --kv-resident 28672 and 32768: no loan, experts
#     100 % resident, VRAM free per card, lanes fitted, first GEN pipelined, engine RSS vs production.  The arm with
#     more lanes (tie: 28672, production's) is LC-MAIN for the rest (LCMAIN_KV=28672|32768 forces it)
#   2 exactness over stdin (greedy + T=0.7 seed 1234, > 32K contexts, exactness settings), every user's tokens ==
#     serial solo spec on the env-off engine (same binary, LC-MAIN's --kv-resident):
#     solo_lc   1 user pipelined (the M1a sequence: < 32K, a 49K prompt, a > 32K resume from the RAM cache)
#     solo_fm   the same with STRATA_PIPELINE_FORCE_MISS=3 (GDN snapshot restores; in the shared profile those
#               snapshots live in lane 2's arena)
#     hold2     2 lanes (HOLD=2; pictures before text in the lane slots)       - whole replies in lanes (KEEP)
#     switch    1 -> 2 -> 3 -> 4 -> 2 -> 1 users on one engine (BGEN while lanes run, BSTOP down; GEN before/after)
#     hold3     3 lanes (HOLD=3), hold4: 4 lanes (HOLD=4)                       - skipped when the boot fitted fewer
#     m1b_new   the new binary under production's env (M1b profile, HOLD=1): lane phase == serial solo spec
#   3 speed: LEVELS 1..6, ABBA x2, production vs LC-MAIN: per-user decode p50 (no loss at 1-2 and 5-6 users, a gain
#     at 3-4), reply identity (1 user strict), lane starts by lane count, request pipelined=1 counts
#   4 16 users: batch_test16 (production's binary env-off vs the new binary env-off vs STRATA_SPEC_LANES=4 NOSTART:
#     byte-identical), HTTP 16 users production vs the new binary in production's env vs LC-MAIN (identical replies)
#   5 the shared arenas poisoned (STRATA_LC_POISON=0x5a) + FORCE_MISS=3: the switch drive again, == serial solo spec
#   6 more boots (if time is left): STRATA_SPEC_LANES=3 at 28672 (STRATA_SOLO_PIPELINE unset: folded in), STRATA_SPEC_LANES=4
#     STRATA_LC_SHARE=0 at 28672
#   7 production restarted unchanged, health checked
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/lanes4; CF=$OUT/cfg
URL=http://127.0.0.1:8090; TPORT=8091; PY=$R/Strata/.venv/bin/python; DRV=$L/lanes_stdin.py
WT=$R/Strata-ll4-lanes; B=$WT/build/strata; NEW=$R/Strata/engine/strata-g3-1441f-lanes4
WANT_COMMIT=f60f4ea5
PRODSRC=$(readlink -f "$C/production.json")                 # what production runs now (its config is only read)
SRV=$R/Strata-143-prod                                      # production's server checkout (ab_quick / run_strata_variant)
TIER=/media/raid0/strata-kvspill-bench/lanes4; ENGLOG=$L/results/engine-lanes4.log
EXACT="--pcie-frac 0 --adapt-every 0 --suffix-draft 0"
XENV=(--env STRATA_IQ_MT_MIN=1 --env STRATA_BATCH_DECODE_SHARE=0 --env STRATA_DECODE_TIMING=1)
NO_OV=(--drop-env STRATA_PIPE_OVERLAP --drop-env STRATA_PIPE_OVERLAP_KVS)   # (an overlapped read sends lane users through the pump)
WINDOW_S=${WINDOW_S:-9000}; RESERVE=480; T0=$(date +%s)
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
# bench_down and production
# (signals are ignored from here on: a second Ctrl-C must not end the script between bench_down and start_prod)
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
    grep -aE '^==>' "$OUT/$name.txt" | cut -c1-300 | sed "s/^/    [$name] /" | tee -a "$OUT/window.log" >/dev/null
    log "step $name: exit $rc"
    bench_down "$name"; return $rc; }
# a step that is not run for a reason other than time: its rc file says skip, so its judges say SKIP too
noskip() { log "step $1: SKIP ($2)"; echo skip > "$OUT/rc_$1"; }
# judge <name> <step> <cmd...>: a check without the GPU (compare / summaries) of what <step> produced, its verdict
# lines into window.log; 'SKIP' when that step was skipped or never ran (no rc file): no verdict from stale files
# (<step> may be a comma list: every one must have run)
step_ran() { [ -f "$OUT/rc_$1" ] && [ "$(cat "$OUT/rc_$1")" != skip ]; }
judge() { local name=$1 from=$2 s; shift 2
    for s in ${from//,/ }; do step_ran "$s" || { log "    [$name] SKIP (step $s skipped)"; return 0; }; done
    timeout 300 "$@" > "$OUT/$name.txt" 2>&1
    grep -aE '^==>' "$OUT/$name.txt" | cut -c1-300 | sed "s/^/    [$name] /" | tee -a "$OUT/window.log" >/dev/null
    grep -qaE '^==>' "$OUT/$name.txt" || log "    [$name] no verdict (see $OUT/$name.txt) FAIL"; }
# scoped <cmd...>: in the bench scope (memory cap as run_strata_variant's; bench_down ends whatever is left of it)
scoped() { systemctl --user stop strata-bench.scope 2>/dev/null; systemctl --user reset-failed strata-bench.scope 2>/dev/null
    systemd-run --user --scope -q --unit=strata-bench -p MemoryMax=200G -p MemorySwapMax=0 "$@"; }
drv() { scoped "$PY" -I "$DRV" "$@"; }
# test HTTP servers: 127.0.0.1:$TPORT (run_strata_variant.sh / ab_quick.sh read PORT and BIND from the env), never
# production's 8090
ab() { ( cd $L && SERVER_DIR=$SRV PORT=$TPORT BIND=127.0.0.1 "$@" ); }
LANES_TIER_ROOT=$TIER
export -f scoped drv ab; export PY DRV L SRV TPORT LANES_TIER_ROOT   # (step runs them under timeout through bash -c)
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }
cfg_exe() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" "$1"; }
jget() { python3 -c "import json,sys
try: print(json.load(open(sys.argv[1])).get(sys.argv[2]) or 0)
except Exception: print(0)" "$1" "$2"; }

# ---- prep (no GPU; production still serving) ------------------------------------------------------------------------
[ -x "$B" ] || { echo "missing $B"; exit 1; }
git -C "$WT" merge-base --is-ancestor $WANT_COMMIT HEAD || { echo "$WT HEAD lacks $WANT_COMMIT (STRATA_SPEC_LANES=2 stays M1b)"; exit 1; }
# (the build may come before the commit's time: compare the sources' times with the binary's, not the commit's)
SRCS=(); for d in src include CMakeLists.txt; do [ -e "$WT/$d" ] && SRCS+=("$WT/$d"); done
NEWER=$(find "${SRCS[@]}" -newer "$B" \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \
        -o -name CMakeLists.txt -o -name '*.cmake' \) | head -3)
[ -z "$NEWER" ] || { echo "$B is older than $NEWER: rebuild first"; exit 1; }
git -C "$WT" diff --quiet HEAD -- src include CMakeLists.txt \
    || { echo "$WT has uncommitted changes under src/, include/ or CMakeLists.txt: the binary is not HEAD"; exit 1; }
for f in "$DRV" "$L/ab_quick.sh" "$L/run_strata_variant.sh" "$L/compare_replies.py" "$R/Strata-1441/tools/batch_test16.py" \
         "$R/Strata-1441/tools/vision_batch_check.py" "$PRODSRC"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
[ -x "$DRV" ] || { echo "$DRV is not executable (chmod +x)"; exit 1; }
"$PY" -I "$DRV" --help > /dev/null || { echo "$DRV does not load"; exit 1; }
curl -s -m 2 "http://127.0.0.1:$TPORT/health" > /dev/null && { echo "port $TPORT already serves something: the test servers need it"; exit 1; }
PEXE=$(cfg_exe "$PRODSRC"); [ -x "$PEXE" ] || { echo "production's exe $PEXE missing"; exit 1; }
python3 - "$PRODSRC" <<'PYEOF' || exit 1
import json, sys
c = json.load(open(sys.argv[1])); e = c["env"]; a = c["args"]
want = {"STRATA_SPEC_LANES": "2", "STRATA_SOLO_PIPELINE": "1"}
bad = {k: e.get(k) for k, v in want.items() if e.get(k) != v}
if bad or "--kv-resident" not in a or int(c.get("parallel") or 0) != 16:
    print(f"production's config {sys.argv[1]} is not the one this window is written for (want {want}, --kv-resident, "
          f"parallel 16; differs: {bad})"); sys.exit(1)
PYEOF
# (before the copy: never overwrite the binary production runs)
[ "$(readlink -f "$PEXE")" != "$(readlink -f "$NEW")" ] || { echo "production already runs $NEW: nothing to compare against"; exit 1; }
# the window restores production to running: it must be running now (an inactive service is not started behind the
# user's back, and the PROD base boot / HTTP arm compare against what it runs)
systemctl --user is-active -q strata.service || { echo "strata.service is not active: start it first"; exit 1; }
cmp -s "$B" "$NEW" || { cp "$B" "$NEW" && chmod +x "$NEW"; } || { echo "cannot copy $B to $NEW"; exit 1; }
# stale results (still before production is stopped): an earlier window's results/lanes4 moves aside, its HTTP engine
# run logs go with it (their names are fixed: a skipped step must not leave an old log for a judge)
if [ -e "$OUT" ]; then PREV=$OUT.prev-$(date +%Y%m%d-%H%M%S); mv "$OUT" "$PREV" || { echo "cannot move $OUT aside"; exit 1; }
    mv "$L"/results/engine-ab-l4*-*.run.log "$PREV"/ 2>/dev/null; fi       # (kept with the window they belong to)
rm -f "$L"/results/engine-ab-l4*-*.run.log
mkdir -p "$OUT" "$CF" "$TIER"
# sweep.jsonl rows of earlier windows with the same labels must not count (the sweep judges read from here on)
SWEEP0=$(cat "$L/results/sweep.jsonl" 2>/dev/null | wc -l)
log "window prep: $(basename $NEW) sha256 $(sha256sum $NEW | cut -c1-16) = $WT $(git -C $WT rev-parse --short HEAD) ($(git -C $WT log -1 --format=%s HEAD | cut -c1-80)); production: $(basename $PRODSRC), exe $(basename $PEXE) sha256 $(sha256sum $PEXE | cut -c1-16)"

# configs, all from production's config: exe, the bench tier dir, the bench log, then the arm's changes:
# K=V sets an env key, -K drops one, ARG--opt=V sets an argument's value
mkcfg() {   # $1 name, $2 exe, then changes ...
    local name=$1 exe=$2; shift 2
    python3 - "$PRODSRC" "$CF/$name.json" "$exe" "$TIER/http" "$ENGLOG" "$@" <<'PYEOF' || exit 1
import json, sys
src, dst, exe, tier, log = sys.argv[1:6]
c = json.load(open(src)); c["exe"] = exe; c["log"] = log; e = c["env"]; e["STRATA_CONV_TIER_DIR"] = tier
for kv in sys.argv[6:]:
    if kv.startswith("ARG"):
        k, v = kv[3:].split("=", 1); a = c["args"]; i = a.index(k); a[i + 1] = v
    elif kv.startswith("-"):
        e.pop(kv[1:], None)
    else:
        k, v = kv.split("=", 1); e[k] = v
json.dump(c, open(dst, "w"), indent=1)
PYEOF
}
LOGS=(STRATA_SPEC_LANES_LOG=1 STRATA_SOLO_PIPELINE_LOG=1)
mkcfg LC-PROD      "$PEXE"                                       # production as it runs (HTTP arm)
mkcfg LC-PRODLOG   "$PEXE" "${LOGS[@]}"                          # the same with the log lines (boot)
mkcfg LC-M1BNEW    "$NEW"                                        # the new binary in production's env (M1b profile)
mkcfg LC-M1BNEWLOG "$NEW" "${LOGS[@]}"
mkcfg LC4-28 "$NEW" STRATA_SPEC_LANES=4 STRATA_SOLO_PIPELINE=1 "${LOGS[@]}" ARG--kv-resident=28672
mkcfg LC4-32 "$NEW" STRATA_SPEC_LANES=4 STRATA_SOLO_PIPELINE=1 "${LOGS[@]}" ARG--kv-resident=32768
mkcfg LC-OFF-28672 "$NEW" -STRATA_SPEC_LANES -STRATA_SOLO_PIPELINE ARG--kv-resident=28672   # env unset: the reference
mkcfg LC-OFF-32768 "$NEW" -STRATA_SPEC_LANES -STRATA_SOLO_PIPELINE ARG--kv-resident=32768
mkcfg BT-PRODX   "$PEXE" -STRATA_SPEC_LANES -STRATA_SOLO_PIPELINE  # today's batch path (production's binary, env unset)
mkcfg BT-OFF     "$NEW"  -STRATA_SPEC_LANES -STRATA_SOLO_PIPELINE  # the new binary, env unset
mkcfg BT-NOSTART "$NEW" STRATA_SPEC_LANES=4 STRATA_SOLO_PIPELINE=1 STRATA_SPEC_LANES_NOSTART=1 STRATA_SPEC_LANES_GIVEBACK=0 STRATA_SPEC_LANES_LOG=1
# (LC3-28 without STRATA_SOLO_PIPELINE: the solo pipeline folded in by STRATA_SPEC_LANES alone - its boot wants the
#  first GEN pipelined=1; every other arm sets it as production does)
mkcfg LC3-28    "$NEW" STRATA_SPEC_LANES=3 -STRATA_SOLO_PIPELINE "${LOGS[@]}" ARG--kv-resident=28672
mkcfg LC4-28-S0 "$NEW" STRATA_SPEC_LANES=4 STRATA_SOLO_PIPELINE=1 STRATA_LC_SHARE=0 "${LOGS[@]}" ARG--kv-resident=28672
mkdir -p "$TIER/http"
# a copy of a config with a fresh, empty tier dir of its own (batch_test16's 40K prompt is tiered: a later arm must
# not find an earlier arm's copy under STRATA_CONV_TIER_WARM)
tiercfg() { local d=$TIER/$2; rm -rf "$d"; mkdir -p "$d"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; json.dump(c,open(sys.argv[3],'w'),indent=1)" \
        "$1" "$d" "$CF/$2.json"; echo "$CF/$2.json"; }

# pictures for the vision case: the config's image encoder runs on the CPU ("gpu": false), so before the window
# (production still serves: off its --aux-cpus 65-127, niced, capped)
timeout 1200 taskset -c 0-63 nice -n 15 "$PY" -I "$DRV" --mode encode --config "$PRODSRC" --out "$OUT/pics" > "$OUT/encode.txt" 2>&1
grep -aE '^==>' "$OUT/encode.txt" | sed 's/^/    [encode] /' | tee -a "$OUT/window.log" >/dev/null
PICS=$OUT/pics/pics.json; [ -f "$PICS" ] || log "    [encode] no pictures: the L_vis group is skipped FAIL"

log "window start ($(( WINDOW_S / 60 )) min cap): stopping strata.service"
STOPPED=1; trap 'abort signal' INT TERM HUP     # from here on an abort ends the bench engines and restarts production
systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || abort "GPUs not free"

# ---- 1. boots: production settings (no exactness flags), one short GEN each ------------------------------------------
BOOTCHK=(--all-resident --want-pipelined)
step boot_PROD 180 600 drv --mode boot --config $CF/LC-PRODLOG.json --label PROD --expect m1b "${BOOTCHK[@]}" \
    --log $OUT/engine-boot_PROD.log --out $OUT/boot_PROD.json
BASE=(--base $OUT/boot_PROD.json --base-name production)
step boot_M1BNEW 180 600 drv --mode boot --config $CF/LC-M1BNEWLOG.json --label M1BNEW --expect m1b "${BOOTCHK[@]}" "${BASE[@]}" \
    --vram-tol 64 --rss-max 256 --log $OUT/engine-boot_M1BNEW.log --out $OUT/boot_M1BNEW.json
step boot_LC4_28 180 600 drv --mode boot --config $CF/LC4-28.json --label LC4-28 --expect lc --want-lanes 2 "${BOOTCHK[@]}" "${BASE[@]}" \
    --rss-max 512 --log $OUT/engine-boot_LC4_28.log --out $OUT/boot_LC4_28.json
step boot_LC4_32 180 600 drv --mode boot --config $CF/LC4-32.json --label LC4-32 --expect lc --want-lanes 2 "${BOOTCHK[@]}" "${BASE[@]}" \
    --rss-max 512 --log $OUT/engine-boot_LC4_32.log --out $OUT/boot_LC4_32.json
for b in LC4_28 LC4_32 M1BNEW; do
    grep -hE 'STRATA_SPEC_LANES|STRATA_SOLO_PIPELINE|VRAM free with everything' "$OUT/boot_$b.txt" 2>/dev/null \
        | grep -vE '^==>' | cut -c1-260 | sed "s/^ */    (boot $b) /" | tee -a "$OUT/window.log" >/dev/null; done
N28=0; grep -q '^==> boot LC4-28: no loan .* PASS$' "$OUT/boot_LC4_28.txt" 2>/dev/null && N28=$(jget $OUT/boot_LC4_28.json lanes)
N32=0; grep -q '^==> boot LC4-32: no loan .* PASS$' "$OUT/boot_LC4_32.txt" 2>/dev/null && N32=$(jget $OUT/boot_LC4_32.json lanes)
M1BNEW_ON=0; grep -q '^==> boot M1BNEW: no loan .* PASS$' "$OUT/boot_M1BNEW.txt" 2>/dev/null && M1BNEW_ON=1
# LC-MAIN: the arm with more lanes; a tie goes to 28672 (production's: the speed A/B then differs in the lanes only)
if [ "${LCMAIN_KV:-}" = 32768 ] || { [ -z "${LCMAIN_KV:-}" ] && [ "$N32" -gt "$N28" ]; }; then KV=32768; NL=$N32; MAINB=LC4-32
else KV=28672; NL=$N28; MAINB=LC4-28; fi
cp "$CF/$MAINB.json" "$CF/LC-MAIN.json"; REF_CFG=$CF/LC-OFF-$KV.json
log "gates: lanes fitted at 28672: $N28, at 32768: $N32 (0: boot failed); LC-MAIN = $MAINB ($NL lanes); new binary in production's env PASS: $M1BNEW_ON"
LANES_ON=0; [ "$NL" -ge 2 ] && LANES_ON=1

# ---- 2. exactness ----------------------------------------------------------------------------------------------------
# 2a the env-off reference over stdin (exactness settings; LC-MAIN's --kv-resident): the solo requests, then every
#    group's and the switch drive's prompts as serial solo GEN
if [ $LANES_ON = 1 ] || [ $M1BNEW_ON = 1 ]; then
  step ref 540 1800 drv --mode solo --seq ref4 --config $REF_CFG --label REF --extra="$EXACT" "${XENV[@]}" \
      --log $OUT/engine-ref.log --out $OUT/ref.json
else noskip ref "no lane arm booted"; fi
LX=(--env STRATA_SPEC_LANES_LOG=1 --env STRATA_SOLO_PIPELINE_LOG=1)
if [ $LANES_ON = 1 ]; then
  # 2b 1 user, pipelined on the solo path (folded in by STRATA_SPEC_LANES): identical, 'request pipelined=1' each, the
  #    > 32K resume restored and streamed
  step solo_lc 300 1200 drv --mode solo --seq m1a --config $CF/LC-MAIN.json --label LC-SOLO --extra="$EXACT" "${XENV[@]}" \
      --log $OUT/engine-solo_lc.log --out $OUT/solo_lc.json
  judge cmp_solo_lc ref,solo_lc "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/solo_lc.json --seq m1a \
      --what "1 user pipelined (LC-MAIN)" --need-pipelined --need-resume
  # 2c forced rollbacks: every 3rd speculative window gets a wrong first token (GDN snapshot restores on CUDA1/2)
  step solo_fm 240 900 drv --mode solo --seq fm --config $CF/LC-MAIN.json --label LC-FM --extra="$EXACT" "${XENV[@]}" \
      --env STRATA_PIPELINE_DEBUG=1 --env STRATA_PIPELINE_FORCE_MISS=3 --log $OUT/engine-solo_fm.log --out $OUT/solo_fm.json
  judge cmp_solo_fm ref,solo_fm,solo_lc "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/solo_fm.json --seq fm \
      --what "1 user FORCE_MISS=3 (LC-MAIN)" --need-pipelined --need-rollback --base-got $OUT/solo_lc.json
  # 2d 2 lanes in the shared profile (HOLD=2, overlap off): each reply wholly in lanes (KEEP) == serial solo spec
  step hold2 300 1200 drv --mode lanes --pairs g2s --pics "$PICS" --config $CF/LC-MAIN.json --label HOLD2 --extra="$EXACT" \
      "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=2 "${NO_OV[@]}" --log $OUT/engine-hold2.log --out $OUT/hold2.json
  judge cmp_hold2 ref,hold2 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold2.json --pairs g2s \
      --what "2 lanes (HOLD=2)" --hold --full
  # 2e the mode switches 1 -> 2 -> 3 -> 4 -> 2 -> 1 (up to the lanes the boot fitted)
  step switch 300 1200 drv --mode switch --config $CF/LC-MAIN.json --label SWITCH --extra="$EXACT" "${XENV[@]}" "${LX[@]}" \
      --env STRATA_SPEC_LANES_HOLD=2 "${NO_OV[@]}" --log $OUT/engine-switch.log --out $OUT/switch.json
  judge cmp_switch ref,switch "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/switch.json --what "switch 1-2-3-4-2-1"
  # 2f 3 and 4 lanes (HOLD=3 / 4): each group's slots all admitted, then every reply wholly in lanes
  if [ "$NL" -ge 3 ]; then
    step hold3 300 1200 drv --mode lanes --pairs g3 --config $CF/LC-MAIN.json --label HOLD3 --extra="$EXACT" \
        "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=3 "${NO_OV[@]}" --log $OUT/engine-hold3.log --out $OUT/hold3.json
    judge cmp_hold3 ref,hold3 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold3.json --pairs g3 \
        --what "3 lanes (HOLD=3)" --hold --full
  else noskip hold3 "the boot fitted $NL lanes"; fi
  if [ "$NL" -ge 4 ]; then
    step hold4 300 1200 drv --mode lanes --pairs g4 --config $CF/LC-MAIN.json --label HOLD4 --extra="$EXACT" \
        "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=4 "${NO_OV[@]}" --log $OUT/engine-hold4.log --out $OUT/hold4.json
    judge cmp_hold4 ref,hold4 "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold4.json --pairs g4 \
        --what "4 lanes (HOLD=4)" --hold --full
  else noskip hold4 "the boot fitted $NL lanes"; fi
else
  for s in solo_lc solo_fm hold2 switch hold3 hold4; do noskip $s "no lane arm booted"; done
fi
# 2g the new binary in production's env (M1b profile, KEEP=0, HOLD=1): the lane phase == serial solo spec (M1b's
#    pairs; the reference has LC-MAIN's --kv-resident: K/V resident or streamed gives the same tokens)
if [ $M1BNEW_ON = 1 ]; then
  step m1b_new 300 1200 drv --mode lanes --pairs all --pics "$PICS" --config $CF/LC-M1BNEW.json --label M1BNEW --extra="$EXACT" \
      "${XENV[@]}" "${LX[@]}" --env STRATA_SPEC_LANES_HOLD=1 "${NO_OV[@]}" --log $OUT/engine-m1b_new.log --out $OUT/m1b_new.json
  judge cmp_m1b_new ref,m1b_new "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/m1b_new.json --pairs all \
      --what "M1b profile, new binary (HOLD=1)" --hold
else noskip m1b_new "the new binary's boot in production's env failed"; fi

# ---- 3. speed: 1..6 users over HTTP, production settings, production vs LC-MAIN, ABBA x2 -----------------------------
if [ $LANES_ON = 1 ]; then
  step speed 1500 3600 ab env LEVELS=1,2,3,4,5,6 REPS=2 ORDER=abba ROUNDS=2 ./ab_quick.sh l4spd $CF/LC-PROD.json $CF/LC-MAIN.json
  judge speed_ident speed "$PY" -I "$DRV" --mode abcheck --file $OUT/speed.txt \
      --what "LC-MAIN replies (strict at 1 user: solo path both; INFO beyond: lanes / pump windows by timing)" --expect identical --strict-users 1
  DMIN="1:-1.5,2:-1.5,3:10,4:8,5:-2,6:-2"
  judge speed_dec speed "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag l4spd --arms LC-PROD,LC-MAIN --reps 2 --dec-min "$DMIN"
  SIZES=$(seq -s, 2 "$NL")
  # (the -long run log holds the -short run's lines too: one engine start, run_strata_variant cuts from its 'engine
  #  started' line both times - so only the -long logs, else every short-run line counts twice)
  judge speed_lanes speed "$PY" -I "$DRV" --mode lanelog --what "LC-MAIN HTTP engines" --want-sizes "$SIZES" \
      "$L/results/engine-ab-l4spd-LC-MAIN-r*-long.run.log"
  judge speed_pl speed "$PY" -I "$DRV" --mode pllog --what "LC-MAIN HTTP engines" "$L/results/engine-ab-l4spd-LC-MAIN-r*-long.run.log"
  grep -hE '^ *(short|long) +[0-9]+ users' "$OUT/speed.txt" | sed 's/^/    /' | tee -a "$OUT/window.log" >/dev/null
  grep -hE '^ *(short|long) +[0-9]+ users:' "$OUT/speed_dec.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$OUT/window.log" >/dev/null
else noskip speed "no lane arm booted"; fi

# ---- 4. 16 users ------------------------------------------------------------------------------------------------------
# 4a batch_test16 (greedy, 16 slots, a cold 40K read beside 15 decoding, overlap on as in production): today's batch
#    path (production's binary, env unset), the new binary env unset, and STRATA_SPEC_LANES=4 NOSTART (+GIVEBACK=0:
#    everything of the lanes beside the pump but never a lane start) - byte-identical
for arm in PRODX OFF NOSTART; do
    cfg=$(tiercfg $CF/BT-$arm.json bt_$arm); exe=$(cfg_exe "$cfg")
    step bt_$arm 240 900 scoped bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-bt_$arm.log env $(cfg_env $cfg) python3 tools/batch_test16.py --exe $exe --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --dump $OUT/bt_$arm.json"
done
python3 - "$OUT" <<'PYEOF' | tee -a "$OUT/window.log"
import json, sys
o = sys.argv[1]
def ran(step):    # the step's rc file: missing or 'skip' -> the step did not run (no verdict from a stale dump)
    try: return open(f"{o}/rc_{step}").read().strip() != "skip"
    except OSError: return False
def dump(arm):
    if not ran(f"bt_{arm}"):
        print(f"    [bt] {arm}: SKIP (step bt_{arm} skipped)"); return None
    try: return json.load(open(f"{o}/bt_{arm}.json"))["batch"]
    except Exception as e: print(f"    [bt] ==> {arm}: no dump ({e}) FAIL"); return None
def cmp(a_name, a, b_name, b):
    bad = [i for i in range(max(len(a), len(b))) if i >= len(a) or i >= len(b) or a[i] != b[i]]
    print(f"    [bt] ==> batch_test16 {b_name} vs {a_name}: {len(a) - len([i for i in bad if i < len(a)])}/{len(a)} slots "
          f"token-identical" + (f" (differ: {bad}) FAIL" if bad or not a else " PASS"))
d = {arm: dump(arm) for arm in ("PRODX", "OFF", "NOSTART")}
for arm in ("OFF", "NOSTART"):
    if d["PRODX"] is not None and d[arm] is not None:
        cmp("production's binary (env unset)", d["PRODX"], arm, d[arm])
if ran("bt_NOSTART"):
    try: t = open(f"{o}/engine-bt_NOSTART.log", errors="replace").read().splitlines()
    except OSError: t = None
    if t is None:
        print("    [bt] ==> NOSTART engine log missing FAIL")
    else:
        nb = sum("NOSTART - the lanes never start" in l for l in t)
        off = sum("STRATA_SPEC_LANES is off" in l or "is not supported" in l for l in t)
        st = sum("STRATA_SPEC_LANES: lanes start" in l for l in t)
        print(f"    [bt] ==> NOSTART boot: {nb} 'NOSTART - the lanes never start' lines (want 1), {off} 'is off' lines "
              f"(want 0), {st} lane starts (want 0) {'PASS' if nb == 1 and off == 0 and st == 0 else 'FAIL'}")
PYEOF
for arm in OFF NOSTART; do f=$OUT/engine-bt_$arm.log
    log "    [bt] $arm: $(grep -c 'overlapped read of' $f 2>/dev/null) overlapped reads, $(grep -c 'drafter catch-up' $f 2>/dev/null) catch-up lines, admissions $(grep -aoE 'admissions [0-9.]+s' $OUT/bt_$arm.txt 2>/dev/null | head -1)"; done
# 4b HTTP 16 users: production vs the new binary in production's env vs LC-MAIN, identical replies (the lanes may run in
#    the ramp and the tail, when 2-4 lane slots decode alone: lane starts counted below)
HTTP16=($CF/LC-PROD.json); [ $M1BNEW_ON = 1 ] && HTTP16+=($CF/LC-M1BNEW.json); [ $LANES_ON = 1 ] && HTTP16+=($CF/LC-MAIN.json)
if [ ${#HTTP16[@]} -ge 2 ]; then
  step http16 720 2400 ab env LEVELS=16 REPS=1 ROUNDS=2 ./ab_quick.sh l4h16 "${HTTP16[@]}"
  judge http16_ident http16 "$PY" -I "$DRV" --mode abcheck --file $OUT/http16.txt --what "16-user replies vs production" --expect identical
  [ $LANES_ON = 1 ] && judge http16_lanes http16 "$PY" -I "$DRV" --mode lanelog --expect any --what "LC-MAIN 16-user engines (lane starts: ramp/tail)" \
      "$L/results/engine-ab-l4h16-LC-MAIN-long.run.log"
  grep -hE '^ *(short|long) +[0-9]+ users' "$OUT/http16.txt" | sed 's/^/    (one run each: |delta| < 1.3 % is not resolved) /' | tee -a "$OUT/window.log" >/dev/null
else noskip http16 "no new arm booted"; fi

# ---- 5. the shared arenas poisoned at every switch (+ forced rollbacks on the solo path) -----------------------------
if [ $LANES_ON = 1 ]; then
  step poison 300 1200 drv --mode switch --config $CF/LC-MAIN.json --label POISON --extra="$EXACT" "${XENV[@]}" "${LX[@]}" \
      --env STRATA_SPEC_LANES_HOLD=2 --env STRATA_LC_POISON=0x5a --env STRATA_PIPELINE_DEBUG=1 --env STRATA_PIPELINE_FORCE_MISS=3 \
      "${NO_OV[@]}" --log $OUT/engine-poison.log --out $OUT/poison.json
  # (--need-rollback: the two GEN must show more rollbacks than in the switch run without FORCE_MISS - else the forced
  #  misses never happened and the 'FORCE_MISS=3' part proves nothing)
  judge cmp_poison ref,poison,switch "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/poison.json \
      --what "switch, arenas poisoned 0x5a + FORCE_MISS=3" --need-rollback --base-got $OUT/switch.json
  step_ran poison && { n=$(grep -c 'POISON - the shared arenas filled with 0x5a' $OUT/engine-poison.log 2>/dev/null)
      log "    [poison] ==> boot line 'POISON - the shared arenas filled with 0x5a': $n (want 1) $([ "$n" = 1 ] && echo PASS || echo FAIL)"; }
else noskip poison "no lane arm booted"; fi

# ---- 6. more boots (the matrix's other corners), if time is left ------------------------------------------------------
step boot_LC3_28 180 600 drv --mode boot --config $CF/LC3-28.json --label LC3-28 --expect lc --want-lanes 2 --max-lanes 3 "${BOOTCHK[@]}" "${BASE[@]}" \
    --rss-max 512 --log $OUT/engine-boot_LC3_28.log --out $OUT/boot_LC3_28.json
step boot_LC4_28_S0 180 600 drv --mode boot --config $CF/LC4-28-S0.json --label LC4-28-S0 --expect lc --want-lanes 2 "${BOOTCHK[@]}" "${BASE[@]}" \
    --rss-max 512 --log $OUT/engine-boot_LC4_28_S0.log --out $OUT/boot_LC4_28_S0.json

# ---- 7. production back, unchanged ------------------------------------------------------------------------------------
# (the abort trap stays armed through bench_down: a signal there still restarts production; then signals are ignored
#  until production is up)
bench_down end
trap '' INT TERM HUP
start_prod || { log "production start failed: one more try"; systemctl --user restart strata.service; start_prod; } \
    || log "PRODUCTION DID NOT COME BACK - check strata.service now"
trap - INT TERM HUP
P=$(systemctl --user show -p MainPID --value strata.service)
log "production engine: $(for c in $(pgrep -P "$P" 2>/dev/null); do readlink /proc/$c/exe; done | tr '\n' ' ')"
rm -rf "$TIER"
log "window done after $(( ($(date +%s) - T0) / 60 )) min"
grep -hE 'PASS|FAIL|SKIP' "$OUT/window.log" | grep -v 'grep' | tail -80
