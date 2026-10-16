#!/usr/bin/env bash
# GPU window for branch ll3-all (Strata-ll3-lanes, HEAD >= ee403737): M1a STRATA_SOLO_PIPELINE=1 (pipelined solo spec
# beside --batch) and M1b STRATA_SPEC_LANES=2 (two spec lanes on slots 0/4), each its own arm (they do not both fit
# CUDA0), against production's behaviour: the env-off arms on the same binary (strata-g3-1441e-lanes) and, where it
# says so, production's binary strata-g3-1441b-ovkv.  Then s28r + the overlapped read, and a prefill profile.
# ~2 h expected, hard cap WINDOW_S (default 9000 s = 2.5 h, production restart included): a step that no longer fits
# is skipped (logged), steps are ordered by value.  Production is stopped for the window and started again unchanged
# (production.json is never touched), also on abort (trap).  Every test engine logs to results/ (never production's
# log) and spills into /media/raid0/strata-kvspill-bench/lanes (never production's tier dir); every test HTTP server
# listens on 127.0.0.1:8091 (never production's port 8090).  A signal (INT/TERM/HUP) before production is stopped only
# exits; after it, it interrupts the running step at once, ends the bench engines and starts production again.
# An existing results/lanes is moved aside to results/lanes.prev-<time> at the start.
#   run:      cd /media/raid0/StrataQ38FN/linux && ./window-lanes.sh
#   verdicts: grep -E 'PASS|FAIL|SKIP' results/lanes/window.log
# Steps (each with PASS/FAIL lines and a time budget):
#   1 boots: env-off, M1a, M1b on production's config: no loan, VRAM free per card, the M1a / M1b cost lines
#   2 the env-off reference over stdin: 25 requests (11 solo + the 14 halves of the 7 lane pairs), its verdict
#     'solo REF: 25/25 requests done'
#   2 M1a: exactness over stdin vs the env-off engine (greedy + T=0.7 seed 1234, < 32K, a 49K prompt, a > 32K resume
#     from the conversation cache with the K/V-streaming line, FORCE_MISS rollbacks), 'request pipelined=1' per request;
#     1-user speed over HTTP (ab_quick, ABBA x2, short + 1.2K)
#   3 M1b: exactness over stdin with STRATA_SPEC_LANES_HOLD (BGEN 0, BADM 0 1, BGEN 4) vs serial solo GEN on the
#     env-off engine (greedy, T=0.7 seed, > 32K, text after pictures in the lane slots, a per-request pcie_frac), the
#     same with STRATA_SPEC_LANES_ONE=1 (whole replies in lanes); 2-user speed (LEVELS "1,2,4", ABBA x2, per-user
#     decode p50, lanes log); 16-user regression with NOSTART (batch_test16 sha + HTTP replies byte-identical);
#     HOLD with the overlapped read on (ee403737: the gate keeps the held slot's admission read serial)
#   4 s28r (11,24,37, --kv-resident 28672, SPLIT_OWN_RESERVE 768) + STRATA_PIPE_OVERLAP=1/_KVS=1: no loan, VRAM,
#     16-user ab_quick vs production's config, identical replies
#   5 prefill tok/s by prompt length (170 .. 64000 tokens; idle and beside 15 decoding slots) on production's config
#   6 production restarted unchanged, health checked
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/lanes; CF=$OUT/cfg
URL=http://127.0.0.1:8090; TPORT=8091; PY=$R/Strata/.venv/bin/python; DRV=$L/lanes_stdin.py
WT=$R/Strata-ll3-lanes; B=$WT/build/strata; NEW=$R/Strata/engine/strata-g3-1441e-lanes
PROD=$C/T16-1441d-ov-prod262r-vision-cpu.json; S28=$C/T16-1441c-s28r-prod262r-vision-cpu.json
S28_EXE=${S28_EXE:-$R/Strata/engine/strata-g3-1441b-ovkv}   # production's binary: s28r + overlap needs only a config
SRV=$R/Strata-143-prod                                      # production's server checkout (ab_quick / run_strata_variant)
TIER=/media/raid0/strata-kvspill-bench/lanes; ENGLOG=$L/results/engine-lanes.log
EXACT="--pcie-frac 0 --adapt-every 0 --suffix-draft 0"
XENV=(--env STRATA_IQ_MT_MIN=1 --env STRATA_BATCH_DECODE_SHARE=0 --env STRATA_DECODE_TIMING=1)
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
abort() { trap - INT TERM HUP; log "window aborted: $*"
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
export -f scoped drv ab; export PY DRV L SRV TPORT   # (step runs them under timeout through bash -c)
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }

# ---- prep (no GPU; production still serving) ------------------------------------------------------------------------
[ -x "$B" ] || { echo "missing $B"; exit 1; }
git -C "$WT" merge-base --is-ancestor ee403737 HEAD || { echo "$WT HEAD lacks ee403737 (HOLD x overlap fix)"; exit 1; }
# (the build came 11 s before ee403737's commit time: compare the sources' times with the binary's, not the commit's)
SRCS=(); for d in src include CMakeLists.txt; do [ -e "$WT/$d" ] && SRCS+=("$WT/$d"); done
NEWER=$(find "${SRCS[@]}" -newer "$B" \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \
        -o -name CMakeLists.txt -o -name '*.cmake' \) | head -3)
[ -z "$NEWER" ] || { echo "$B is older than $NEWER: rebuild first"; exit 1; }
git -C "$WT" diff --quiet HEAD -- src include CMakeLists.txt \
    || { echo "$WT has uncommitted changes under src/, include/ or CMakeLists.txt: the binary is not HEAD"; exit 1; }
for f in "$DRV" "$L/ab_quick.sh" "$L/run_strata_variant.sh" "$R/Strata-1441/tools/batch_test16.py" \
         "$R/Strata-1441/tools/vision_batch_check.py" "$PROD" "$S28" "$S28_EXE"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
[ -x "$DRV" ] || { echo "$DRV is not executable (chmod +x)"; exit 1; }
"$PY" -I "$DRV" --help > /dev/null || { echo "$DRV does not load"; exit 1; }
curl -s -m 2 "http://127.0.0.1:$TPORT/health" > /dev/null && { echo "port $TPORT already serves something: the test servers need it"; exit 1; }
[ "$(readlink $C/production.json)" = "$(basename $PROD)" ] || echo "WARNING: production.json -> $(readlink $C/production.json), not $(basename $PROD)"
cmp -s "$B" "$NEW" || { cp "$B" "$NEW" && chmod +x "$NEW"; } || { echo "cannot copy $B to $NEW"; exit 1; }
PEXE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" "$PROD")
# stale results (still before production is stopped): an earlier window's results/lanes moves aside, its HTTP engine
# run logs go (their names are fixed: a skipped step must not leave an old log for a judge)
if [ -e "$OUT" ]; then PREV=$OUT.prev-$(date +%Y%m%d-%H%M%S); mv "$OUT" "$PREV" || { echo "cannot move $OUT aside"; exit 1; }
    mv "$L"/results/engine-ab-ln*-*.run.log "$PREV"/ 2>/dev/null; fi       # (kept with the window they belong to)
rm -f "$L"/results/engine-ab-ln*-*.run.log
mkdir -p "$OUT" "$CF" "$TIER"
# sweep.jsonl rows of earlier windows with the same labels must not count (the sweep judges read from here on)
SWEEP0=$(cat "$L/results/sweep.jsonl" 2>/dev/null | wc -l)
log "window prep: $(basename $NEW) sha256 $(sha256sum $NEW | cut -c1-16) = $WT $(git -C $WT rev-parse --short HEAD) ($(git -C $WT log -1 --format=%s HEAD | cut -c1-80)); production exe $(basename $PEXE)"

# configs, all from production's config: exe, the bench tier dir, the bench log, the arm's env
mkcfg() {   # $1 name, $2 source config, $3 exe, then K=V ...
    local name=$1 src=$2 exe=$3; shift 3
    python3 - "$src" "$CF/$name.json" "$exe" "$TIER/http" "$ENGLOG" "$@" <<'PYEOF' || exit 1
import json, sys
src, dst, exe, tier, log = sys.argv[1:6]
c = json.load(open(src)); c["exe"] = exe; c["log"] = log; e = c["env"]; e["STRATA_CONV_TIER_DIR"] = tier
for kv in sys.argv[6:]:
    k, v = kv.split("=", 1); e[k] = v
json.dump(c, open(dst, "w"), indent=1)
PYEOF
}
mkcfg LN-PROD    "$PROD" "$PEXE"
mkcfg LN-OFF     "$PROD" "$NEW"
mkcfg LN-M1A     "$PROD" "$NEW" STRATA_SOLO_PIPELINE=1 STRATA_SOLO_PIPELINE_LOG=1
mkcfg LN-M1B     "$PROD" "$NEW" STRATA_SPEC_LANES=2 STRATA_SPEC_LANES_LOG=1
mkcfg LN-NOSTART "$PROD" "$NEW" STRATA_SPEC_LANES=2 STRATA_SPEC_LANES_LOG=1 STRATA_SPEC_LANES_NOSTART=1 STRATA_SPEC_LANES_GIVEBACK=0
mkcfg LN-S28ROV  "$S28"  "$S28_EXE" STRATA_PIPE_OVERLAP=1 STRATA_PIPE_OVERLAP_KVS=1
mkdir -p "$TIER/http"
# a copy of a config with a fresh, empty tier dir of its own (batch_test16's 40K prompt is tiered: a later arm must
# not find an earlier arm's copy under STRATA_CONV_TIER_WARM)
tiercfg() { local d=$TIER/$2; rm -rf "$d"; mkdir -p "$d"
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['env']['STRATA_CONV_TIER_DIR']=sys.argv[2]; json.dump(c,open(sys.argv[3],'w'),indent=1)" \
        "$1" "$d" "$CF/$2.json"; echo "$CF/$2.json"; }

# pictures for the vision case: the config's image encoder runs on the CPU ("gpu": false), so before the window
"$PY" -I "$DRV" --mode encode --config "$PROD" --out "$OUT/pics" > "$OUT/encode.txt" 2>&1
grep -aE '^==>' "$OUT/encode.txt" | sed 's/^/    [encode] /' | tee -a "$OUT/window.log" >/dev/null
PICS=$OUT/pics/pics.json; [ -f "$PICS" ] || log "    [encode] no pictures: the L_vis pair is skipped FAIL"

log "window start ($(( WINDOW_S / 60 )) min cap): stopping strata.service"
STOPPED=1; trap 'abort signal' INT TERM HUP     # from here on an abort ends the bench engines and restarts production
systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || abort "GPUs not free"

# ---- 1. boots: production settings (no exactness flags), one short GEN each ------------------------------------------
step boot_OFF 240 600 drv --mode boot --config $CF/LN-OFF.json --label LN-OFF --log $OUT/engine-boot_OFF.log --out $OUT/boot_OFF.json
step boot_M1A 240 600 drv --mode boot --config $CF/LN-M1A.json --label LN-M1A --expect m1a --base $OUT/boot_OFF.json --log $OUT/engine-boot_M1A.log --out $OUT/boot_M1A.json
step boot_M1B 240 600 drv --mode boot --config $CF/LN-M1B.json --label LN-M1B --expect m1b --base $OUT/boot_OFF.json --log $OUT/engine-boot_M1B.log --out $OUT/boot_M1B.json
grep -hE 'took|kept out|MiB free, the|drafter|stays serial|is off|too little room' "$OUT/boot_M1A.txt" "$OUT/boot_M1B.txt" 2>/dev/null \
    | cut -c1-260 | sed 's/^ */    (boot) /' | tee -a "$OUT/window.log" >/dev/null
M1A_ON=0; grep -q 'first GEN pipelined=1 yes' "$OUT/boot_M1A.txt" 2>/dev/null && M1A_ON=1
M1B_ON=0; grep -q 'spec_lanes=2 lane_slots=0,4 yes' "$OUT/boot_M1B.txt" 2>/dev/null && M1B_ON=1
log "gates: M1a pipelines the solo path: $M1A_ON; M1b lanes on: $M1B_ON (0: that arm's steps are skipped)"

# ---- 2. M1a ----------------------------------------------------------------------------------------------------------
# 2a the env-off reference over stdin (exactness settings), shared with step 3: the M1a requests, then every lane
#    pair's two prompts as serial solo GEN in admission order
if [ $M1A_ON = 1 ] || [ $M1B_ON = 1 ]; then
  step ref 420 1500 drv --mode solo --seq ref --config $CF/LN-OFF.json --label REF --extra="$EXACT" "${XENV[@]}" \
      --log $OUT/engine-ref.log --out $OUT/ref.json
fi
if [ $M1A_ON = 1 ]; then
  # 2b the same requests with M1a: identical tokens, every one 'request pipelined=1', the > 32K resume restored and streamed
  step m1a_exact 360 1200 drv --mode solo --seq m1a --config $CF/LN-M1A.json --label M1A --extra="$EXACT" "${XENV[@]}" \
      --log $OUT/engine-m1a_exact.log --out $OUT/m1a_exact.json
  judge cmp_m1a_exact ref,m1a_exact "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/m1a_exact.json --seq m1a --what "M1a exact" \
      --need-pipelined --need-resume
  # 2c forced rollbacks: every 3rd speculative window gets a wrong first token (GDN snapshot restores on CUDA1/2)
  step m1a_fm 300 900 drv --mode solo --seq fm --config $CF/LN-M1A.json --label M1A-FM --extra="$EXACT" "${XENV[@]}" \
      --env STRATA_PIPELINE_DEBUG=1 --env STRATA_PIPELINE_FORCE_MISS=3 --log $OUT/engine-m1a_fm.log --out $OUT/m1a_fm.json
  judge cmp_m1a_fm ref,m1a_fm,m1a_exact "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/m1a_fm.json --seq fm --what "M1a FORCE_MISS=3" \
      --need-pipelined --need-rollback --base-got $OUT/m1a_exact.json
  # 2d 1 user over HTTP, production settings: env-off vs M1a, ABBA x2, short then 1.2K prompts, 4 replies of 512 each
  step m1a_http 720 2400 ab env LEVELS=1 REPS=2 ORDER=abba ROUNDS=4 ./ab_quick.sh lnm1a $CF/LN-OFF.json $CF/LN-M1A.json
  judge m1a_http_ident m1a_http "$PY" -I "$DRV" --mode abcheck --file $OUT/m1a_http.txt --what "M1a 1-user replies" --expect identical
  judge m1a_http_speed m1a_http "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag lnm1a --arms LN-OFF,LN-M1A --reps 2 --gain-users 1 --gain-min 5
  judge m1a_http_log m1a_http "$PY" -I "$DRV" --mode pllog --what "M1a HTTP engines" "$L/results/engine-ab-lnm1a-LN-M1A-r*-long.run.log"
  grep -hE '^ *(short|long) +[0-9]+ users' "$OUT/m1a_http.txt" | sed 's/^/    /' | tee -a "$OUT/window.log" >/dev/null
fi

# ---- 3. M1b ----------------------------------------------------------------------------------------------------------
LANES_X=(--env STRATA_SPEC_LANES=2 --env STRATA_SPEC_LANES_LOG=1 --env STRATA_SPEC_LANES_HOLD=1)
NO_OV=(--drop-env STRATA_PIPE_OVERLAP --drop-env STRATA_PIPE_OVERLAP_KVS)
if [ $M1B_ON = 1 ]; then
  # 3a HOLD drive, overlap off: each slot == its serial solo GEN over the lane phase (until the other slot ends: the
  #    default gate then sends the rest through batch rows; the full length is reported beside it)
  step hold 480 1500 drv --mode lanes --pairs all --pics "$PICS" --config $CF/LN-OFF.json --label HOLD --extra="$EXACT" \
      "${XENV[@]}" "${LANES_X[@]}" "${NO_OV[@]}" --log $OUT/engine-hold.log --out $OUT/hold.json
  judge cmp_hold ref,hold "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold.json --pairs all --what "M1b HOLD" --hold
  # 3b STRATA_SPEC_LANES_ONE=1: a lone lane slot keeps its lane, so each reply is lane windows from admission to end
  step hold_one 360 1200 drv --mode lanes --pairs one --config $CF/LN-OFF.json --label HOLD-ONE --extra="$EXACT" \
      "${XENV[@]}" "${LANES_X[@]}" --env STRATA_SPEC_LANES_ONE=1 "${NO_OV[@]}" --log $OUT/engine-hold_one.log --out $OUT/hold_one.json
  judge cmp_hold_one ref,hold_one "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold_one.json --pairs one --what "M1b ONE" --full
  # 3c 1/2/4 users over HTTP, production settings: env-off vs lanes, ABBA x2; 1 user must be identical (solo path)
  step lanes_http 900 2700 ab env LEVELS=1,2,4 REPS=2 ORDER=abba ROUNDS=2 ./ab_quick.sh lnm1b $CF/LN-OFF.json $CF/LN-M1B.json
  judge lanes_http_ident lanes_http "$PY" -I "$DRV" --mode abcheck --file $OUT/lanes_http.txt --what "M1b replies (identical at 1 user: solo path; INFO beyond: lanes are another numerical path)" --expect identical --strict-users 1
  judge lanes_http_speed lanes_http "$PY" -I "$DRV" --mode sweep --since-line "$SWEEP0" --tag lnm1b --arms LN-OFF,LN-M1B --reps 2 --target-users 2 --target-dec 100 --flat-users 1,4
  judge lanes_http_log lanes_http "$PY" -I "$DRV" --mode lanelog --what "M1b HTTP engines" "$L/results/engine-ab-lnm1b-LN-M1B-r*-long.run.log"
  grep -hE '^ *(short|long) +[0-9]+ users' "$OUT/lanes_http.txt" | sed 's/^/    /' | tee -a "$OUT/window.log" >/dev/null
  grep -h 'lanes ran' $L/results/engine-ab-lnm1b-LN-M1B-r1-long.run.log 2>/dev/null | tail -3 | cut -c1-260 | sed 's/^/    /' | tee -a "$OUT/window.log" >/dev/null
  # 3d 16 users, NOSTART (+GIVEBACK=0): the lanes' side effects without lanes must leave the batch path byte-identical.
  #    batch_test16 (greedy, 16 slots, a cold 40K read beside 15 decoding, overlap on as in production): production's
  #    binary, env-off, NOSTART; then HTTP 16 users env-off vs NOSTART
  for arm in PROD OFF NOSTART; do
      cfg=$(tiercfg $CF/LN-$arm.json bt_$arm); exe=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" "$cfg")
      step bt_$arm 240 900 scoped bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-bt_$arm.log env $(cfg_env $cfg) python3 tools/batch_test16.py --exe $exe --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --dump $OUT/bt_$arm.json"
  done
  python3 - "$OUT" <<'PYEOF' | tee -a "$OUT/window.log"
import json, os, sys
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
d = {arm: dump(arm) for arm in ("PROD", "OFF", "NOSTART")}
for arm in ("OFF", "NOSTART"):
    if d["PROD"] is not None and d[arm] is not None:
        cmp("production's binary", d["PROD"], arm, d[arm])
if d["OFF"] is not None and d["NOSTART"] is not None:    # the lanes' side effects alone, same binary
    cmp("OFF (same binary)", d["OFF"], "NOSTART", d["NOSTART"])
# NOSTART must really have been NOSTART: exactly one boot line saying so, and no 'STRATA_SPEC_LANES=2 is off'
if ran("bt_NOSTART"):
    try: t = open(f"{o}/engine-bt_NOSTART.log", errors="replace").read().splitlines()
    except OSError: t = None
    if t is None:
        print("    [bt] ==> NOSTART engine log missing FAIL")
    else:
        nb = sum("NOSTART - the lanes never start" in l for l in t)
        off = sum("STRATA_SPEC_LANES=2 is off" in l for l in t)
        print(f"    [bt] ==> NOSTART boot: {nb} 'NOSTART - the lanes never start' lines (want 1), "
              f"{off} 'STRATA_SPEC_LANES=2 is off' lines (want 0) {'PASS' if nb == 1 and off == 0 else 'FAIL'}")
PYEOF
  for arm in OFF NOSTART; do f=$OUT/engine-bt_$arm.log
      log "    [bt] $arm: $(grep -c 'overlapped read of' $f 2>/dev/null) overlapped reads, $(grep -c 'lanes start' $f 2>/dev/null) lane starts, $(grep -c 'NOSTART - the lanes never start' $f 2>/dev/null) NOSTART boot lines, admissions $(grep -aoE 'admissions [0-9.]+s' $OUT/bt_$arm.txt 2>/dev/null | head -1)"; done
  step nostart_http 420 1500 ab env LEVELS=16 REPS=1 ROUNDS=2 ./ab_quick.sh lnnost $CF/LN-OFF.json $CF/LN-NOSTART.json
  judge nostart_ident nostart_http "$PY" -I "$DRV" --mode abcheck --file $OUT/nostart_http.txt --what "NOSTART 16-user replies" --expect identical
  judge nostart_log nostart_http "$PY" -I "$DRV" --mode lanelog --what "NOSTART engines (bt + HTTP)" --expect none \
      "$OUT/engine-bt_NOSTART.log" "$L/results/engine-ab-lnnost-LN-NOSTART-long.run.log"
  grep -hE '^ *(short|long) +[0-9]+ users' "$OUT/nostart_http.txt" | sed 's/^/    (one run each: |delta| < 1.3 % is not resolved) /' | tee -a "$OUT/window.log" >/dev/null
  # 3e HOLD with the overlapped read on (production's env): the held slot's admission read stays serial (ee403737)
  step hold_ov 300 900 drv --mode lanes --pairs ov --config $CF/LN-OFF.json --label HOLD-OV --extra="$EXACT" \
      "${XENV[@]}" "${LANES_X[@]}" --log $OUT/engine-hold_ov.log --out $OUT/hold_ov.json
  judge cmp_hold_ov ref,hold_ov "$PY" -I "$DRV" --mode compare --ref $OUT/ref.json --got $OUT/hold_ov.json --pairs ov --what "M1b HOLD + overlap" --hold --ov
fi

# ---- 4. s28r + the overlapped read vs production's config, 16 users, ABBA x2 -----------------------------------------
step s28ov_http 780 2400 ab env LEVELS=16 REPS=2 ORDER=abba ROUNDS=2 ./ab_quick.sh lns28ov $CF/LN-PROD.json $CF/LN-S28ROV.json
judge s28ov_ident s28ov_http "$PY" -I "$DRV" --mode abcheck --file $OUT/s28ov_http.txt --what "s28r+overlap 16-user replies" --expect identical
grep -hE '^ *(short|long) +[0-9]+ users' "$OUT/s28ov_http.txt" | sed 's/^/    /' | tee -a "$OUT/window.log" >/dev/null
step_ran s28ov_http || log "    [s28ov] SKIP (step s28ov_http skipped)"
for r in 1 2; do f=$L/results/engine-ab-lns28ov-LN-S28ROV-r$r-long.run.log; [ -f "$f" ] || continue
    nl=$(grep -c 'no loan' $f); nb=$(grep -c 'borrows' $f); low=$(grep 'VRAM free with everything loaded' $f | grep -c 'LOW\|WARNING')
    v=PASS; [ "$nl" -eq 4 ] && [ "$nb" -eq 0 ] && [ "$low" -eq 0 ] || v=FAIL
    log "    [s28ov] r$r: no loan $nl/4, borrows $nb, LOW $low; VRAM free $(grep -o '[0-9]* MiB of VRAM free with everything loaded' $f | cut -d' ' -f1 | tr '\n' '/') MiB; $(grep -c 'overlapped read of' $f) overlapped reads, refused: $(grep -o 'overlapped read off: .*' $f | sort | uniq -c | tr '\n' ';' | cut -c1-200) $v"
done

# ---- 5. prefill speed by prompt length on production's config (binary + env) -----------------------------------------
step prefill 480 1500 drv --mode prefill --config $CF/LN-PROD.json --label PREFILL --sizes 170,512,1200,4000,16000,64000 \
    --reps 2 --phases idle,dec15 --log $OUT/engine-prefill.log --out $OUT/prefill.json

# ---- 6. production back, unchanged ------------------------------------------------------------------------------------
trap - INT TERM HUP
bench_down end
start_prod || log "PRODUCTION DID NOT COME BACK - check strata.service now"
P=$(systemctl --user show -p MainPID --value strata.service)
log "production engine: $(for c in $(pgrep -P "$P" 2>/dev/null); do readlink /proc/$c/exe; done | tr '\n' ' ')"
rm -rf "$TIER"
log "window done after $(( ($(date +%s) - T0) / 60 )) min"
grep -hE 'PASS|FAIL|SKIP' "$OUT/window.log" | grep -v 'grep' | tail -60
