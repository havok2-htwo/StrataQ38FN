#!/usr/bin/env bash
# GPU window for STRATA_PIPE_OVERLAP_KVS (branch ll3-ovkv, Strata-ll3-ovkv): the overlapped prompt read with a streamed
# K/V cache (--kv-resident) on production's config.  ~2-2.5 h with production stopped.  Production comes back
# exactly as before (production.json untouched), also on abort.  EVERY bench engine (production's binary in arm OLD
# included) spills into /media/raid0/strata-kvspill-bench and logs to results/, never production's tier dir.
# Results: results/ovkv; verdicts: grep -E 'PASS|FAIL' results/ovkv/window.log
#   install: cp <scratchpad>/window-ovkv.sh <scratchpad>/ovkv_ident.py /media/raid0/StrataQ38FN/linux/
#   run:     cd /media/raid0/StrataQ38FN/linux && ./window-ovkv.sh
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/ovkv; mkdir -p "$OUT"
URL=http://127.0.0.1:8090; PY=$R/Strata/.venv/bin/python; ENGLOG=$L/results/engine-ovkv.log
B=$R/Strata-ll3-ovkv/build; NEW=$R/Strata/engine/strata-g3-1441b-ovkv; OLD=$R/Strata/engine/strata-g3-1441b
PROD=$C/T16-1441b-prod262r-vision-cpu.json
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
vram() { nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tr '\n' ';'; }
free_gpus() { for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do case "$(readlink -f /proc/$p/exe 2>/dev/null)" in $R/Strata*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production as before (production.json -> $(readlink $C/production.json))"
    systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
        systemctl --user is-active -q strata.service || { log "strata.service died during the start"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "start timeout"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s"; }
bench_down() { systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus || log "GPUs busy after $1"; }
abort() { log "window aborted: $*"; bench_down abort; start_prod; exit 1; }
trap 'abort signal' INT TERM
step() { local name=$1 limit=$2; shift 2; log "step $name: start"
    timeout -s INT -k 120 $limit "$@" > "$OUT/$name.txt" 2>&1; local rc=$?; echo $rc > "$OUT/rc_$name"
    log "step $name: exit $rc $(grep -aE '==>|tests passed|PASS|FAIL' "$OUT/$name.txt" | tail -1 | cut -c1-140)"
    free_gpus || log "GPUs busy after $name"; }
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }
serve() { local label=$1 cfg=$2
    ( cd $L && KEEP=1 timeout -s INT 1500 ./run_strata_variant.sh "$label" "$cfg" 4 --rounds 1 --max-tok 64 \
        --prompt-tok 100 --temperature 0 ) > "$OUT/$label.serve.txt" 2>&1
    curl -s -m 3 $URL/health | grep -q '"loaded": *true'; }
engine_tail() { local n; n=$(grep -n 'engine started' $ENGLOG | tail -1 | cut -d: -f1); tail -n +"${n:-1}" $ENGLOG > "$1"; }

# --- before the window (no GPU): the engine and the arms, all from production's config with the bench tier dir and
# log.  OVKV-OLD = production's binary; on the new binary: OVKV-A = default path, OVKV-B = + STRATA_PIPE_OVERLAP=1
# (must refuse: "overlapped read off: K/V streaming"), OVKV-C = + KVS=1; OVKV-AV / OVKV-CV = A / C +
# STRATA_SLOT_PARK_VERIFY=1; OVKV-CP = C + FORCE_PADUP=1; OVKV-CW = C + FORCE_WAIT=2; OVKV-CF<f> = C + TEST_FAULT=<f>.
[ -x "$NEW" ] || cp "$B/strata" "$NEW"
[ -f "$L/ovkv_ident.py" ] || { echo "ovkv_ident.py missing in $L (see install)"; exit 1; }
FAULTS="poll throw cuda"
for arm in OLD A B C AV CV CP CW $(for f in $FAULTS; do echo CF$f; done); do
python3 - "$PROD" "$C/OVKV-$arm.json" "$NEW" "$OLD" "$arm" <<'PYEOF' || exit 1
import json, sys
c = json.load(open(sys.argv[1])); arm = sys.argv[5]; e = c["env"]
c["exe"] = sys.argv[4] if arm == "OLD" else sys.argv[3]
e["STRATA_CONV_TIER_DIR"] = "/media/raid0/strata-kvspill-bench"
c["log"] = "/media/raid0/StrataQ38FN/linux/results/engine-ovkv.log"
if arm not in ("OLD", "A", "AV"): e["STRATA_PIPE_OVERLAP"] = "1"
if arm.startswith("C"): e["STRATA_PIPE_OVERLAP_KVS"] = "1"
if arm.endswith("V"): e["STRATA_SLOT_PARK_VERIFY"] = "1"
if arm == "CP": e["STRATA_PIPE_OVERLAP_FORCE_PADUP"] = "1"
if arm == "CW": e["STRATA_PIPE_OVERLAP_FORCE_WAIT"] = "2"
if arm.startswith("CF"): e["STRATA_PIPE_OVERLAP_TEST_FAULT"] = arm[2:]
json.dump(c, open(sys.argv[2], "w"), indent=1)
PYEOF
done
mkdir -p /media/raid0/strata-kvspill-bench

log "window start: stopping strata.service"; systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || abort "GPUs not free"

# 1. ctests on one card: the new kernel concurrency test + the streaming/slot tests
step ctest 1500 bash -c "cd $B && CUDA_VISIBLE_DEVICES=0 ctest --output-on-failure -j1 -R 'kv_stream_overlap_parity|overlap_kvs_test|kv_stream_parity|qsa_slots_stream_parity|conversation_snapshot|conversation_transfer|slot_chain|draft_kv_plan'"

# 2. token identity, greedy, 16 slots, a cold 40K read beside 15 decoding slots: OLD (production's binary) vs A, B, C,
# and C with the debug knobs FORCE_PADUP=1 (CP) and FORCE_WAIT=2 (CW).  batch_test16.py writes the engine's stderr to
# $BATCH_TEST_LOG (one file per arm): the gate's lines are checked there, not in the step's stdout.
for arm in OLD A B C CP CW; do
    exe=$NEW; [ $arm = OLD ] && exe=$OLD; cfg=$C/OVKV-$arm.json
    log "ident_$arm: VRAM before (index, used, free): $(vram)"
    step ident_$arm 1800 bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-ident_$arm.log env $(cfg_env $cfg) python3 tools/batch_test16.py --exe $exe --config $cfg --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --dump $OUT/ident_$arm.json"
done
python3 - "$OUT" <<'PYEOF' | tee -a "$OUT/window.log"
import json, sys
o = sys.argv[1]; ref = json.load(open(f"{o}/ident_OLD.json"))["batch"]
for arm in ("A", "B", "C", "CP", "CW"):
    try: got = json.load(open(f"{o}/ident_{arm}.json"))["batch"]
    except Exception as e: print(f"ident {arm}: FAIL no dump ({e})"); continue
    bad = [i for i, (a, b) in enumerate(zip(ref, got)) if a != b] + list(range(min(len(ref), len(got)), max(len(ref), len(got))))
    print(f"ident {arm} vs OLD: {len(ref) - len(bad)}/{len(ref)} slots token-identical " + (f"FAIL differ: {bad}" if bad else "PASS"))
PYEOF
# the gate must have done what the arm is for, else identity proves nothing: B refuses for --kv-resident; C/CP/CW
# overlap (at least one 'overlapped read of', no 'overlapped read off:'), no OVERFLOW; the first-read line has VRAM
if grep -q 'overlapped read off: K/V streaming (--kv-resident)' "$OUT/engine-ident_B.log"; then log "ident B gate: refused as expected PASS"
else log "ident B gate: FAIL (no 'overlapped read off: K/V streaming (--kv-resident)' in engine-ident_B.log)"; fi
for arm in C CP CW; do f=$OUT/engine-ident_$arm.log
    nov=$(grep -c 'overlapped read of' "$f" 2>/dev/null); noff=$(grep -c 'overlapped read off:' "$f" 2>/dev/null)
    if [ "${nov:-0}" -gt 0 ] && [ "${noff:-0}" -eq 0 ] && ! grep -q OVERFLOW "$f"; then v=PASS; else v=FAIL; fi
    log "ident $arm gate: $v (${nov:-0} overlapped, ${noff:-0} refused $(grep -m1 'overlapped read off:' "$f" | cut -c1-140))"
    grep -h 'captured .* group layout\|STRATA_PIPE_OVERLAP_KVS=1: first overlapped read\|KV streaming:' "$f" | head -4 | cut -c1-260 | tee -a "$OUT/window.log"
done

# 2b. the fault paths (C + STRATA_PIPE_OVERLAP_TEST_FAULT, once per process during an overlapped read).  No token check
# (the faulted request fails, or the engine exits for cuda: #224); recorded: exit code and the fault's lines.
for f in $FAULTS; do
    step ident_CF$f 1200 bash -c "cd $R/Strata-1441 && BATCH_TEST_LOG=$OUT/engine-ident_CF$f.log env $(cfg_env $C/OVKV-CF$f.json) python3 tools/batch_test16.py --exe $NEW --config $C/OVKV-CF$f.json --batch 16 --n 16 --max-new 160 --skip-solo --long-tokens 40000 --dump $OUT/ident_CF$f.json"
    log "fault $f: rc $(cat $OUT/rc_ident_CF$f); $(grep -h 'test fault\|ERR\|watchdog' $OUT/engine-ident_CF$f.log | head -3 | cut -c1-160 | tr '\n' '|')"
done

# 2c. token identity over HTTP with the windows' side streaming too: 4 background conversations of ~45K tokens (more
# than --kv-resident 32768: their selections miss and CLOCK evicts while they decode; rounds 2-4 resume them from the
# conversation cache, reads with p0 > 0 staging a restored host copy), cold joins of 1K, 32K, 100K and 200K beside
# them, each join's second turn (a resume from its parked slot or the RAM cache), STRATA_SLOT_PARK_VERIFY=1.
# AV (default path) vs CV (overlap + KVS): every reply identical.
for arm in AV CV; do
    if serve ovkvh-$arm $C/OVKV-$arm.json; then
        log "hident $arm: VRAM after start (index, used, free): $(vram)"
        ( cd $L && timeout -s INT 3000 $PY -I ovkv_ident.py --url $URL --out $OUT/hident-$arm.json ) > "$OUT/hident-$arm.txt" 2>&1
        log "hident $arm: $(tail -1 "$OUT/hident-$arm.txt"); VRAM after (index, used, free): $(vram)"
        f=$OUT/engine-hident-$arm.log; engine_tail "$f"
        log "hident $arm: $(grep -c 'overlapped read of' $f) overlapped, $(grep -c 'overlapped read off:' $f) refused, $(grep -c OVERFLOW $f) overflow, park verify lines: $(grep -ci 'park.*verif' $f); $(grep -m1 'overlapped read off:' $f | cut -c1-140)"
        grep -h 'STRATA_PIPE_OVERLAP_KVS=1: first overlapped read\|KV streaming:\|differ\|mismatch' "$f" | tail -4 | cut -c1-260 | tee -a "$OUT/window.log"
    else log "hident $arm: server did not come up FAIL"; fi
    bench_down "hident-$arm"
done
( cd $L && $PY -I ovkv_ident.py --compare $OUT/hident-AV.json $OUT/hident-CV.json ) 2>&1 | tee -a "$OUT/window.log"

# 3. speed + reply identity over HTTP: A vs C, ABBA x2, lockstep then mixed lengths
step ab 2700 bash -c "cd $L && REPS=2 ORDER=abba ./ab_quick.sh ovkv $C/OVKV-A.json $C/OVKV-C.json"; bench_down ab
for f in $L/results/engine-ab-ovkv-OVKV-C-r*-short.run.log; do [ -f "$f" ] && log "$(basename $f): $(grep -c 'overlapped read of' $f) overlapped, $(grep -c 'overlapped read off:' $f) refused $(grep -m1 'overlapped read off:' $f | cut -c1-120)"; done
step abmix 1800 bash -c "cd $L && MIX=128 ROUNDS=4 ./ab_quick.sh ovkvmix $C/OVKV-A.json $C/OVKV-C.json"; bench_down abmix

# 4. cold join beside 4 decoding streams: longest pause per stream and the join's TTFT, A vs C, 32K, 100K, 200K
s=21
for arm in OVKV-A OVKV-C; do
    if serve ovkvjoin-$arm $C/$arm.json; then
        for jt in 32000 100000 200000; do
            ( cd $L && timeout -s INT 1500 $PY -I join_pause.py --url $URL --label ovkvjoin-$arm-$jt --bg 4 --join-tok $jt \
                --reps 3 --seed $s --out $OUT/join.jsonl ) > "$OUT/join-$arm-$jt.txt" 2>&1
            log "join $arm $jt: $(tail -1 "$OUT/join-$arm-$jt.txt")"
        done
        engine_tail "$OUT/engine-join-$arm.log"
        grep -h 'overlapped read of\|overlapped read off\|first overlapped read' "$OUT/engine-join-$arm.log" | tail -5 | cut -c1-220 | tee -a "$OUT/window.log"
    else log "join $arm: server did not come up"; fi
    bench_down "join-$arm"; s=$((s + 1))
done

trap - INT TERM
start_prod || log "PRODUCTION DID NOT COME BACK - check strata.service now"
log "window done"
grep -h 'users' "$OUT/ab.txt" "$OUT/abmix.txt" 2>/dev/null | tee -a "$OUT/window.log"
grep -hE 'PASS|FAIL' "$OUT/window.log" | tail -20
