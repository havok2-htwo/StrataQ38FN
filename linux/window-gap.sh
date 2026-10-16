#!/usr/bin/env bash
# GPU window (approved 2026-10-16; ~55-70 min with production stopped): where the admission stall goes on
# strata-g3-1441b, and whether STRATA_PIPE_OVERLAP pays on today's engine.  Measures only, no new code.
# Production comes back exactly as before (systemctl --user start strata.service; production.json untouched), also on
# abort (trap) and on every failed step.  Bench engines log to results/engine-gap.log and spill into their own tier dir
# (/media/raid0/strata-kvspill-bench), never production's.  Results: results/gap/.
#   install: cp <scratchpad>/gap/window-gap.sh <scratchpad>/gap/join_pause.py /media/raid0/StrataQ38FN/linux/
#   run:     cd /media/raid0/StrataQ38FN/linux && G=<scratchpad>/gap ./window-gap.sh
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/gap; mkdir -p "$OUT"
URL=http://127.0.0.1:8090; BURL=http://127.0.0.1:8090
PY=$R/Strata/.venv/bin/python; ENGLOG=$L/results/engine-gap.log
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
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
# a bench server with <cfg> on 8090 (run_strata_variant's start, a 4-user warm-up that also makes admissions beside
# decoding slots, so an overlap gate refusal is logged), left running (KEEP=1); 1 = it did not come up
serve() { local label=$1 cfg=$2
    ( cd $L && KEEP=1 timeout -s INT 1500 ./run_strata_variant.sh "$label" "$cfg" 4 --rounds 1 --max-tok 64 \
        --prompt-tok 100 --temperature 0 ) > "$OUT/$label.serve.txt" 2>&1
    curl -s -m 3 $BURL/health | grep -q '"loaded": *true'; }
ov_check() {   # did the overlap arm overlap? (the refusal line is logged without STRATA_TRACE: generate.cpp:14115)
    local f=$L/results/engine-$1.run.log
    if grep -q 'overlapped read off:' "$f"; then log "$1: OVERLAP REFUSED: $(grep -m1 'overlapped read off:' "$f")"; return 1; fi
    log "$1: overlapped reads: $(grep -c 'strata batch: overlapped read of' "$f")"; return 0; }

mkdir -p /media/raid0/strata-kvspill-bench

log "window start: stopping strata.service"; systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || abort "GPUs not free"

# 1. admission phases on production's config (+ STRATA_TRACE): lockstep 16 users short + 1.2K (~6 min)
log "step trace-ab: start"
( cd $L && timeout -s INT 1200 ./ab_quick.sh gaptr $C/GAP-trace.json ) > "$OUT/trace-ab.txt" 2>&1
log "step trace-ab: exit $?"; bench_down trace-ab
# 2. same config, agent-like load (~12 min): multi-turn 8 x 4 (restores, give-backs, slot copies) and bench_agentic
if serve gaptr-agent $C/GAP-trace.json; then
    log "step trace-agent: multiturn"
    ( cd $L && timeout -s INT 600 $PY -I bench_multiturn.py --url $BURL --label gaptr-mt --users 8 --turns 4 --temperature 0 \
        --out $OUT/multiturn.jsonl ) > "$OUT/trace-mt.txt" 2>&1; log "multiturn exit $?"
    log "step trace-agent: agentic"
    ( cd $L && timeout -s INT 900 $PY -I bench_agentic.py --url $BURL --label gaptr-agentic --out $OUT/agentic.jsonl ) \
        > "$OUT/trace-agentic.txt" 2>&1; log "agentic exit $?"
    n=$(grep -n 'engine started' $ENGLOG | tail -1 | cut -d: -f1); tail -n +"${n:-1}" $ENGLOG > "$OUT/engine-trace-agent.log"
else log "trace-agent: server did not come up (see $OUT/gaptr-agent.serve.txt)"; fi
bench_down trace-agent

# 2b. low-concurrency baseline for batch-MTP: 1/2/4/8 users, ~1.2K prompts, 512 tokens, greedy; production and s28r
for lc in LC-prod LC-s28r; do
    log "step lowconc $lc: start"
    ( cd $L && timeout -s INT 1500 ./run_strata_variant.sh "lowconc-$lc" $C/$lc.json "1,2,4,8" --rounds 2 --max-tok 512 \
        --temperature 0 ) > "$OUT/lowconc-$lc.txt" 2>&1
    log "step lowconc $lc: exit $?"; bench_down "lowconc-$lc"
done

# 3. fit + gate probe of the overlap arm at 64K; on failure the 32K pair (production's resident K/V is 32K too)
P=64k; JT=32000
if ! serve gapprobe-64k $C/GAP-64k-ov.json; then
    log "64K overlap arm did not come up (VRAM?): falling back to 32K"; bench_down probe64; P=32k; JT=24000
    serve gapprobe-32k $C/GAP-32k-ov.json || { bench_down probe32; abort "32K overlap arm did not come up either"; }
fi
ov_check gapprobe-$P; OVOK=$?
bench_down probe
if [ $OVOK -ne 0 ]; then log "overlap refused on this engine/config: steps 4-5 skipped (no 'overlap neutral' result)"
else
    # 4. overlap on vs off, lockstep and mixed lengths, ABBA x2 (~20 min); replies must match (compare_replies)
    log "step ov-ab ($P): start"
    ( cd $L && REPS=2 ORDER=abba timeout -s INT 2400 ./ab_quick.sh gapov $C/GAP-$P.json $C/GAP-$P-ov.json ) > "$OUT/ov-ab.txt" 2>&1
    log "step ov-ab: exit $?"; bench_down ov-ab
    for f in $L/results/engine-ab-gapov-GAP-$P-ov-r*-short.run.log; do [ -f "$f" ] && ov_check "$(basename "$f" .run.log | sed 's/^engine-//')"; done
    log "step ov-mix ($P): start"
    ( cd $L && MIX=128 ROUNDS=4 timeout -s INT 1500 ./ab_quick.sh gapovmix $C/GAP-$P.json $C/GAP-$P-ov.json ) > "$OUT/ov-mix.txt" 2>&1
    log "step ov-mix: exit $?"; bench_down ov-mix
    # 5. cold long join beside 4 decoding streams: longest pause per stream and the join's TTFT, off vs on (~8 min)
    s=11
    for arm in GAP-$P GAP-$P-ov; do
        if serve gapjoin-$arm $C/$arm.json; then
            ( cd $L && timeout -s INT 900 $PY -I join_pause.py --url $BURL --label gapjoin-$arm --bg 4 --join-tok $JT \
                --reps 3 --seed $s --out $OUT/join.jsonl ) > "$OUT/join-$arm.txt" 2>&1
            log "join $arm: $(tail -1 "$OUT/join-$arm.txt")"
            n=$(grep -n 'engine started' $ENGLOG | tail -1 | cut -d: -f1); tail -n +"${n:-1}" $ENGLOG > "$OUT/engine-join-$arm.log"
        else log "join $arm: server did not come up"; fi
        bench_down "join-$arm"; s=$((s + 1))
    done
fi

trap - INT TERM
start_prod || log "PRODUCTION DID NOT COME BACK - check strata.service now"
log "window done"
grep -h 'users' "$OUT/trace-ab.txt" "$OUT/ov-ab.txt" "$OUT/ov-mix.txt" 2>/dev/null | tee -a "$OUT/window.log"
