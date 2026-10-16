#!/usr/bin/env bash
# M1a + M1b together with --kv-resident 28672 (frees ~150-200 MiB/card for M1a's room check on CUDA0); production config
# with overlap as reference.  Test servers on 127.0.0.1:8091, own tier dir; production back unchanged.  Results: results/both.
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/both; mkdir -p "$OUT"; URL=http://127.0.0.1:8090
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
free_gpus() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production (production.json -> $(readlink $C/production.json))"; systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do [ $(( $(date +%s) - t0 )) -gt 900 ] && break; sleep 5; done
    log "production up after $(( $(date +%s) - t0 )) s"; }
STEP=
abort() { trap - INT TERM HUP; log "aborted"; [ -n "$STEP" ] && kill -INT -- -"$STEP" 2>/dev/null; sleep 5; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus; start_prod; exit 1; }
run() { timeout -s INT -k 60 "$@" & STEP=$!; wait $STEP; local rc=$?; STEP=; return $rc; }
ab() { ( cd $L && export SERVER_DIR=$R/Strata-143-prod PORT=8091 BIND=127.0.0.1; "$@" ); }
trap 'echo "aborted before production was stopped"; exit 1' INT TERM HUP
log "stopping strata.service"; trap abort INT TERM HUP; systemctl --user stop strata.service; free_gpus
# 1. low concurrency 1/2/4 users, ABBA x1 (prod, both, both, prod)
for arm in BOTH-prod BOTH-28k BOTH-28k BOTH-prod; do
  n=$(( ${n:-0} + 1 )); log "lowconc $arm (#$n): start"
  run 1500 bash -c "cd $L && PORT=8091 BIND=127.0.0.1 SERVER_DIR=$R/Strata-143-prod ./run_strata_variant.sh both-lc$n-$arm $C/$arm.json 1,2,4 --rounds 2 --max-tok 512 --temperature 0" > $OUT/lc$n-$arm.txt 2>&1
  log "lowconc $arm (#$n): exit $?"; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus
  f=$L/results/engine-both-lc$n-$arm.run.log
  [ -f "$f" ] && grep -aE "prompt buffers|borrows|VRAM free with everything|SOLO_PIPELINE|spec_lanes|too little room|keeps the solo path serial|experts resident" "$f" | head -14 | cut -c1-200 | sed 's/^/    /' | tee -a $OUT/window.log >/dev/null
  [ -f "$f" ] && log "  lanes start $(grep -c 'lanes start' $f), pipelined=1 $(grep -c 'pipelined=1' $f), pipelined=0 $(grep -c 'pipelined=0' $f)"
done
# 2. 16 users, reply identity + speed, one rep each
log "ab16: start"
run 2400 bash -c "cd $L && PORT=8091 BIND=127.0.0.1 SERVER_DIR=$R/Strata-143-prod REPS=1 ./ab_quick.sh both16 $C/BOTH-prod.json $C/BOTH-28k.json" > $OUT/ab16.txt 2>&1
log "ab16: exit $?"; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus
grep -E "^ *(short|long) +16 users" $OUT/ab16.txt | tee -a $OUT/window.log
trap - INT TERM HUP
start_prod; log "window done"
