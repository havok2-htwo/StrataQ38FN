#!/usr/bin/env bash
# M0 (spec lanes, DESIGN-spec-lanes.md): does the pipelined solo-spec path (--pipeline-windows 2, pl_* API) work on the
# 3090 split and keep the tokens?  1 user, ~170 / ~1.2K prompts, greedy, serial solo vs pipelined, ABBA x2; replies
# compared by ab_quick.  Production back unchanged after.  Results: results/m0.
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/m0; mkdir -p "$OUT"; URL=http://127.0.0.1:8090
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
free_gpus() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production"; systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do [ $(( $(date +%s) - t0 )) -gt 900 ] && break; sleep 5; done
    log "production up after $(( $(date +%s) - t0 )) s"; }
trap 'log aborted; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus; start_prod; exit 1' INT TERM
log "m0: stopping strata.service"; systemctl --user stop strata.service; free_gpus
( cd $L && LEVELS=1 REPS=2 ORDER=abba timeout -s INT 3000 ./ab_quick.sh m0 $C/M0-serial.json $C/M0-pl.json ) > $OUT/ab.txt 2>&1
log "m0 ab exit $?"; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus
for f in $L/results/engine-ab-m0-M0-pl-r1-short.run.log; do
  log "pl engine: $(grep -aE 'pipeline-windows|pipelin' $f | head -3 | tr '\n' ' ' | cut -c1-300)"; done
grep -E "^ *(short|long) +1 users" $OUT/ab.txt | tee -a $OUT/window.log
trap - INT TERM
start_prod
