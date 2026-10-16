#!/usr/bin/env bash
# 1/2/4/8 users (~1.2K prompts, 512 tokens, greedy): production config vs s28r; production back unchanged after.
set -u
R=/media/raid0/StrataQ38FN; L=$R/linux; C=$L/configs; OUT=$L/results/gap; URL=http://127.0.0.1:8090
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
free_gpus() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production"; systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do [ $(( $(date +%s) - t0 )) -gt 900 ] && break; sleep 5; done
    log "production up after $(( $(date +%s) - t0 )) s"; }
trap 'log aborted; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus; start_prod; exit 1' INT TERM
log "lowconc: stopping strata.service"; systemctl --user stop strata.service; free_gpus
for lc in LC-prod LC-s28r; do
    log "step lowconc $lc: start"
    ( cd $L && timeout -s INT 1500 ./run_strata_variant.sh "lowconc-$lc" $C/$lc.json "1,2,4,8" --rounds 2 --max-tok 512 \
        --temperature 0 ) > "$OUT/lowconc-$lc.txt" 2>&1
    log "step lowconc $lc: exit $?"; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus
done
trap - INT TERM
start_prod
