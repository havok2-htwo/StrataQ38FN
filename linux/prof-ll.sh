#!/usr/bin/env bash
# profile window (approved 2026-10-16): pure batch decode, 16 slots, 30 s, production config, old (1441a) vs new (1441b),
# plain (rows/s) and STRATA_VERIFY_PROFILE (phase table); production back unchanged afterwards.  Results: results/ll/prof.
set -u
R=/media/raid0/StrataQ38FN; D=$R/linux/results/ll/prof; P=$R/linux/results/profile2; URL=http://127.0.0.1:8090
log() { echo "[$(date +%T)] $*" | tee -a "$D/prof.log"; }
gpu_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production (unchanged)"; systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do [ $(( $(date +%s) - t0 )) -gt 900 ] && break; sleep 5; done
    log "production up after $(( $(date +%s) - t0 )) s"; }
trap 'log aborted; start_prod; exit 1' INT TERM
log "stopping strata.service"; systemctl --user stop strata.service; gpu_free || { log "GPUs busy"; start_prod; exit 1; }
for arm in old new oldp newp; do
  who=${arm:0:3}; a="--mode decode --batch 16 --seconds 30"
  [ "${arm: -1}" = p ] && a="$a --env STRATA_VERIFY_PROFILE=1"
  gpu_free || log "GPUs busy before $arm"
  log "$arm start"
  timeout -s INT 900 python3 $P/prof_driver.py --config $D/time_$who.json --out $D/t_$arm $a > $D/t_$arm.driver.out 2>&1
  log "$arm rc=$? $(python3 $P/dec_rows.py $D/t_$arm/events.log 2>&1 | tail -1)"
done
trap - INT TERM
gpu_free; start_prod
