#!/usr/bin/env bash
# GPU window for E1/E2 (strata-g3-143v3, STRATA_SLOT_CHAIN_REUSE=1 + STRATA_GIVEBACK_CHECKS=1), then one restart:
#   - stop production, run tools/cache_reuse_check.py on the 4 GPUs (results in linux/results/e12/)
#   - server code -> g3-143 72764321 (E1/E2, prefix diag, upstream #2026/#2037/#1994/#656 server fixes)
#   - server env via a systemd drop-in: STRATA_ADMIT_SHORT_FIRST=8192, STRATA_PREFIX_DIAG=1 (the config's "env" only
#     reaches the engine)
#   - production.json -> v3 config when the check says GO, else back to v2 (the binary production ran so far)
# Rollback: linux/deploy-vrope-v2.sh (v2) or --rollback (k5); rm the drop-in and daemon-reload to drop the server env.
set -u
R=/media/raid0/StrataQ38FN; OUT=$R/linux/results/e12; mkdir -p "$OUT"
URL=http://127.0.0.1:8090; LINK=$R/linux/configs/production.json
V2=T16-143-prod262r-vision-cpu-v2.json; V3=T16-143-prod262r-vision-cpu-v3.json
DROP=~/.config/systemd/user/strata.service.d; CODE=7a49445b
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
free_gpus() { for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do case "$(readlink -f /proc/$p/exe 2>/dev/null)" in $R/Strata*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
start_prod() {
    ln -sfn "$1" "$LINK"; log "production.json -> $1, server $(git -C $R/Strata-143-prod log --oneline -1 | cut -c1-60)"
    systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
        systemctl --user is-active -q strata.service || { log "strata.service died during the start"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "start timeout"; return 1; }; sleep 5; done
    log "loaded after $(( $(date +%s) - t0 )) s"; }
restore() { log "$1 - back on v2"; free_gpus; start_prod "$V2"; }
trap 'restore "window aborted"; exit 1' INT TERM

log "window start: stopping strata.service"
systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || { restore "GPUs not free"; exit 1; }
cd $R/Strata-143-vrope
timeout -s INT -k 120 2700 python3 tools/cache_reuse_check.py --exe /media/raid0/StrataQ38FN/Strata-143-vrope/build-vrope/strata \
    --config $R/linux/configs/T16-143-prod262r-vision-cpu-v2.json --old-exe $R/Strata/engine/strata-g3-143v2 \
    --no-draft-pair --out $OUT/cache_reuse.json > $OUT/cache_reuse.txt 2>&1
RC=$?; log "cache_reuse_check exit $RC: $(grep -aE '==>' $OUT/cache_reuse.txt | tail -1)"
trap - INT TERM
free_gpus || { restore "GPUs not free after the check"; exit 1; }

git -C $R/Strata-143-prod checkout -q --detach $CODE
mkdir -p $DROP
printf '[Service]\nEnvironment=STRATA_ADMIT_SHORT_FIRST=8192\nEnvironment=STRATA_PREFIX_DIAG=1\n' > $DROP/server-env.conf
systemctl --user daemon-reload
if [ "$RC" = 0 ]; then
    start_prod "$V3" || { log "v3 did not start"; systemctl --user stop strata.service; free_gpus; start_prod "$V2"; }
else
    log "check not GO (exit $RC) - production on v2 with the new server"; start_prod "$V2"
fi
P=$(systemctl --user show -p MainPID --value strata.service)
log "server env: $(tr '\0' '\n' < /proc/$P/environ | grep -E 'STRATA_ADMIT_SHORT_FIRST|STRATA_PREFIX_DIAG' | tr '\n' ' ')"
log "window done"
