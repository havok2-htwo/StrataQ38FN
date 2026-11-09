#!/usr/bin/env bash
# GPU window: validate g3-1441 (v0.1.44.1 + our series, engine strata-g3-1441a) against production v3, then bring
# production back EXACTLY as before (production.json and Strata-143-prod untouched). Results: linux/results/r1441/.
set -u
R=/media/raid0/StrataQ38FN; OUT=$R/linux/results/r1441; mkdir -p "$OUT"; URL=http://127.0.0.1:8090
NEW=$R/Strata/engine/strata-g3-1441a; OLD=$R/Strata/engine/strata-g3-143v3; CFG=$R/linux/configs/T16-143-prod262r-vision-cpu-v3.json
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
free_gpus() { for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do case "$(readlink -f /proc/$p/exe 2>/dev/null)" in $R/Strata*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production as before (production.json -> $(readlink $R/linux/configs/production.json))"
    systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
        systemctl --user is-active -q strata.service || { log "strata.service died during the start"; return 1; }
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "start timeout"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s"; }
trap 'log "window aborted"; free_gpus; start_prod; exit 1' INT TERM
step() { local name=$1 limit=$2; shift 2; log "step $name: start"
    timeout -s INT -k 120 $limit "$@" > "$OUT/$name.txt" 2>&1; local rc=$?; echo $rc > "$OUT/rc_$name"
    log "step $name: exit $rc $(grep -aE 'VERDICT|==>' "$OUT/$name.txt" | tail -1 | cut -c1-120)"; free_gpus || log "GPUs busy after $name"; }

log "window start: stopping strata.service"; systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || { log "GPUs not free"; start_prod; exit 1; }
cd $R/Strata-1441
step text_ab 1500 python3 tools/vision_batch_check.py --exe $NEW --config $CFG --batch 16 --text-ab $OLD --logs $OUT/text-ab --out $OUT/text_ab.json
step vision 1800 python3 tools/vision_batch_check.py --exe $NEW --config $CFG --batch 16 --expect-from $OUT/text_ab.json --logs $OUT/vision --out $OUT/vision.json
step msg_ckpt 2400 python3 tools/msg_ckpt_check.py --exe $NEW --config $CFG --old-exe $OLD --no-draft-pair --out $OUT/msg_ckpt.json
step cache_reuse 2400 python3 tools/cache_reuse_check.py --exe $NEW --config $CFG --old-exe $OLD --no-draft-pair --out $OUT/cache_reuse.json
step gpu_ctest 1500 bash -c "cd build-1441 && ctest --output-on-failure -j1"
trap - INT TERM
free_gpus; start_prod; log "window done"
