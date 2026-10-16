#!/usr/bin/env bash
# which switch changes the image / solo-path tokens (window-ll text_ab NO-GO): one --text-ab per switch, then production back
set -u
R=/media/raid0/StrataQ38FN; OUT=$R/linux/results/ll/bisect; mkdir -p "$OUT"; C=$R/linux/configs; URL=http://127.0.0.1:8090
NEW=$R/Strata/engine/strata-g3-1441ll; OLD=$R/Strata/engine/strata-g3-1441a
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/bisect.log"; }
gpu_free() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production"; systemctl --user start strata.service; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "start timeout"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s"; }
trap 'log aborted; start_prod; exit 1' INT TERM
for arm in ${ARMS:-off vb qss grfast grpf grnorm mmvf es}; do
    gpu_free || log "GPUs busy"
    log "arm $arm: start"
    timeout -s INT -k 120 1500 python3 $R/Strata-1441/tools/vision_batch_check.py --exe $NEW --config $C/LL-$arm.json --batch 16 \
        --text-ab $OLD --long-tokens 40000 --logs $OUT/$arm --out $OUT/$arm.json > $OUT/$arm.txt 2>&1
    log "arm $arm: $(grep -a '==>' $OUT/$arm.txt | tail -1) $(grep -ac 'FAIL (v)' $OUT/$arm.txt) FAIL lines"
done
trap - INT TERM
gpu_free; [ -n "${NOPROD:-}" ] || start_prod; log done
