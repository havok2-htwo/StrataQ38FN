#!/usr/bin/env bash
# follow-up window: T=0.7 exactness with a fixed seed (window 2's t07 had none: the engine seeds from the clock), and the
# feature-ON build (MMQ_KQUANTS / Q6K_EXPERTS on, upstream defaults) against production's binary.  Production back after.
set -u
R=/media/raid0/StrataQ38FN; OUT=$R/linux/results/ll/w3; mkdir -p "$OUT"; URL=http://127.0.0.1:8090; C=$R/linux/configs
NEW=$R/Strata/engine/strata-g3-1441ll; FULL=$R/Strata/engine/strata-g3-1441ll-full; OLD=$R/Strata/engine/strata-g3-1441a
PROD=$C/T16-1441-prod262r-vision-cpu.json; ALL=$C/LL-all.json
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window.log"; }
gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader; }
free_gpus() { for i in $(seq 45); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done
    for p in $(gpu_apps); do case "$(readlink -f /proc/$p/exe 2>/dev/null)" in $R/Strata*) kill "$p" ;; esac; done
    for i in $(seq 30); do [ -z "$(gpu_apps)" ] && return 0; sleep 2; done; return 1; }
start_prod() { log "starting production"; systemctl --user start strata.service; local t0; t0=$(date +%s)
    until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
        [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "start timeout"; return 1; }; sleep 5; done
    log "production loaded after $(( $(date +%s) - t0 )) s"; }
trap 'log "window aborted"; free_gpus; start_prod; exit 1' INT TERM
step() { local name=$1 limit=$2; shift 2; log "step $name: start"
    timeout -s INT -k 120 $limit "$@" > "$OUT/$name.txt" 2>&1; local rc=$?; echo $rc > "$OUT/rc_$name"
    log "step $name: exit $rc $(grep -aE '==>' "$OUT/$name.txt" | tail -1 | cut -c1-140)"; free_gpus || log "GPUs busy after $name"; }
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }
log "window start: stopping strata.service"; systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || { log "GPUs not free"; start_prod; exit 1; }
K="temperature=0.7 top_k=20 seed=1234"
step t07_old 1500 bash -c "cd $R/Strata-1441 && env $(cfg_env $PROD) python3 tools/batch_test16.py --exe $OLD --config $PROD --batch 16 --n 16 --max-new 160 --keys '$K' --long-tokens 40000 --dump $OUT/t07_old.json"
step t07_old2 1500 bash -c "cd $R/Strata-1441 && env $(cfg_env $PROD) python3 tools/batch_test16.py --exe $OLD --config $PROD --batch 16 --n 16 --max-new 160 --keys '$K' --long-tokens 40000 --dump $OUT/t07_old2.json"
step t07_new 1500 bash -c "cd $R/Strata-1441 && env $(cfg_env $ALL) python3 tools/batch_test16.py --exe $NEW --config $ALL --batch 16 --n 16 --max-new 160 --keys '$K' --long-tokens 40000 --dump $OUT/t07_new.json"
cmp -s $OUT/t07_old.json $OUT/t07_old2.json && log "t07 old vs old (determinism): IDENTICAL" || log "t07 old vs old (determinism): DIFFER"
cmp -s $OUT/t07_old.json $OUT/t07_new.json && log "t07 old vs new (all switches): IDENTICAL" || log "t07 old vs new (all switches): DIFFER"
step text_ab_full 2400 python3 $R/Strata-1441/tools/vision_batch_check.py --exe $FULL --config $C/LL-all-full.json --batch 16 --text-ab $OLD --long-tokens 40000 --logs $OUT/text-ab-full --out $OUT/text_ab_full.json
trap - INT TERM
free_gpus; start_prod; log "window done"
