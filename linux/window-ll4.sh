#!/usr/bin/env bash
# GPU window (approved 2026-10-16): round 2 of the low-level levers on strata-g3-1441c (= production 1441b + opt-in
# STRATA_GR_DOWN_SPLIT / STRATA_GR_UP_SPLIT (HC read on 2-4x the blocks), STRATA_ROUTE_PLAN_FUSE (top-10 + plan in one
# launch), STRATA_SPLIT_OWN_RESERVE_MIB).  Reference = production's strata-g3-1441b.  Test engines park into
# results/ll/w4/tier.  Production comes back unchanged.  Results: results/ll/w4.
set -u
R=/media/raid0/StrataQ38FN; OUT=$R/linux/results/ll/w4; mkdir -p "$OUT"; URL=http://127.0.0.1:8090; C=$R/linux/configs
NEW=$R/Strata/engine/strata-g3-1441c; OLD=$R/Strata/engine/strata-g3-1441b; B=$R/Strata-1441-ll/build-ll2
PROD=$C/LL2-prod.json
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
trap 'log "window aborted"; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus; start_prod; exit 1' INT TERM
step() { local name=$1 limit=$2; shift 2; log "step $name: start"
    timeout -s INT -k 120 $limit "$@" > "$OUT/$name.txt" 2>&1; local rc=$?; echo $rc > "$OUT/rc_$name"
    log "step $name: exit $rc $(grep -aE '==>|tests passed' "$OUT/$name.txt" | tail -1 | cut -c1-140)"
    free_gpus || log "GPUs busy after $name"; }
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }

log "window start: stopping strata.service"; systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || { log "GPUs not free"; start_prod; exit 1; }
# 1. kernel parity: every gr_parity arm (incl. the new split arms), the fused router/plan test, the QSA tests
step ctest 1500 bash -c "cd $B && CUDA_VISIBLE_DEVICES=0 ctest --output-on-failure -j1 -R 'gr_parity|route_plan|qsa_slots|qsa_vbatch|kv_stream_parity|kv_hybrid_parity|sampler_same'"
# 2. HC read micro-bench incl. the SPLIT section (T=1..8, cold weights), production's HC switches set
step grbench 900 bash -c "cd $B && for a in '' 'STRATA_GR_DOWN_SPLIT=2 STRATA_GR_UP_SPLIT=2' 'STRATA_GR_DOWN_SPLIT=4 STRATA_GR_UP_SPLIT=4'; do echo \"=== arm [\$a]\"; env CUDA_VISIBLE_DEVICES=1 STRATA_HC_SPLIT=2 STRATA_GR_FAST=1 STRATA_GR_DOWN_PF=2 STRATA_GR_NORM_FAST=1 \$a ./fused_gr_bench 300 1 8 8; done"
# 3. engine exactness, greedy, 16 slots, a 40K slot, images: both split widths + fuse vs production's binary
step text_ab4 2400 python3 $R/Strata-1441/tools/vision_batch_check.py --exe $NEW --config $C/LL2-all4.json --batch 16 --text-ab $OLD --long-tokens 40000 --logs $OUT/text-ab4 --out $OUT/text_ab4.json
step text_ab2 2400 python3 $R/Strata-1441/tools/vision_batch_check.py --exe $NEW --config $C/LL2-all2.json --batch 16 --text-ab $OLD --long-tokens 40000 --logs $OUT/text-ab2 --out $OUT/text_ab2.json
# 4. T=0.7 with a fixed seed: old vs new (all4)
K="temperature=0.7 top_k=20 seed=1234"
step t07_old 1500 bash -c "cd $R/Strata-1441 && env $(cfg_env $PROD) python3 tools/batch_test16.py --exe $OLD --config $PROD --batch 16 --n 16 --max-new 160 --keys '$K' --long-tokens 40000 --dump $OUT/t07_old.json"
step t07_new 1500 bash -c "cd $R/Strata-1441 && env $(cfg_env $C/LL2-all4.json) python3 tools/batch_test16.py --exe $NEW --config $C/LL2-all4.json --batch 16 --n 16 --max-new 160 --keys '$K' --long-tokens 40000 --dump $OUT/t07_new.json"
cmp -s $OUT/t07_old.json $OUT/t07_new.json && log "t07 old vs new (all4): IDENTICAL" || log "t07 old vs new (all4): DIFFER"
# 5. pure decode rows/s, 16 slots, 30 s per arm
for a in prod hc2 hc4 fuse all2 all4; do
  log "decode $a: start"
  timeout -s INT 900 python3 $R/linux/results/profile2/prof_driver.py --config $C/LL2-$a.json --out $OUT/dec_$a --mode decode --batch 16 --seconds 30 > $OUT/dec_$a.driver.out 2>&1
  log "decode $a: $(python3 $R/linux/results/profile2/dec_rows.py $OUT/dec_$a/events.log 2>&1 | tail -1 | sed 's#.*events.log: ##')"; free_gpus
done
# 6. HTTP, 16 users, ABBA x2: production vs the two combined arms
log "step ab: start"
( cd $R/linux && SERVER_DIR=$R/Strata-143-prod REPS=2 ORDER=abba timeout -s INT 4000 ./ab_quick.sh ll4 $PROD $C/LL2-all2.json $C/LL2-all4.json ) > $OUT/ab.txt 2>&1
log "step ab: exit $?"; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus || log "GPUs busy after ab"
trap - INT TERM
start_prod; log "window done"
