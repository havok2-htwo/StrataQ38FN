#!/usr/bin/env bash
# GPU window (approved 2026-10-16): low-level levers on g3-1441-ll (engine strata-g3-1441ll = 1441a + 4 opt-in switches:
# STRATA_QSA_SLOTS_STREAM, STRATA_ATTN_VBATCH, STRATA_GR_DOWN_PF / STRATA_GR_NORM_FAST, STRATA_EARLY_SAMPLE; plus the
# zero-code arms STRATA_GR_FAST, STRATA_MMVF_ROWS).  Production comes back exactly as before.  Results: results/ll/.
set -u
R=/media/raid0/StrataQ38FN; OUT=$R/linux/results/ll; mkdir -p "$OUT"; URL=http://127.0.0.1:8090
NEW=$R/Strata/engine/strata-g3-1441ll; OLD=$R/Strata/engine/strata-g3-1441a; C=$R/linux/configs
PROD=$C/T16-1441-prod262r-vision-cpu.json; ALL=$C/LL-all.json; B=$R/Strata-1441-ll/build-ll
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
    log "step $name: exit $rc $(grep -aE 'VERDICT|==>|tests passed|PASS|FAIL' "$OUT/$name.txt" | tail -1 | cut -c1-140)"
    free_gpus || log "GPUs busy after $name"; }
# the production config's env (batch_test16 takes the engine env from its own environment, not the config)
cfg_env() { python3 -c "import json,sys,shlex; print(' '.join(k+'='+shlex.quote(v) for k,v in json.load(open(sys.argv[1]))['env'].items()))" "$1"; }

log "window start: stopping strata.service"; systemctl --user stop strata.service
for i in $(seq 60); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
free_gpus || { log "GPUs not free"; start_prod; exit 1; }

# 1. kernel parity (synthetic): the new tests first, then the whole suite (ple_parity / conversation_snapshot_test failed
#    in the r1441 window already, see results/r1441/gpu_ctest.txt)
step ctest_new 1200 bash -c "cd $B && CUDA_VISIBLE_DEVICES=0 ctest --output-on-failure -j1 -R 'qsa_slots_stream_parity|qsa_vbatch_parity|gr_parity|qsa_slots_parity|kv_stream_parity|kv_hybrid_parity|sampler_same_test'"
# 2. HC read micro-bench: staged read, T=4 (one stage window's rows), cold weights; hashes must match across arms
step grbench 900 bash -c "cd $B && for a in '' STRATA_GR_FAST=1 STRATA_GR_DOWN_PF=2 STRATA_GR_DOWN_PF=3 STRATA_GR_NORM_FAST=1 'STRATA_GR_FAST=1 STRATA_GR_DOWN_PF=2 STRATA_GR_NORM_FAST=1' 'STRATA_GR_FAST=1 STRATA_GR_DOWN_PF=3 STRATA_GR_NORM_FAST=1'; do echo \"=== arm [\$a]\"; env CUDA_VISIBLE_DEVICES=1 STRATA_HC_SPLIT=2 \$a ./fused_gr_bench 300 1 8 8; done"
# 3. exactness, greedy, engine level: all switches on vs production's binary, batch 16, a 40K-token slot (evicts)
step text_ab 2400 python3 $R/Strata-1441/tools/vision_batch_check.py --exe $NEW --config $ALL --batch 16 --text-ab $OLD --long-tokens 40000 --logs $OUT/text-ab --out $OUT/text_ab.json
# 4. exactness, temperature 0.7 (early sampling only acts on sampled rows): token dumps old vs new, compared below
step t07_old 1500 bash -c "cd $R/Strata-1441 && env $(cfg_env $PROD) python3 tools/batch_test16.py --exe $OLD --config $PROD --batch 16 --n 16 --max-new 160 --keys 'temperature=0.7 top_k=20' --long-tokens 40000 --dump $OUT/t07_old.json"
step t07_new 1500 bash -c "cd $R/Strata-1441 && env $(cfg_env $ALL) python3 tools/batch_test16.py --exe $NEW --config $ALL --batch 16 --n 16 --max-new 160 --keys 'temperature=0.7 top_k=20' --long-tokens 40000 --dump $OUT/t07_new.json"
cmp -s $OUT/t07_old.json $OUT/t07_new.json && log "t07 dumps: IDENTICAL" || log "t07 dumps: DIFFER"
# 5. speed over HTTP (production's server), 16 users greedy short + 1.2K, arms alternating ABBA x2; replies vs arm 1
log "step ab: start"
( cd $R/linux && SERVER_DIR=$R/Strata-143-prod REPS=2 ORDER=abba timeout -s INT 6000 ./ab_quick.sh ll $PROD $C/LL-off.json $C/LL-qss.json $C/LL-vb.json $C/LL-gr.json $ALL ) > $OUT/ab.txt 2>&1
log "step ab: exit $?"; systemctl --user stop strata-bench.scope 2>/dev/null; free_gpus || log "GPUs busy after ab"
trap - INT TERM
start_prod; log "window done"
