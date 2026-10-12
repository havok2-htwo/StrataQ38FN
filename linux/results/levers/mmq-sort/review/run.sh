#!/bin/bash
# review fix (whole_tiles verdict cached per layer in MmqPlan::group_sort): lever 3's covering gates, one arm at a time
D=/media/raid0/StrataQ38FN/linux/results/levers/mmq-sort/review
P=/media/raid0/StrataQ38FN/linux/results/levers/mmq-sort
T=/media/raid0/StrataQ38FN/Strata-142/tools
EXE=/media/raid0/StrataQ38FN/Strata/engine/strata-g3-142lv
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142fcb.json
EXTRA="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 --layer-split 12,24,37"
CFGENV="STRATA_MMQ_DIRECT=1 STRATA_SLOT_COPY_DIRECT=1 STRATA_PIPE_BYIELD_NODRAIN=1 STRATA_ADMIT_MERGE_FIRST=1 STRATA_CKPT_POOL=1 STRATA_CKPT_THREADS=1 STRATA_SLOT_CKPT_MOVE=1 STRATA_SLOT_COPY_BATCHSYNC=1 STRATA_SPLIT_MTP_BATCH=1 STRATA_IQ_MT_MIN=1"
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  pd=""; cmd=""
  case $arm in
    t_ref) pd="--config $P/cfg-ref.json --mode prefill --phases idle --reps 6" ;;
    t_gs)  pd="--config $P/cfg-lv.json --mode prefill --phases idle --reps 6 --env STRATA_MMQ_GROUP_SORT=1" ;;
    d_ref) pd="--config $P/cfg-ref.json --mode prefill --phases idle --reps 1 --env STRATA_PREFILL_DUMP_R_ALL=$D/d_ref.rall --env STRATA_PREFILL_DUMP_R=$D/d_ref.r64" ;;
    d_gs)  pd="--config $P/cfg-lv.json --mode prefill --phases idle --reps 1 --env STRATA_MMQ_GROUP_SORT=1 --env STRATA_PREFILL_DUMP_R_ALL=$D/d_gs.rall --env STRATA_PREFILL_DUMP_R=$D/d_gs.r64" ;;
    bt16_gs) cmd="python3 $T/batch_test16.py --exe $EXE --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra \"$EXTRA\" --dump $D/bt16-gs.json" ;;
    *) echo "unknown $arm"; continue ;;
  esac
  case $arm in d_*) rm -f $D/$arm.rall $D/$arm.r64 ;; esac
  for i in $(seq 150); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  echo "$arm start $(date +%T)"
  if [ -n "$pd" ]; then
    PF_STAMP=fixed systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-lv-mmqr-$arm \
      python3 $P/pf_driver.py --out $D/$arm $pd > $D/$arm.driver.out 2>&1
  else
    systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-lv-mmqr-$arm \
      --setenv=BATCH_TEST_LOG=$D/$arm.engine.log $(for kv in $CFGENV STRATA_MMQ_GROUP_SORT=1; do echo --setenv=$kv; done) \
      bash -c "$cmd" > $D/$arm.out 2>&1
  fi
  echo "$arm rc=$? $(date +%T)"
  sleep 5
done
