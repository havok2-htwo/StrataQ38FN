#!/bin/bash
# arms: plain (STRATA_TRACE only) and timing (+ STRATA_PREFILL_TIMING=1); config T16-142fc (all fixed-cost switches)
D=/media/raid0/StrataQ38FN/linux/results/batchadmit/scaling
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142fc.json
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  case $arm in
    plain) envs="" ;;
    timing) envs="--env STRATA_PREFILL_TIMING=1" ;;
    *) echo "unknown $arm"; continue ;;
  esac
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-ba-scale-$arm \
    python3 $D/scale_driver.py --config $CFG --out $D/$arm --reps 4 $envs > $D/$arm.driver.out 2>&1
  echo "$arm rc=$? $(date +%T)"
done
