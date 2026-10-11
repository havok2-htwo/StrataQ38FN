#!/bin/bash
# driver arms (production args of T16-142, STRATA_TRACE=1, 24 measured admissions, no BYIELD): every new switch off vs on
D=/media/raid0/StrataQ38FN/linux/results/fixedcost/final
EXE=/media/raid0/StrataQ38FN/Strata/engine/strata-g3-142fc
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142.json
DRV=/media/raid0/StrataQ38FN/linux/results/fixedcost/s2/fc_driver.py
ALL="--env STRATA_PIPE_BYIELD_NODRAIN=1 --env STRATA_ADMIT_MERGE_FIRST=1 --env STRATA_CKPT_POOL=1 --env STRATA_CKPT_THREADS=1 --env STRATA_SLOT_CKPT_MOVE=1 --env STRATA_SLOT_COPY_BATCHSYNC=1"
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  case $arm in
    off) envs="" ;;
    on)  envs="$ALL" ;;
    *) echo "unknown $arm"; continue ;;
  esac
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-fc-final-drv-$arm \
    python3 $DRV --config $CFG --exe $EXE --out $D/drv-$arm --admissions 24 --byield none --env STRATA_TRACE=1 $envs \
    > $D/drv-$arm.driver.out 2>&1
  echo "$arm rc=$? $(date +%T)"
  python3 /media/raid0/StrataQ38FN/linux/results/overlap/m0/m0_analyze.py $D/drv-$arm > $D/drv-$arm/m0_phases.txt 2>&1
done
