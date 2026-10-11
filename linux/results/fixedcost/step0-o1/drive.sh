#!/bin/bash
# driver arms (production args of T16-142, STRATA_TRACE=1, 24 measured admissions, BYIELD none/read/late round robin)
D=/media/raid0/StrataQ38FN/linux/results/fixedcost/step0-o1
EXE=/media/raid0/StrataQ38FN/Strata/engine/strata-g3-142fc
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142.json
GL="GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=17179869184"
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  case $arm in
    off)   envs="" ;;
    o1)    envs="--env STRATA_PIPE_BYIELD_NODRAIN=1" ;;
    glibc) envs="--env ${GL#--env }" ;;
    *) echo "unknown $arm"; continue ;;
  esac
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-fc-drv-$arm \
    python3 $D/fc_driver.py --config $CFG --exe $EXE --out $D/drv-$arm --admissions 24 --env STRATA_TRACE=1 $envs \
    > $D/drv-$arm.driver.out 2>&1
  echo "$arm rc=$? $(date +%T)"
done
