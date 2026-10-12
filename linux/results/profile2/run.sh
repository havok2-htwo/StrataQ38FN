#!/bin/bash
# profile of the T16-142fca default (2026-10-12): prefill phases, batch decode profile, nsys traces
D=/media/raid0/StrataQ38FN/linux/results/profile2
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142fca.json
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  case $arm in
    pf_plain)  a="--mode prefill" ;;
    pf_timing) a="--mode prefill --env STRATA_PREFILL_TIMING=1" ;;
    dec16_plain) a="--mode decode --batch 16 --seconds 30" ;;
    dec16_prof)  a="--mode decode --batch 16 --seconds 30 --env STRATA_VERIFY_PROFILE=1" ;;
    dec32_plain) a="--mode decode --batch 32 --seconds 30" ;;
    dec32_prof)  a="--mode decode --batch 32 --seconds 30 --env STRATA_VERIFY_PROFILE=1" ;;
    nsys_pf)   a="--mode prefill --reps 2 --phases dec15,idle --nsys prpf --nsys-cap dec15:1200:1,dec15:150:1,idle:1200:1,idle:150:1" ;;
    nsys_dec16) a="--mode decode --batch 16 --seconds 12 --nsys prd16 --nsys-sec 1.0" ;;
    nsys_dec32) a="--mode decode --batch 32 --seconds 12 --nsys prd32 --nsys-sec 1.0" ;;
    nsys_dec16g) a="--mode decode --batch 16 --seconds 12 --nsys prd16g --nsys-sec 1.0 --nsys-graph graph" ;;
    nsys_dec32g) a="--mode decode --batch 32 --seconds 12 --nsys prd32g --nsys-sec 1.0 --nsys-graph graph" ;;
    *) echo "unknown $arm"; continue ;;
  esac
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  echo "$arm start $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-pr-$arm \
    python3 $D/prof_driver.py --config $CFG --out $D/$arm $a > $D/$arm.driver.out 2>&1
  echo "$arm rc=$? $(date +%T)"
  sleep 5
done
