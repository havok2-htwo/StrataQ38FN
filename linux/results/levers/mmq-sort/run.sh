#!/bin/bash
# lever 3 (MMQ group sort keeping the partial group): arms run one at a time, each in its own scope
D=/media/raid0/StrataQ38FN/linux/results/levers/mmq-sort
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  case $arm in
    t_ref)    c=cfg-ref; a="--mode prefill --phases idle --reps 6" ;;
    t_ne)     c=cfg-ref; a="--mode prefill --phases idle --reps 6 --env STRATA_MMQ_RESIDENT_SORT_NE=1" ;;
    t_gs)     c=cfg-lv;  a="--mode prefill --phases idle --reps 6 --env STRATA_MMQ_GROUP_SORT=1" ;;
    t_lv0)    c=cfg-lv;  a="--mode prefill --phases idle --reps 6" ;;
    d_ref)    c=cfg-ref; a="--mode prefill --phases idle --reps 1 --env STRATA_PREFILL_DUMP_R_ALL=$D/d_ref.rall --env STRATA_PREFILL_DUMP_R=$D/d_ref.r64" ;;
    d_gs)     c=cfg-lv;  a="--mode prefill --phases idle --reps 1 --env STRATA_MMQ_GROUP_SORT=1 --env STRATA_PREFILL_DUMP_R_ALL=$D/d_gs.rall --env STRATA_PREFILL_DUMP_R=$D/d_gs.r64" ;;
    d_ne)     c=cfg-ref; a="--mode prefill --phases idle --reps 1 --env STRATA_MMQ_RESIDENT_SORT_NE=1 --env STRATA_PREFILL_DUMP_R_ALL=$D/d_ne.rall --env STRATA_PREFILL_DUMP_R=$D/d_ne.r64" ;;
    *) echo "unknown $arm"; continue ;;
  esac
  case $arm in d_*) rm -f $D/$arm.rall $D/$arm.r64 ;; esac
  for i in $(seq 150); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  echo "$arm start $(date +%T)"
  PF_STAMP=fixed systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-lv-mmq-$arm \
    python3 $D/pf_driver.py --config $D/$c.json --out $D/$arm $a > $D/$arm.driver.out 2>&1
  echo "$arm rc=$? $(date +%T)"
  sleep 5
done
