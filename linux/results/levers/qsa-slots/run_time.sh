#!/bin/bash
# batch decode rows/s and STRATA_VERIFY_PROFILE, reference (strata-g3-143) vs new (strata-g3-143l5 + STRATA_QSA_SLOTS=1),
# T16-143gsc env + STRATA_NAN_GUARD=0; arms: {ref,new,off}{16,32}[p][_rN]
D=/media/raid0/StrataQ38FN/linux/results/levers/qsa-slots
P=/media/raid0/StrataQ38FN/linux/results/profile2
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  base=${arm%%_*}; who=${base:0:3}; rest=${base:3}; b=${rest%p}
  a="--mode decode --batch $b --seconds 30"
  [ "${rest: -1}" = p ] && a="$a --env STRATA_VERIFY_PROFILE=1"
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  echo "$arm start $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-l5-$arm \
    python3 $P/prof_driver.py --config $D/time_$who.json --out $D/t_$arm $a > $D/t_$arm.driver.out 2>&1
  echo "$arm rc=$? $(date +%T)"
  python3 $P/dec_rows.py $D/t_$arm/events.log 2>&1 | tail -1
  sleep 5
done
