#!/bin/bash
# STRATA_HANDOFF_PIPE timing: prefill reads (150/1200, idle + 15 decoding) and decode rows/s, back to back.
# arms: pf_<who>[_rN] (who = ref|off|p4|p8|p16, config time_<who>.json), dec_<who>[_rN]
D=/media/raid0/StrataQ38FN/linux/results/levers/handoff
P=/media/raid0/StrataQ38FN/linux/results/profile2
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  kind=${arm%%_*}; rest=${arm#*_}; who=${rest%%_*}
  case $kind in
    pf)  a="--mode prefill --reps 8" ;;
    dec) a="--mode decode --batch 16 --seconds 30" ;;
  esac
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  echo "$arm start $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-kh-$arm \
    python3 $P/prof_driver.py --config $D/time_$who.json --out $D/t_$arm $a > $D/t_$arm.driver.out 2>&1
  echo "$arm rc=$? $(date +%T)"
  if [ $kind = pf ]; then python3 $P/pf_reads.py $D/t_$arm/events.log; else python3 $P/dec_rows.py $D/t_$arm/events.log 2>&1 | tail -1; fi
  sleep 5
done
