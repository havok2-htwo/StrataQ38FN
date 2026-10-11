#!/bin/bash
# fixedcost review fixes: one arm per call.  Usage: run.sh <arm>...  (logs: <arm>.out, <arm>.engine.log, <arm>.json)
R=/media/raid0/StrataQ38FN/linux/results/fixedcost/review-fix
S=/media/raid0/StrataQ38FN/Strata
T=/media/raid0/StrataQ38FN/Strata-142/tools
EXE=$S/engine/strata-g3-142fc
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142.json
EXACT="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
go() {  # name envs cmd...
  local name=$1 envs=$2; shift 2
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "=== $name: GPUs busy, skipped" >> $R/summary.txt; return; }
  echo "=== $name $(date +%T) env=[$envs] cmd=[$*]" >> $R/summary.txt
  local t0=$(date +%s)
  env STRATA_MMQ_DIRECT=1 STRATA_IQ_MT_MIN=1 STRATA_SLOT_COPY_DIRECT=1 $envs BATCH_TEST_LOG=$R/$name.engine.log \
    systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-fc-$name "$@" > $R/$name.out 2>&1
  echo "rc=$? $(date +%T) ($(( $(date +%s) - t0 )) s)" >> $R/summary.txt
  grep -E "^batch:|DIFFERS|DID NOT|^D3|^note:|compare|RESULT|ERR|DIFFER|threads new|after the aux|single-CPU" $R/$name.out | grep -v "IDENTICAL$" >> $R/summary.txt
  echo "identical slots: $(grep -c 'IDENTICAL$' $R/$name.out); 'gave back' $(grep -c 'gave back' $R/$name.engine.log); decode timing 1-window lines: $(grep 'decode timing: 1 windows' $R/$name.engine.log | grep -o 'avg T [0-9.]*' | sort | uniq -c | tr '\n' ' ')" >> $R/summary.txt
}
BT16=(python3 $T/batch_test16.py --exe $EXE --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$EXACT")
EDGE=(python3 $R/fix_edge.py --exe $EXE --config $CFG --extra "$EXACT")
ALL="STRATA_SLOT_CKPT_MOVE=1 STRATA_CKPT_POOL=1 STRATA_CKPT_THREADS=1 STRATA_ADMIT_MERGE_FIRST=1 STRATA_SLOT_COPY_BATCHSYNC=1"
for arm in "$@"; do
  case $arm in
    edge-off) go $arm "STRATA_TRACE=1" "${EDGE[@]}" --dump $R/$arm.json ;;
    edge-all) go $arm "$ALL STRATA_TRACE=1" "${EDGE[@]}" --dump $R/$arm.json --compare $R/edge-off.json --allow-resume D2 ;;
    bt16-all) go $arm "$ALL STRATA_TRACE=1" "${BT16[@]}" --dump $R/$arm.json ;;
    aux)      go $arm "STRATA_CKPT_POOL=1 STRATA_CKPT_THREADS=1" python3 $R/aux_check.py --exe $EXE --config $CFG --extra "$EXACT --aux-cpus auto" ;;
    *) echo "unknown arm $arm" >> $R/summary.txt ;;
  esac
done
echo "DONE $* $(date +%T)" >> $R/summary.txt
