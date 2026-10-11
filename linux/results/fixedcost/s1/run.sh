#!/bin/bash
# fixedcost step S1 exactness: one arm per call.  Usage: run.sh <arm>...  (logs: <arm>.out, <arm>.engine.log, <arm>.json)
# engine/strata-g3-142fc (g3-142 + step 0 trace + O1 + AB + K1/K2-lite + STRATA_SLOT_CKPT_MOVE), config T16-142 (STRATA_SLOT_COPY_DIRECT=1).
R=/media/raid0/StrataQ38FN/linux/results/fixedcost/s1
REF=/media/raid0/StrataQ38FN/linux/results/rebase142/gpu
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
  grep -E "^batch:|DIFFERS|DID NOT|^G2: |^note:|^log:|compare|RESULT|D gave|ERR|requests differ|DIFFER" $R/$name.out | grep -v "IDENTICAL$" >> $R/summary.txt
  echo "identical slots: $(grep -c 'IDENTICAL$' $R/$name.out); engine log: 'admission of slot' $(grep -c 'admission of slot' $R/$name.engine.log), moved $(grep -c 'checkpoint copy [0-9.]* (moved)' $R/$name.engine.log), restored from a slot's $(grep -c "restored from slot" $R/$name.engine.log), moved back $(grep -c "moved back into the chain" $R/$name.engine.log), gave back turn ckpt $(grep -c "(its turn checkpoint)" $R/$name.engine.log), gave back all $(grep -c "(all it holds)" $R/$name.engine.log), watchdog/terminate $(grep -ciE 'watchdog|terminate called' $R/$name.engine.log)" >> $R/summary.txt
}
BT16=(python3 $T/batch_test16.py --exe $EXE --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$EXACT")
OVT=(python3 $T/batch_overlap_test.py --exe $EXE --config $CFG --extra "$EXACT")
EDGE=(python3 $R/s1_edge.py --exe $EXE --config $CFG --extra "$EXACT")
S1="STRATA_SLOT_CKPT_MOVE=1"
ALL="STRATA_SLOT_CKPT_MOVE=1 STRATA_CKPT_POOL=1 STRATA_CKPT_THREADS=1 STRATA_ADMIT_MERGE_FIRST=1"
for arm in "$@"; do
  case $arm in
    edge-off) go $arm "STRATA_TRACE=1" "${EDGE[@]}" --dump $R/$arm.json ;;
    edge-on)  go $arm "$S1 STRATA_TRACE=1" "${EDGE[@]}" --dump $R/$arm.json --compare $R/edge-off.json --allow-resume D2 ;;
    edge-all) go $arm "$ALL STRATA_TRACE=1" "${EDGE[@]}" --dump $R/$arm.json --compare $R/edge-off.json --allow-resume D2 ;;
    ov-all)   go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=0 --env STRATA_SLOT_CKPT_MOVE=1 --env STRATA_CKPT_POOL=1 --env STRATA_CKPT_THREADS=1 --env STRATA_ADMIT_MERGE_FIRST=1 --env STRATA_TRACE=1 --solo-json $REF/ov-on.json --compare $R/../step0-o1/ov-off.json ;;
    il-all)   go $arm "$ALL STRATA_TRACE=1" python3 $T/batch_interleave_test.py --exe $EXE --config $CFG --extra "$EXACT" ;;
    bt16-all) go $arm "$ALL STRATA_TRACE=1" "${BT16[@]}" --dump $R/$arm.json ;;
    *) echo "unknown arm $arm" >> $R/summary.txt ;;
  esac
done
echo "DONE $* $(date +%T)" >> $R/summary.txt
