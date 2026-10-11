#!/bin/bash
# fixedcost final exactness: engine/strata-g3-142fc (g3-142 c9bbd1fc) with every new switch on vs all off vs the
# reference engine/strata-g3-142, config T16-142, exact flags.  Usage: run.sh <arm>...  (logs: <arm>.out, .engine.log, .json)
R=/media/raid0/StrataQ38FN/linux/results/fixedcost/final
REF=/media/raid0/StrataQ38FN/linux/results/rebase142/gpu
S=/media/raid0/StrataQ38FN/Strata
T=/media/raid0/StrataQ38FN/Strata-142/tools
FC=$S/engine/strata-g3-142fc
RE=$S/engine/strata-g3-142
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142.json
EXACT="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
ALL="STRATA_PIPE_BYIELD_NODRAIN=1 STRATA_ADMIT_MERGE_FIRST=1 STRATA_CKPT_POOL=1 STRATA_CKPT_THREADS=1 STRATA_SLOT_CKPT_MOVE=1 STRATA_SLOT_COPY_BATCHSYNC=1"
ALLE=""; for kv in $ALL; do ALLE="$ALLE --env $kv"; done
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
go() {  # name envs cmd...
  local name=$1 envs=$2; shift 2
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "=== $name: GPUs busy, skipped" >> $R/summary.txt; return; }
  echo "=== $name $(date +%T) env=[$envs] cmd=[$*]" >> $R/summary.txt
  local t0=$(date +%s)
  env STRATA_MMQ_DIRECT=1 STRATA_IQ_MT_MIN=1 STRATA_SLOT_COPY_DIRECT=1 $envs BATCH_TEST_LOG=$R/$name.engine.log \
    systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-fc-final-$name "$@" > $R/$name.out 2>&1
  echo "rc=$? $(date +%T) ($(( $(date +%s) - t0 )) s)" >> $R/summary.txt
  grep -E "^batch:|DIFFERS|DID NOT|^G2: |^note:|^log:|compare|RESULT|D gave|ERR|requests differ|DIFFER|^solo:|^batch:" $R/$name.out | grep -v "IDENTICAL$" >> $R/summary.txt
  echo "identical slots: $(grep -c 'IDENTICAL$' $R/$name.out); admissions $(grep -c 'admission of slot' $R/$name.engine.log), merged windows $(grep -c 'the first window takes' $R/$name.engine.log), watchdog/terminate $(grep -ciE 'watchdog|terminate called' $R/$name.engine.log)" >> $R/summary.txt
}
for arm in "$@"; do
  case $arm in
    bt16-ref) go $arm "" python3 $T/batch_test16.py --exe $RE --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$EXACT" --dump $R/$arm.json ;;
    bt16-off) go $arm "STRATA_TRACE=1" python3 $T/batch_test16.py --exe $FC --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$EXACT" --dump $R/$arm.json ;;
    bt16-on)  go $arm "$ALL STRATA_TRACE=1" python3 $T/batch_test16.py --exe $FC --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$EXACT" --dump $R/$arm.json ;;
    ov-ref)   go $arm "" python3 $T/batch_overlap_test.py --exe $RE --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 --solo-json $REF/ov-on.json --compare $REF/ov-on-sc.json ;;
    ov-off)   go $arm "" python3 $T/batch_overlap_test.py --exe $FC --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 --env STRATA_TRACE=1 --solo-json $REF/ov-on.json --compare $R/ov-ref.json ;;
    ov-on)    go $arm "" python3 $T/batch_overlap_test.py --exe $FC --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 $ALLE --env STRATA_TRACE=1 --solo-json $REF/ov-on.json --compare $R/ov-ref.json ;;
    il-ref)   go $arm "" python3 $T/batch_interleave_test.py --exe $RE --config $CFG --extra "$EXACT" ;;
    il-off)   go $arm "STRATA_TRACE=1" python3 $T/batch_interleave_test.py --exe $FC --config $CFG --extra "$EXACT" ;;
    il-on)    go $arm "$ALL STRATA_TRACE=1" python3 $T/batch_interleave_test.py --exe $FC --config $CFG --extra "$EXACT" ;;
    *) echo "unknown arm $arm" >> $R/summary.txt ;;
  esac
done
echo "DONE $* $(date +%T)" >> $R/summary.txt
