#!/bin/bash
# exactness-r1: one arm per call.  Usage: run.sh <arm>...   (logs: <arm>.out, <arm>.engine.log, <arm>.json)
R=/media/raid0/StrataQ38FN/linux/results/overlap/exactness-r1
S=/media/raid0/StrataQ38FN/Strata
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-mmqtab.json
CFG1325=$R/T16-mmqtab-1325.json
EXACT="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
OV=strata-g3-overlap REF=strata-g3-mmqtab
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
go() {  # name envs cmd...
  local name=$1 envs=$2; shift 2
  for i in $(seq 60); do gpu_free && break; sleep 2; done
  gpu_free || { echo "=== $name: GPUs busy, skipped" >> $R/summary.txt; return; }
  echo "=== $name $(date +%T) env=[$envs] cmd=[$*]" >> $R/summary.txt
  env STRATA_MMQ_DIRECT=1 STRATA_IQ_MT_MIN=1 $envs BATCH_TEST_LOG=$R/$name.engine.log \
    systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-ov-x1-$name "$@" > $R/$name.out 2>&1
  echo "rc=$? $(date +%T)" >> $R/summary.txt
  grep -E "^batch:|DIFFERS|DID NOT|^G2: |^note:|^log:|compare|RESULT|^fault:|   log:|D gave|IDENTICAL$|ERR|requests differ" $R/$name.out | grep -v "IDENTICAL$" >> $R/summary.txt
  echo "overlap lines in engine log: 'overlapped read of' $(grep -c 'overlapped read of' $R/$name.engine.log), 'eligible' $(grep -c 'eligible' $R/$name.engine.log), 'overlapped read' $(grep -c 'overlapped read' $R/$name.engine.log), 'not captured before' $(grep -c 'not captured before' $R/$name.engine.log), ERR-ish $(grep -ciE 'watchdog|terminate called' $R/$name.engine.log)" >> $R/summary.txt
}
OVT=(python3 $S/tools/batch_overlap_test.py --exe $S/engine/$OV --config $CFG --extra "$EXACT")
OVT_REF=(python3 $S/tools/batch_overlap_test.py --exe $S/engine/$REF --config $CFG --extra "$EXACT")
F="--env STRATA_PIPE_OVERLAP=1 --expect-fault --bg 4 --bg-new 200"
for arm in "$@"; do
  case $arm in
    bt16-off)  go $arm "" python3 $S/tools/batch_test16.py --exe $S/engine/$OV --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$EXACT" --dump $R/$arm.json ;;
    bt32-off)  go $arm "" python3 $S/tools/batch_test32.py --exe $S/engine/$OV --config $CFG --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra "$EXACT" --dump $R/$arm.json ;;
    bt32-ref)  go $arm "" python3 $S/tools/batch_test32.py --exe $S/engine/$REF --config $CFG --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra "$EXACT" --dump $R/$arm.json ;;
    bt16-off-1325) go $arm "" python3 $S/tools/batch_test16.py --exe $S/engine/$OV --config $CFG1325 --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$EXACT" --dump $R/$arm.json ;;
    bt32-off-1325) go $arm "" python3 $S/tools/batch_test32.py --exe $S/engine/$OV --config $CFG1325 --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra "$EXACT" --dump $R/$arm.json ;;
    il-off)    go $arm "" python3 $S/tools/batch_interleave_test.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" ;;
    il-ref)    go $arm "" python3 $S/tools/batch_interleave_test.py --exe $S/engine/$REF --config $CFG --extra "$EXACT" ;;
    il-off-1325) go $arm "" python3 $S/tools/batch_interleave_test.py --exe $S/engine/$OV --config $CFG1325 --extra "$EXACT" ;;
    il-on)     go $arm "STRATA_PIPE_OVERLAP=1" python3 $S/tools/batch_interleave_test.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" ;;
    ov-on)     go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 ;;
    ov-off)    go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=0 --solo-json $R/ov-on.json --compare $R/ov-on.json ;;
    ov-ref)    go $arm "" "${OVT_REF[@]}" --dump $R/$arm.json --solo-json $R/ov-on.json --compare $R/ov-on.json ;;
    ov-on-1325)  go $arm "" "${OVT[@]}" --dump $R/$arm.json --split 13,25,37 --env STRATA_PIPE_OVERLAP=1 ;;
    ov-off-1325) go $arm "" "${OVT[@]}" --dump $R/$arm.json --split 13,25,37 --env STRATA_PIPE_OVERLAP=0 --solo-json $R/ov-on-1325.json --compare $R/ov-on-1325.json ;;
    ov-padup)  go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_PADUP=1 --solo-json $R/ov-on.json --compare $R/ov-off.json ;;
    ov-wait1)  go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_WAIT=1 --solo-json $R/ov-on.json --compare $R/ov-off.json ;;
    ov-wait2)  go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_WAIT=2 --solo-json $R/ov-on.json --compare $R/ov-off.json ;;
    ov-padup-1325) go $arm "" "${OVT[@]}" --dump $R/$arm.json --split 13,25,37 --env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_PADUP=1 --solo-json $R/ov-on-1325.json --compare $R/ov-off-1325.json ;;
    ov-wait1-1325) go $arm "" "${OVT[@]}" --dump $R/$arm.json --split 13,25,37 --env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_WAIT=1 --solo-json $R/ov-on-1325.json --compare $R/ov-off-1325.json ;;
    soak-on)   go $arm "" python3 $R/soak64.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 ;;
    soak-off)  go $arm "" python3 $R/soak64.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=0 --solo-json $R/soak-on.json --compare $R/soak-on.json ;;
    soak2-on)  go $arm "" python3 $R/soak64b.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 ;;
    soak2-off) go $arm "" python3 $R/soak64b.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=0 --solo-json $R/soak2-on.json --compare $R/soak2-on.json ;;
    soak2-ref) go $arm "" python3 $R/soak64b.py --exe $S/engine/$REF --config $CFG --extra "$EXACT" --dump $R/$arm.json --solo-json $R/soak2-on.json ;;
    soak2-ref2) go $arm "" python3 $R/soak64b.py --exe $S/engine/$REF --config $CFG --extra "$EXACT" --dump $R/$arm.json --solo-json $R/soak2-on.json ;;
    soak2-on2) go $arm "" python3 $R/soak64b.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 --solo-json $R/soak2-on.json ;;
    reuse-ref) go $arm "" python3 $R/reuse4.py --exe $S/engine/$REF --config $CFG --extra "$EXACT" --dump $R/$arm.json ;;
    reuse-off) go $arm "" python3 $R/reuse4.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=0 ;;
    reuse-on)  go $arm "" python3 $R/reuse4.py --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 ;;
    soak3-on)  go $arm "" python3 $R/soak64c.py --long-new --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 ;;
    soak3-off) go $arm "" python3 $R/soak64c.py --long-new --exe $S/engine/$OV --config $CFG --extra "$EXACT" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=0 --solo-json $R/soak3-on.json --compare $R/soak3-on.json ;;
    fault-a)   go $arm "" "${OVT[@]}" $F --env STRATA_PIPE_OVERLAP_TEST_FAULT=poll ;;
    fault-b)   go $arm "" "${OVT[@]}" $F --env STRATA_PIPE_OVERLAP_TEST_FAULT=throw ;;
    fault-c)   go $arm "" "${OVT[@]}" $F --env STRATA_PIPE_OVERLAP_TEST_FAULT=stage ;;
    fault-d)   go $arm "" "${OVT[@]}" $F --env STRATA_PIPE_OVERLAP_TEST_FAULT=cuda ;;
    fault-e)   go $arm "" "${OVT[@]}" $F --env STRATA_PIPE_OVERLAP_TEST_FAULT=stall --env STRATA_WATCHDOG_S=10 --env STRATA_PIPE_OVERLAP_JOIN_S=2 ;;
    *) echo "unknown arm $arm" >> $R/summary.txt ;;
  esac
done
echo "DONE $* $(date +%T)" >> $R/summary.txt
