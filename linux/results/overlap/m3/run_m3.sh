#!/bin/bash
# M3: tools/batch_overlap_test.py arms. Usage: run_m3.sh <name> <exe> "<test args>"   (logs: m3/<name>.*)
R=/media/raid0/StrataQ38FN/linux/results/overlap/m3
S=/media/raid0/StrataQ38FN/Strata
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-mmqtab.json
name=$1 exe=$2 targs=$3
echo "=== $name $(date +%T) exe=$exe test=[$targs]" >> $R/summary.txt
BATCH_TEST_LOG=$R/$name.engine.log systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-ov-m3-$name \
  python3 $S/tools/batch_overlap_test.py --exe $S/engine/$exe --config $CFG \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow" --dump $R/$name.json $targs > $R/$name.out 2>&1
echo "rc=$? $(date +%T)" >> $R/summary.txt
grep -E "IDENTICAL|DIFFERS|PREFIX|cancelled|^log:|compare|RESULT|^fault:|   log:" $R/$name.out >> $R/summary.txt
