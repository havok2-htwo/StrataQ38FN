#!/bin/bash
# R1 (review fixes): tools/batch_overlap_test.py arms. Usage: run_m3.sh <name> <exe> "<test args>"   (logs: r1/<name>.*)
R=/media/raid0/StrataQ38FN/linux/results/overlap/r1
S=/media/raid0/StrataQ38FN/Strata
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-mmqtab.json
name=$1 exe=$2 targs=$3
echo "=== $name $(date +%T) exe=$exe test=[$targs]" >> $R/summary.txt
BATCH_TEST_LOG=$R/$name.engine.log systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-ov-r1-$name \
  python3 $S/tools/batch_overlap_test.py --exe $S/engine/$exe --config $CFG \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow" --dump $R/$name.json $targs > $R/$name.out 2>&1
echo "rc=$? $(date +%T)" >> $R/summary.txt
grep -E "^G2: |DID NOT|^note:|IDENTICAL|DIFFERS|PREFIX|cancelled|^log:|compare|RESULT|^fault:|   log:" $R/$name.out >> $R/summary.txt
