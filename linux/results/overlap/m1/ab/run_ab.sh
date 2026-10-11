#!/bin/bash
# M1 A/B: tools/batch_test16.py dumps of the reference engine (strata-g3-mmqtab) and the M1 engine (strata-g3-overlap),
# env off and STRATA_PIPE_OVERLAP=1 (only the PLE setters exist in M1), production PLE mode (ram) and --ple-io direct.
R=/media/raid0/StrataQ38FN/linux/results/overlap/m1/ab
S=/media/raid0/StrataQ38FN/Strata
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-mmqtab.json
BASE="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
run() {  # name exe envon extra
  local name=$1 exe=$2 envon=$3 extra=$4
  local envs="STRATA_MMQ_DIRECT=1"
  [ "$envon" = 1 ] && envs="$envs STRATA_PIPE_OVERLAP=1"
  echo "=== $name $(date +%T) exe=$exe env=[$envs] extra=[$extra]" >> $R/summary.txt
  env $envs BATCH_TEST_LOG=$R/$name.engine.log systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-ov-m1-$name \
    python3 $S/tools/batch_test16.py --exe $S/engine/$exe --config $CFG --batch 16 --n 16 --max-new 64 \
      --long-tokens 1200 --extra "$BASE $extra" --dump $R/$name.json > $R/$name.out 2>&1
  echo "rc=$? $(date +%T)" >> $R/summary.txt
  grep -E "^batch:|DIFFERS" $R/$name.out >> $R/summary.txt
}
cd $S
run ref-ram strata-g3-mmqtab 0 ""
run off-ram strata-g3-overlap 0 ""
run on-ram strata-g3-overlap 1 ""
run ref-direct strata-g3-mmqtab 0 "--ple-io direct"
run on-direct strata-g3-overlap 1 "--ple-io direct"
echo ALLDONE >> $R/summary.txt
