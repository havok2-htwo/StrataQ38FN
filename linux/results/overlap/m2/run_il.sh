#!/bin/bash
# M2: tools/batch_interleave_test.py (3 slots = 3 pipelined groups of 1 on the 12,24,36 split), reference vs env on
R=/media/raid0/StrataQ38FN/linux/results/overlap/m2
S=/media/raid0/StrataQ38FN/Strata
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-mmqtab.json
for arm in "$@"; do
  case $arm in
    il-ref) exe=strata-g3-mmqtab; envs="STRATA_MMQ_DIRECT=1" ;;
    il-on) exe=strata-g3-overlap; envs="STRATA_MMQ_DIRECT=1 STRATA_PIPE_OVERLAP=1" ;;
    il-off) exe=strata-g3-overlap; envs="STRATA_MMQ_DIRECT=1" ;;
  esac
  echo "=== $arm $(date +%T) exe=$exe env=[$envs]" >> $R/summary.txt
  env $envs BATCH_TEST_LOG=$R/$arm.engine.log systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-ov-m2-$arm \
    python3 $S/tools/batch_interleave_test.py --exe $S/engine/$exe --config $CFG \
      --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow" > $R/$arm.out 2>&1
  echo "rc=$? $(date +%T)" >> $R/summary.txt
  grep -cE "overlapped read eligible" $R/$arm.engine.log | sed 's/^/eligible lines: /' >> $R/summary.txt
  grep -E "overlapped read (off|: captured)|not captured before" $R/$arm.engine.log >> $R/summary.txt
done
echo "DONE $* $(date +%T)" >> $R/summary.txt
