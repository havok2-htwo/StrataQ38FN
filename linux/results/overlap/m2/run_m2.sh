#!/bin/bash
# M2 (gate + prepare, dry run): tools/batch_test16.py on the reference engine (strata-g3-mmqtab) and the M2 engine
# (strata-g3-overlap), env off / STRATA_PIPE_OVERLAP=1, plus refusal arms. Usage: run_m2.sh <arm>...
R=/media/raid0/StrataQ38FN/linux/results/overlap/m2
S=/media/raid0/StrataQ38FN/Strata
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-mmqtab.json
EXACT="--pcie-frac 0 --adapt-every 1000000"
run() {  # name exe "envs" "extra" "testargs"
  local name=$1 exe=$2 envs="STRATA_MMQ_DIRECT=1 $3" extra=$4 targs=${5:---n 16 --max-new 64 --long-tokens 1200}
  echo "=== $name $(date +%T) exe=$exe env=[$envs] extra=[$extra] test=[$targs]" >> $R/summary.txt
  env $envs BATCH_TEST_LOG=$R/$name.engine.log systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-ov-m2-$name \
    python3 $S/tools/batch_test16.py --exe $S/engine/$exe --config $CFG --batch 16 $targs \
      --extra "$EXACT $extra" --dump $R/$name.json > $R/$name.out 2>&1
  echo "rc=$? $(date +%T)" >> $R/summary.txt
  grep -E "^batch:|DIFFERS" $R/$name.out >> $R/summary.txt
  grep -cE "overlapped read eligible" $R/$name.engine.log | sed 's/^/eligible lines: /' >> $R/summary.txt
  grep -E "overlapped read (off|: captured)|not captured before" $R/$name.engine.log >> $R/summary.txt
}
cd $S
SMALL="--n 8 --max-new 32 --long-tokens 1200"
for arm in "$@"; do
  case $arm in
    ref) run ref strata-g3-mmqtab "" "--no-prefill-borrow" ;;
    off) run off strata-g3-overlap "" "--no-prefill-borrow" ;;
    on) run on strata-g3-overlap "STRATA_PIPE_OVERLAP=1" "--no-prefill-borrow" ;;
    on-borrow) run on-borrow strata-g3-overlap "STRATA_PIPE_OVERLAP=1" "" ;;
    on-noprio) run on-noprio strata-g3-overlap "STRATA_PIPE_OVERLAP=1 STRATA_PIPE_OVERLAP_PRIO=0" "--no-prefill-borrow" "$SMALL" ;;
    r-plebatch) run r-plebatch strata-g3-overlap "STRATA_PIPE_OVERLAP=1 STRATA_PLE_BATCH=0" "--no-prefill-borrow" "$SMALL" ;;
    r-kvres) run r-kvres strata-g3-overlap "STRATA_PIPE_OVERLAP=1" "--no-prefill-borrow --kv-resident 8192" "$SMALL" ;;
    r-help) run r-help strata-g3-overlap "STRATA_PIPE_OVERLAP=1 STRATA_PREFILL_HELP=1" "--no-prefill-borrow" "$SMALL" ;;
    r-ownbuf) run r-ownbuf strata-g3-overlap "STRATA_PIPE_OVERLAP=1 STRATA_SPLIT_OWN_BUFFERS=0" "" "$SMALL" ;;
    r-syncsub) run r-syncsub strata-g3-overlap "STRATA_PIPE_OVERLAP=1" "--no-prefill-borrow --ple-io direct --ple-sync-submit" "$SMALL" ;;
    r-idxchk) run r-idxchk strata-g3-overlap "STRATA_PIPE_OVERLAP=1 STRATA_IDX_FP16_CHECK=1" "--no-prefill-borrow" "$SMALL" ;;
    ref-idxchk) run ref-idxchk strata-g3-mmqtab "STRATA_IDX_FP16_CHECK=1" "--no-prefill-borrow" "$SMALL" ;;
    on-mtpprio) run on-mtpprio strata-g3-overlap "STRATA_PIPE_OVERLAP=1 STRATA_PIPE_OVERLAP_MTP_PRIO=1" "--no-prefill-borrow" ;;
    r-noar) run r-noar strata-g3-overlap "STRATA_PIPE_OVERLAP=1 STRATA_VERIFY_ALL_RESIDENT=0" "--no-prefill-borrow" "$SMALL" ;;
  esac
done
echo "DONE $* $(date +%T)" >> $R/summary.txt
