#!/bin/bash
# rebase142 GPU exactness: one arm per call.  Usage: run.sh <arm>...   (logs: <arm>.out, <arm>.engine.log, <arm>.json)
# engine/strata-g3-142 (g3-142 e20b3a18), tools from the Strata-142 worktree, config T16-mmqtab.
R=/media/raid0/StrataQ38FN/linux/results/rebase142/gpu
S=/media/raid0/StrataQ38FN/Strata
T=/media/raid0/StrataQ38FN/Strata-142/tools
EXE=$S/engine/strata-g3-142
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-mmqtab.json
EXACT="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
E1=/media/raid0/StrataQ38FN/linux/results/overlap/exactness-r1
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
go() {  # name envs cmd...
  local name=$1 envs=$2; shift 2
  for i in $(seq 90); do gpu_free && break; sleep 2; done
  gpu_free || { echo "=== $name: GPUs busy, skipped" >> $R/summary.txt; return; }
  echo "=== $name $(date +%T) env=[$envs] cmd=[$*]" >> $R/summary.txt
  local t0=$(date +%s)
  env STRATA_MMQ_DIRECT=1 STRATA_IQ_MT_MIN=1 $envs BATCH_TEST_LOG=$R/$name.engine.log \
    systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-r142-$name "$@" > $R/$name.out 2>&1
  echo "rc=$? $(date +%T) ($(( $(date +%s) - t0 )) s)" >> $R/summary.txt
  grep -E "^batch:|DIFFERS|DID NOT|^G2: |^note:|^log:|compare|RESULT|D gave|ERR|requests differ" $R/$name.out | grep -v "IDENTICAL$" >> $R/summary.txt
  echo "identical slots: $(grep -c 'IDENTICAL$' $R/$name.out); engine log: 'overlapped read of' $(grep -c 'overlapped read of' $R/$name.engine.log), 'not captured before' $(grep -c 'not captured before' $R/$name.engine.log), TAIL_SKIP lines $(grep -c 'TAIL_SKIP\|route tail skip' $R/$name.engine.log), watchdog/terminate $(grep -ciE 'watchdog|terminate called' $R/$name.engine.log)" >> $R/summary.txt
}
BT16=(python3 $T/batch_test16.py --exe $EXE --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$EXACT")
BT32=(python3 $T/batch_test32.py --exe $EXE --config $CFG --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra "$EXACT")
OVT=(python3 $T/batch_overlap_test.py --exe $EXE --config $CFG --extra "$EXACT")
for arm in "$@"; do
  case $arm in
    bt16)      go $arm "" "${BT16[@]}" --dump $R/$arm.json ;;
    bt32)      go $arm "" "${BT32[@]}" --dump $R/$arm.json ;;
    il)        go $arm "" python3 $T/batch_interleave_test.py --exe $EXE --config $CFG --extra "$EXACT" ;;
    il-sc)     go $arm "STRATA_SLOT_COPY_DIRECT=1" python3 $T/batch_interleave_test.py --exe $EXE --config $CFG --extra "$EXACT" ;;
    ov-on)     go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 ;;
    ov-off)    go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=0 --solo-json $R/ov-on.json --compare $R/ov-on.json ;;
    ov-on-sc)  go $arm "" "${OVT[@]}" --dump $R/$arm.json --env STRATA_PIPE_OVERLAP=1 --env STRATA_SLOT_COPY_DIRECT=1 --solo-json $R/ov-on.json --compare $R/ov-on.json ;;
    bt16-mmq0) go $arm "STRATA_MMQ_DIRECT=0" "${BT16[@]}" --dump $R/$arm.json ;;
    bt32-mmq0) go $arm "STRATA_MMQ_DIRECT=0" "${BT32[@]}" --dump $R/$arm.json ;;
    bt16-ts7)  go $arm "STRATA_ROUTE_TAIL_SKIP=7" "${BT16[@]}" --dump $R/$arm.json ;;
    bt16-sc)   go $arm "STRATA_SLOT_COPY_DIRECT=1" "${BT16[@]}" --dump $R/$arm.json ;;
    bt16-ts0)  go $arm "STRATA_ROUTE_TAIL_SKIP=0" "${BT16[@]}" --dump $R/$arm.json ;;
    *) echo "unknown arm $arm" >> $R/summary.txt ;;
  esac
done
echo "DONE $* $(date +%T)" >> $R/summary.txt
