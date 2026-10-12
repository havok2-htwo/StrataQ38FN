#!/bin/bash
# exactness gates of engine/strata-g3-143ho with STRATA_HANDOFF_PIPE=8 (T16-143ho.json = qsa-slots/T16-143l5.json + the
# switch, exe -> 143ho) against the dumps of engine/strata-g3-143l5 in levers/qsa-slots; *off = the same binary, switch unset
D=/media/raid0/StrataQ38FN/linux/results/levers/handoff; T=/media/raid0/StrataQ38FN/Strata-143/tools
R=/media/raid0/StrataQ38FN/linux/results/levers/qsa-slots
CMP=/media/raid0/StrataQ38FN/linux/results/rebase143/exact/cmp.py
E=/media/raid0/StrataQ38FN/Strata/engine/strata-g3-143ho
X="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 --layer-split 12,24,37"
for arm in "$@"; do
  C=$D/T16-143ho.json; ref=$arm
  case $arm in *off) C=$D/T16-143ho-off.json; ref=${arm%off} ;; esac
  case $arm in
  bt16|bt16off) $D/arm.sh $D $arm $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$X" --dump $D/$arm.json ;;
  bt32) $D/arm.sh $D $arm $C python3 $T/batch_test32.py --exe $E --config $C --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra "$X" --dump $D/$arm.json ;;
  il)   $D/arm.sh $D $arm $C python3 $T/batch_interleave_test.py --exe $E --config $C --extra "$X" --dump $D/$arm.json ;;
  ov1)  $D/arm.sh $D $arm $C python3 $T/batch_overlap_test.py --exe $E --config $C --extra "$X" --env STRATA_PIPE_OVERLAP=1 --dump $D/$arm.json --compare $R/ov1.json ;;
  ovf)  $D/arm.sh $D $arm $C python3 $T/batch_overlap_test.py --exe $E --config $C --extra "$X" --env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_TEST_FAULT=stage --expect-fault --solo-json $R/ov1.json ;;
  t07)  $D/arm.sh $D $arm $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 1200 --keys "temperature=0.7 top_k=20 seed=1234" --extra "$X" --dump $D/$arm.json ;;
  *) echo "unknown $arm"; continue ;;
  esac
  [ -f $D/$arm.json ] && python3 $CMP $D/$arm.json $R/$ref.json; sleep 5
done
echo EXACT DONE
