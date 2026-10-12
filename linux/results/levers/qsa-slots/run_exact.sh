#!/bin/bash
# exactness gates of engine/strata-g3-143l5 with STRATA_QSA_SLOTS=1 (T16-143l5.json = T16-143gsc + the switch, exe -> 143l5)
# against the reference dumps of engine/strata-g3-143 in rebase143/exact
D=/media/raid0/StrataQ38FN/linux/results/levers/qsa-slots; C=$D/T16-143l5.json; T=/media/raid0/StrataQ38FN/Strata-143/tools
R=/media/raid0/StrataQ38FN/linux/results/rebase143/exact
E=/media/raid0/StrataQ38FN/Strata/engine/strata-g3-143l5
X="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 --layer-split 12,24,37"
for arm in "$@"; do case $arm in
  bt16) $D/arm.sh $D bt16 $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$X" --dump $D/bt16.json ;;
  bt32) $D/arm.sh $D bt32 $C python3 $T/batch_test32.py --exe $E --config $C --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra "$X" --dump $D/bt32.json ;;
  il)   $D/arm.sh $D il $C python3 $T/batch_interleave_test.py --exe $E --config $C --extra "$X" --dump $D/il.json ;;
  ov1)  $D/arm.sh $D ov1 $C python3 $T/batch_overlap_test.py --exe $E --config $C --extra "$X" --env STRATA_PIPE_OVERLAP=1 --dump $D/ov1.json --compare $R/ov1.json ;;
  t07)  $D/arm.sh $D t07 $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 1200 --keys "temperature=0.7 top_k=20 seed=1234" --extra "$X" --dump $D/t07.json ;;
  *) echo "unknown $arm" ;;
esac; [ -f $D/$arm.json ] && python3 $R/cmp.py $D/$arm.json $R/$arm.json; sleep 5; done
echo EXACT DONE
