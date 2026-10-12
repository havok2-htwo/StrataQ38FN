#!/bin/bash
# GPU exactness of engine/strata-g3-143 (Strata-143 g3-143 @42740f84) with T16-143gsc.json (= T16-142gsc, exe -> 143)
D=/media/raid0/StrataQ38FN/linux/results/rebase143/exact; C=$D/T16-143gsc.json; T=/media/raid0/StrataQ38FN/Strata-143/tools
E=/media/raid0/StrataQ38FN/Strata/engine/strata-g3-143
X="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 --layer-split 12,24,37"
for arm in "$@"; do case $arm in
  bt16) $D/arm.sh $D bt16 $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$X" --dump $D/bt16.json ;;
  bt32) $D/arm.sh $D bt32 $C python3 $T/batch_test32.py --exe $E --config $C --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra "$X" --dump $D/bt32.json ;;
  il)   $D/arm.sh $D il $C python3 $T/batch_interleave_test.py --exe $E --config $C --extra "$X" --dump $D/il.json ;;
  ov1)  $D/arm.sh $D ov1 $C python3 $T/batch_overlap_test.py --exe $E --config $C --extra "$X" --env STRATA_PIPE_OVERLAP=1 --dump $D/ov1.json ;;
  ov0)  $D/arm.sh $D ov0 $C python3 $T/batch_overlap_test.py --exe $E --config $C --extra "$X" --env STRATA_PIPE_OVERLAP=0 --dump $D/ov0.json --compare $D/ov1.json ;;
  t07)  $D/arm.sh $D t07 $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 1200 --keys "temperature=0.7 top_k=20 seed=1234" --extra "$X" --dump $D/t07.json ;;
  mmqtab) $D/arm.sh $D mmqtab $C env CUDA_VISIBLE_DEVICES=0 /media/raid0/StrataQ38FN/Strata-143/build-143/prefill_mmq_tab_test ;;
  *) echo "unknown $arm" ;;
esac; sleep 5; done
echo EXACT DONE
