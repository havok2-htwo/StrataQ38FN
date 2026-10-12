#!/bin/bash
# exactness gates of engine/strata-g3-143k4 (STRATA_CONV_TIER_*, stage 3) on the default config (T16-143-64khocas)
# against the stage-1 reference dumps (engine 143ho == 143kt == 143k3).  run_exact.sh <arm>:<cfg> ...
#   arms: bt16 t07 il ov1 ; cfg: off | ton | tfv (json in this dir)
D=/media/raid0/StrataQ38FN/linux/results/kvtier/stage3; R=/media/raid0/StrataQ38FN/linux/results/kvtier/stage1
T=/media/raid0/StrataQ38FN/Strata-143/tools
CMP=/media/raid0/StrataQ38FN/linux/results/rebase143/exact/cmp.py
X="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 --layer-split 12,24,37"
for spec in "$@"; do
  arm=${spec%%:*}; cfg=${spec#*:}; C=$D/$cfg.json; N=$arm-$cfg
  E=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" $C)
  case $arm in
  bt16) $D/arm.sh $D $N $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$X" --dump $D/$N.json ;;
  t07)  $D/arm.sh $D $N $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 1200 --keys "temperature=0.7 top_k=20 seed=1234" --extra "$X" --dump $D/$N.json ;;
  il)   $D/arm.sh $D $N $C python3 $T/batch_interleave_test.py --exe $E --config $C --extra "$X" --dump $D/$N.json ;;
  ov1)  $D/arm.sh $D $N $C python3 $T/batch_overlap_test.py --exe $E --config $C --extra "$X" --env STRATA_PIPE_OVERLAP=1 --solo-json /media/raid0/StrataQ38FN/linux/results/levers/handoff/ov1.json --compare /media/raid0/StrataQ38FN/linux/results/levers/handoff/ov1.json --dump $D/$N.json ;;
  *) echo "unknown $arm"; continue ;;
  esac
  echo "--- $N vs stage1 $arm-ref"
  [ -f $D/$N.json ] && [ -f $R/$arm-ref.json ] && python3 $CMP $D/$N.json $R/$arm-ref.json
  echo "tier lines: $(grep -c 'conversation tier:' $D/$N.engine.log)  spills: $(grep -c 'tier: spilled' $D/$N.engine.log)  disk restores: $(grep -c 'restored .* from disk' $D/$N.engine.log)  queue restores: $(grep -c 'from the write queue' $D/$N.engine.log)  misses: $(grep -c 'tier: miss' $D/$N.engine.log)"
  sleep 5
done
echo EXACT DONE
