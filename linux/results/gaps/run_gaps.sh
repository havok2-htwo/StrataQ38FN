#!/bin/bash
# Remaining exactness gaps from the 2026-10-12 review, reference (T16-142fcb / strata-g3-142fc) vs candidate
# (T16-142gsc / strata-g3-142lv: STRATA_MMQ_GROUP_SORT + STRATA_BATCH_SELF_COMMIT).
G=/media/raid0/StrataQ38FN/linux/results/gaps; C=/media/raid0/StrataQ38FN/linux/configs; T=/media/raid0/StrataQ38FN/Strata-142/tools
E=/media/raid0/StrataQ38FN/Strata/engine
X="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 --layer-split 12,24,37"
XA="--pcie-frac 0 --adapt-every 1000000 --aux-cpus 65-127 --layer-split 12,24,37"   # agent: production loans visible
for arm in ref:142fc:T16-142fcb:X16-agent-142fcb cand:142lv:T16-142gsc:X16-agent-142gsc; do
  IFS=: read n b cfg acfg <<< "$arm"
  $G/arm.sh $G/interleave il_final_$n $C/$cfg.json python3 $T/batch_interleave_test.py --exe $E/strata-g3-$b --config $C/$cfg.json --extra "$X" --dump $G/interleave/il-final-$n.json
  $G/arm.sh $G/sampling bt16_t07_$n $C/$cfg.json python3 $T/batch_test16.py --exe $E/strata-g3-$b --config $C/$cfg.json --batch 16 --n 16 --max-new 64 --long-tokens 1200 --keys "temperature=0.7 top_k=20 seed=1234" --extra "$X" --dump $G/sampling/bt16-t07-$n.json
  $G/arm.sh $G/nextturn reuse5_$n $C/$cfg.json python3 $G/nextturn/reuse5.py --exe $E/strata-g3-$b --config $C/$cfg.json --extra "$X" --dump $G/nextturn/reuse5-$n.json
  $G/arm.sh $G/agent bt16_agent_$n $C/$acfg.json python3 $T/batch_test16.py --exe $E/strata-g3-$b --config $C/$acfg.json --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra "$XA" --dump $G/agent/bt16-agent-$n.json
done
echo GAPS DONE
