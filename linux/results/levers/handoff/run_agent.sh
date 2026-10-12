#!/bin/bash
# agent-config gate (KV streaming --kv-resident, prompt cache every 8192 -> checkpoint parts via on_stage_chunk, split
# 12,25,37): batch_test16 with a ~9000-token last prompt, reference binary 143l5 vs 143ho + STRATA_HANDOFF_PIPE=8
D=/media/raid0/StrataQ38FN/linux/results/levers/handoff; T=/media/raid0/StrataQ38FN/Strata-143/tools
CMP=/media/raid0/StrataQ38FN/linux/results/rebase143/exact/cmp.py
X="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 --layer-split 12,25,37"
for who in "$@"; do
  C=$D/X16-$who.json; E=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exe'])" $C)
  $D/arm.sh $D ag_$who $C python3 $T/batch_test16.py --exe $E --config $C --batch 16 --n 16 --max-new 64 --long-tokens 9000 --extra "$X" --dump $D/ag_$who.json
  sleep 5
done
[ -f $D/ag_ref.json ] && [ -f $D/ag_ho.json ] && python3 $CMP $D/ag_ho.json $D/ag_ref.json
