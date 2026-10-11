#!/usr/bin/env bash
# Engine 0.1.42 + our series (strata-g3-142, Strata-142/serve/server.py): base vs overlap vs overlap+MAX_QUEUE=0,
# 3 repetitions, short + 1.2K prompts; then 24 / 32 slots once.
cd "$(dirname "$0")"
export SERVER_DIR=/media/raid0/StrataQ38FN/Strata-142
R=./run_strata_variant.sh
S="--rounds 2 --max-tok 512 --prompt-tok 100"; L="--rounds 2 --max-tok 512"
for rep in 1 2 3; do for cfg in ${ARMS:-T16-142 T16-142-overlap T16-142-ovq0}; do
  $R strata-$cfg-short-r$rep configs/$cfg.json 1,8,16,24 $S 2>&1 | tail -5
  $R strata-$cfg-long-r$rep configs/$cfg.json 1,8,16,24 $L 2>&1 | tail -5
done; done
$R strata-T24-142-short configs/T24-142.json 24,32 $S 2>&1 | tail -3
$R strata-T24-142-long configs/T24-142.json 24,32 $L 2>&1 | tail -3
$R strata-T32-142-short configs/T32-142.json 32 $S 2>&1 | tail -2
$R strata-T32-142-long configs/T32-142.json 32 $L 2>&1 | tail -2
echo SERIES 142 DONE
