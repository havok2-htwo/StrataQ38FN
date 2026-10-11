#!/usr/bin/env bash
# Config-only quick win from the code analysis: explicit split 12,24,36 + --trim-stage-weights (own prompt buffers, no loans)
cd "$(dirname "$0")"
until grep -q "SERIES Q DONE" results/series_q.out 2>/dev/null; do sleep 15; done
R=./run_strata_variant.sh
S="--rounds 2 --max-tok 512 --prompt-tok 100"
L="--rounds 2 --max-tok 512"
for n in 12 16; do
  $R strata-T$n-trim-short configs/T$n-trim-split.json 1,$n,$((n*2)) $S 2>&1 | tail -3
  $R strata-T$n-trim-long configs/T$n-trim-split.json $n $L 2>&1 | tail -1
  grep -E "loads the dense weights|own prompt buffers|borrows|expert cache [0-9]+ slots|of its 6144|zero-doorbell" results/engine-strata-T$n-trim-short.log | sort -u | cut -c1-150
done
echo SERIES T DONE
