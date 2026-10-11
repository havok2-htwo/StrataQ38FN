#!/usr/bin/env bash
# Step 1: more slots / bigger prefill chunks on the trim+split base; step 2: multi-turn agent workload
cd "$(dirname "$0")"
R=./run_strata_variant.sh
S="--rounds 2 --max-tok 512 --prompt-tok 100"
L="--rounds 2 --max-tok 512"
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
for n in 20 24; do
  $R strata-T$n-trim-short configs/T$n-trim-split.json $n $S 2>&1 | tail -1
  $R strata-T$n-trim-long configs/T$n-trim-split.json $n $L 2>&1 | tail -1
  n0=$(grep -n "engine started" results/engine-strata-T$n-trim-long.log | tail -1 | cut -d: -f1)
  tail -n +$n0 results/engine-strata-T$n-trim-long.log | grep -E "runs layers .* expert cache|^strata generate: expert cache [0-9]|borrows|own prompt buffers" | cut -c1-130
done
for ch in 1024 2048; do
  $R strata-T16-pf$ch-long configs/T16-trim-pf$ch.json 16 $L 2>&1 | tail -1
  $R strata-T16-pf$ch-short configs/T16-trim-pf$ch.json 16 $S 2>&1 | tail -1
done
# step 2: multi-turn
for cfg in T16-trim-split X16-agent-trim-262k; do
  KEEP=1 $R strata-$cfg-mt-warm configs/$cfg.json 1 --rounds 1 --max-tok 16 --warmup 0 > /dev/null 2>&1
  curl -s -m 5 http://127.0.0.1:8090/health | grep -q '"loaded": *true' || { echo "$cfg not ready"; continue; }
  for u in 8 16; do
    $PY -I bench_multiturn.py --url http://127.0.0.1:8090 --label strata-$cfg-mt-u$u --users $u --turns 5 --out results/multiturn.jsonl 2>&1 | tail -9
  done
  systemctl --user stop strata-bench.scope
done
echo SERIES R DONE
