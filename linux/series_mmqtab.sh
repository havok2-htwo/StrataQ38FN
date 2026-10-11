#!/usr/bin/env bash
# B-tab (pointer-table MMQ, STRATA_MMQ_DIRECT=1) vs baseline, through the HTTP server
cd "$(dirname "$0")"
R=./run_strata_variant.sh; PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
S="--rounds 2 --max-tok 512 --prompt-tok 100"; L="--rounds 2 --max-tok 512"
$R strata-T16-mmqtab-short configs/T16-mmqtab.json 1,16,24 $S 2>&1 | tail -3
$R strata-T16-mmqtab-long configs/T16-mmqtab.json 1,16 $L 2>&1 | tail -2
f=results/engine-strata-T16-mmqtab-long.log; n=$(grep -n "engine started" $f | tail -1 | cut -d: -f1)
tail -n +$n $f | grep -E "in place|guard" | head -3 | cut -c1-160
tail -n +$n $f | grep -oE "prompt 12[0-9]{2} tokens = 0 reused \+ [0-9]+ read in [0-9]+ ms" | head -4
$R strata-T16-pleram-ref-long configs/T16-direct-pleram.json 1,16 $L 2>&1 | tail -2
for cfg in X16-agent-mmqtab X16-agent-direct; do
  KEEP=1 $R strata-$cfg-mtwarm configs/$cfg.json 1 --rounds 1 --max-tok 8 --warmup 0 >/dev/null 2>&1
  $PY -I bench_multiturn.py --url http://127.0.0.1:8090 --label strata-$cfg-mt-u16 --users 16 --turns 5 --out results/multiturn.jsonl 2>&1 | tail -8
  systemctl --user stop strata-bench.scope
done
echo SERIES MMQTAB DONE
