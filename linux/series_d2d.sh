#!/usr/bin/env bash
# Direct (device-to-device) slot copy: exactness re-check, throughput and multi-turn vs the host-copy engine
cd "$(dirname "$0")"
R=./run_strata_variant.sh
S="--rounds 2 --max-tok 512 --prompt-tok 100"
L="--rounds 2 --max-tok 512"
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
( cd /media/raid0/StrataQ38FN/Strata && .venv/bin/python tools/batch_test16.py --exe engine/strata-g3b --config ../linux/configs/T16-direct.json --batch 16 --n 16 --max-new 150 --extra '--layer-split 12,24,36 --trim-stage-weights --batch-groups 4 --pcie-frac 0 --adapt-every 1000000' > ../linux/results/exact_direct_batch16_v2.txt 2>&1 )
echo "exactness: $(grep -c IDENTICAL results/exact_direct_batch16_v2.txt)/16 identical"
$R strata-T16-direct-short configs/T16-direct.json 16,24 $S 2>&1 | tail -2
$R strata-T16-direct-long configs/T16-direct.json 16 $L 2>&1 | tail -1
grep -o "copied in [0-9.]* ms" results/engine-strata-T16-direct-long.log | tail -200 | awk '{s+=$3;n++} END {printf "slot copy avg %.1f ms over %d copies\n", s/n, n}'
KEEP=1 $R strata-X16-direct-mt-warm configs/X16-agent-direct.json 1 --rounds 1 --max-tok 16 --warmup 0 > /dev/null 2>&1
$PY -I bench_multiturn.py --url http://127.0.0.1:8090 --label strata-X16-direct-mt-u16 --users 16 --turns 5 --out results/multiturn.jsonl 2>&1 | tail -9
systemctl --user stop strata-bench.scope
echo SERIES D2D DONE
