#!/usr/bin/env bash
# Long-context candidates vs production: 16 users short/1.2K (greedy) + one ~170K-token request (if the context fits).
cd "$(dirname "$0")"; export SERVER_DIR=/media/raid0/StrataQ38FN/Strata-143
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
for cfg in ${ARMS:-T16-143-64khocast T16-143-L262k32 T16-143-L262k48 T16-143-L128k64}; do
  echo "=== $cfg"
  KEEP=1 ./run_strata_variant.sh lc-$cfg-short configs/$cfg.json 16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 2>&1 | grep -E "ready|died|timeout|^ +16 " 
  systemctl --user is-active -q strata-bench.scope || { echo "server not running"; continue; }
  f=results/engine-lc-$cfg-short.run.log; grep -hE "borrows|keeps its own|100% of the experts|KV streaming" $f 2>/dev/null | head -6 | cut -c1-140
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '; echo
  $PY -I bench_concurrency.py --url http://127.0.0.1:8090 --label lc-$cfg-long --levels 16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0 --out results/sweep.jsonl 2>&1 | grep -E "^ +16 "
  case $cfg in *L262*) $PY -I bench_concurrency.py --url http://127.0.0.1:8090 --label lc-$cfg-170k --levels 1 --rounds 1 --max-tok 256 --prompt-tok 170000 --temperature 0 --warmup 0 --timeout 1800 --out results/sweep.jsonl 2>&1 | grep -E "^ +1 |users";; esac
  case $cfg in *L128*) $PY -I bench_concurrency.py --url http://127.0.0.1:8090 --label lc-$cfg-120k --levels 1 --rounds 1 --max-tok 256 --prompt-tok 120000 --temperature 0 --warmup 0 --timeout 1800 --out results/sweep.jsonl 2>&1 | grep -E "^ +1 |users";; esac
  systemctl --user stop strata-bench.scope; sleep 5
done
echo LC DONE
