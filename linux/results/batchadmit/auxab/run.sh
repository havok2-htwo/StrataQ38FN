#!/bin/bash
# Track H through the server: T16-142fc + --aux-cpus 64-127 (prefill stage threads off the host's CPU 0), ab_quick arms
A=/media/raid0/StrataQ38FN/linux/results/batchadmit/auxab
L=/media/raid0/StrataQ38FN/linux
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}' || { echo GPUs busy; exit 1; }
cd /media/raid0/StrataQ38FN/Strata-142
PYTHONDONTWRITEBYTECODE=1 systemd-run --user --scope -q -p MemoryMax=90G -p MemorySwapMax=0 --unit=strata-ba-auxab \
  $PY serve/server.py --engine strata --config $A/T16-142fc-aux.json --port 8091 > $A/server.out 2>&1 < /dev/null &
sleep 5; t0=$(date +%s)
until curl -s -m 3 http://127.0.0.1:8091/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata-ba-auxab.scope || { echo died; tail -20 $A/server.out; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo timeout; break; }
  sleep 5
done
echo "ready $(( $(date +%s) - t0 )) s"
$PY -I $L/bench_concurrency.py --url http://127.0.0.1:8091 --label ba-aux-short --levels 16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 --out $A/sweep.jsonl 2>&1 | tail -1 | tee $A/short.txt
$PY -I $L/bench_concurrency.py --url http://127.0.0.1:8091 --label ba-aux-long --levels 16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0 --out $A/sweep.jsonl 2>&1 | tail -1 | tee $A/long.txt
systemctl --user stop strata-ba-auxab.scope
echo done
