#!/bin/bash
# queue-depth probe: T16-142fc through the Strata-142 server, engine behind a stdin/stdout timestamp proxy
D=/media/raid0/StrataQ38FN/linux/results/batchadmit/qprobe
L=/media/raid0/StrataQ38FN/linux
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}' || { echo GPUs busy; exit 1; }
cd /media/raid0/StrataQ38FN/Strata-142
PYTHONDONTWRITEBYTECODE=1 systemd-run --user --scope -q -p MemoryMax=90G -p MemorySwapMax=0 --unit=strata-ba-qprobe \
  $PY serve/server.py --engine strata --config $D/T16-142fc-qprobe.json --port 8091 > $D/server.out 2>&1 < /dev/null &
sleep 5; t0=$(date +%s)
until curl -s -m 3 http://127.0.0.1:8091/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata-ba-qprobe.scope || { echo died; tail -20 $D/server.out; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo timeout; break; }
  sleep 5
done
echo "ready $(( $(date +%s) - t0 )) s"
for arm in "short2 --rounds 2 --prompt-tok 100" "long2 --rounds 2" "short4 --rounds 4 --prompt-tok 100" "long4 --rounds 4"; do
  set -- $arm; name=$1; shift
  echo "$(python3 -c 'import time;print(f"{time.monotonic():.4f}")') ARM $name" >> $D/proto.log
  $PY -I $L/bench_concurrency.py --url http://127.0.0.1:8091 --label qprobe-$name --levels 16 --max-tok 512 \
     --temperature 0 --warmup 0 --out $D/sweep.jsonl "$@" 2>&1 | tail -1 | tee $D/$name.txt
done
systemctl --user stop strata-ba-qprobe.scope
echo done
