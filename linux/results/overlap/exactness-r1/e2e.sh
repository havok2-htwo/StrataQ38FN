#!/bin/bash
# exactness-r1 item 6: serve/server.py with T16-mmqtab + STRATA_PIPE_OVERLAP=1 (engine/strata-g3-overlap), bench_concurrency
R=/media/raid0/StrataQ38FN/linux/results/overlap/exactness-r1
S=/media/raid0/StrataQ38FN/Strata
for i in $(seq 60); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}' && break; sleep 2; done
echo "=== e2e $(date +%T)" >> $R/summary.txt
: > $R/e2e.engine.log
cd $S
systemd-run --user --scope -q -p MemoryMax=90G -p MemorySwapMax=0 --unit=strata-ov-x1-e2e \
  $S/.venv/bin/python $S/serve/server.py --engine strata --config $R/T16-overlap-e2e.json --port 8097 > $R/e2e.server.out 2>&1 &
for i in $(seq 300); do
  curl -s -m 2 http://127.0.0.1:8097/health | grep -q '"loaded": true' && break; sleep 2
done
curl -s -m 2 http://127.0.0.1:8097/health >> $R/e2e.bench.out; echo >> $R/e2e.bench.out
python3 /media/raid0/StrataQ38FN/linux/bench_concurrency.py --url http://127.0.0.1:8097 --levels 4,16 --rounds 2 \
  --max-tok 256 --prompt-tok 1000 --label ov-e2e --out $R/e2e.bench.jsonl >> $R/e2e.bench.out 2>&1
echo "bench rc=$?" >> $R/e2e.bench.out
python3 /media/raid0/StrataQ38FN/linux/bench_concurrency.py --url http://127.0.0.1:8097 --levels 16 --rounds 1 \
  --max-tok 128 --prompt-tok 4000 --label ov-e2e-4k --out $R/e2e.bench.jsonl >> $R/e2e.bench.out 2>&1
echo "bench4k rc=$?" >> $R/e2e.bench.out
systemctl --user stop strata-ov-x1-e2e.scope
sleep 3
echo "rc done $(date +%T)" >> $R/summary.txt
tail -30 $R/e2e.bench.out >> $R/summary.txt
for L in $R/e2e.engine.log $R/e2e.server.out; do
  echo "$L: 'overlapped read of' $(grep -c 'strata batch: overlapped read of' $L), ERR $(grep -c '^ERR\|ERR ' $L), watchdog $(grep -ciE 'watchdog|issue #29' $L), terminate $(grep -c 'terminate called' $L), 'not captured before' $(grep -c 'not captured before' $L), 'overlapped read off' $(grep -c 'overlapped read off' $L)" >> $R/summary.txt
done
echo "E2E-DONE" >> $R/summary.txt
