#!/bin/bash
# rebase142 HTTP sanity: Strata-142/serve/server.py + engine/strata-g3-142 (T16-mmqtab-142.json, STRATA_MMQ_DIRECT=1), port 8091
R=/media/raid0/StrataQ38FN/linux/results/rebase142
W=/media/raid0/StrataQ38FN/Strata-142
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python
PORT=8091
: > $R/http.engine.log
cd $W
systemd-run --user --scope -q -p MemoryMax=90G -p MemorySwapMax=0 --unit=strata-r142-http \
  $PY $W/serve/server.py --engine strata --config $R/T16-mmqtab-142.json --port $PORT > $R/http.server.out 2>&1 < /dev/null &
t0=$(date +%s)
until curl -s -m 3 http://127.0.0.1:$PORT/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata-r142-http.scope || { echo "server died during start"; tail -30 $R/http.server.out; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo "start timeout"; exit 1; }
  sleep 5
done
echo "ready after $(( $(date +%s) - t0 )) s"
curl -s http://127.0.0.1:$PORT/health > $R/http.health.json; curl -s http://127.0.0.1:$PORT/v1/status > $R/http.status.json
$PY -I /media/raid0/StrataQ38FN/linux/bench_concurrency.py --url http://127.0.0.1:$PORT --levels 1,16 --rounds 2 --max-tok 512 \
  --prompt-tok 100 --out $R/sweep.jsonl --label strata-142-short 2>&1 | tee $R/strata-142-short.txt
echo "short rc=${PIPESTATUS[0]}"
$PY -I /media/raid0/StrataQ38FN/linux/bench_concurrency.py --url http://127.0.0.1:$PORT --levels 1,16 --rounds 2 --max-tok 512 \
  --out $R/sweep.jsonl --label strata-142-long 2>&1 | tee $R/strata-142-long.txt
echo "long rc=${PIPESTATUS[0]}"
systemctl --user stop strata-r142-http.scope
echo HTTP DONE
