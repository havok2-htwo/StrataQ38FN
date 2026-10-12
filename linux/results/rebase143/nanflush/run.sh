#!/usr/bin/env bash
# run.sh <old|new>: start the server with cfg-<name>.json in its own scope, run client.py, stop it
D=/media/raid0/StrataQ38FN/linux/results/rebase143/nanflush
N=$1; U=strata-r143-nan$N
cd /media/raid0/StrataQ38FN/Strata-143
systemd-run --user --scope -q -p MemoryMax=90G -p MemorySwapMax=0 --unit=$U \
  /media/raid0/StrataQ38FN/Strata/.venv/bin/python serve/server.py --engine strata --config $D/cfg-$N.json --port 8090 --host 127.0.0.1 \
  > $D/server-$N.out 2>&1 < /dev/null &
t0=$(date +%s)
until curl -s -m 3 http://127.0.0.1:8090/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q $U.scope || { echo died; tail -20 $D/server-$N.out; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo timeout; systemctl --user stop $U.scope; exit 1; }
  sleep 5
done
echo ready $(( $(date +%s) - t0 ))s
/media/raid0/StrataQ38FN/Strata/.venv/bin/python -I $D/client.py > $D/client-$N.out 2>&1
systemctl --user stop $U.scope
echo done
