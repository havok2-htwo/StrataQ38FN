#!/usr/bin/env bash
# Start Strata with one config variant in its own memory-capped systemd scope, wait until it serves, run the
# concurrency sweep against it, stop it.
#   run_strata_variant.sh <label> <config.json> <levels> [extra bench args...]
set -uo pipefail
LABEL=$1; CFG=$(readlink -f "$2"); LEVELS=$3; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
STRATA=/media/raid0/StrataQ38FN/Strata
PORT=${PORT:-8090}
LOG="$HERE/results/server-$LABEL.out"
mkdir -p "$HERE/results"
systemctl --user stop strata-bench.scope 2>/dev/null
systemctl --user reset-failed strata-bench.scope 2>/dev/null
cd "$STRATA"
systemd-run --user --scope --unit=strata-bench -p MemoryMax=${MEMMAX:-90G} -p MemorySwapMax=0 \
  "$STRATA/.venv/bin/python" "$STRATA/serve/server.py" --engine strata --config "$CFG" --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
sleep 5
t0=$(date +%s)
until curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"loaded": *true'; do
  if ! systemctl --user is-active -q strata-bench.scope; then echo "[$LABEL] server died during start"; tail -30 "$LOG"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 1200 ]; then echo "[$LABEL] start timeout"; exit 1; fi
  sleep 5
done
echo "[$LABEL] ready after $(( $(date +%s) - t0 )) s"
curl -s "http://127.0.0.1:$PORT/v1/status" > "$HERE/results/status-$LABEL.json"
"$STRATA/.venv/bin/python" -I "$HERE/bench_concurrency.py" --url "http://127.0.0.1:$PORT" --label "$LABEL" \
  --levels "$LEVELS" --out "$HERE/results/sweep.jsonl" "$@" 2>&1 | tee "$HERE/results/$LABEL.txt"
cp "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['log'])" "$CFG")" "$HERE/results/engine-$LABEL.log" 2>/dev/null
[ "${KEEP:-0}" = "1" ] || systemctl --user stop strata-bench.scope
