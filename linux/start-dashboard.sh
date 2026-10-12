#!/usr/bin/env bash
# Start the LAN dashboard (dashboard/dashboard.py) on Linux in its own systemd user scope.
# It runs with or without a Strata server: GPUs (all 4, incl. PCIe), CPU, RAM and disks are always live; the engine
# panels (live requests, answering progress, batch slots, speeds) fill in when a Strata server answers GET /metrics.
#   ./start-dashboard.sh                       -> http://<host>:8082/  (e.g. http://192.168.20.126:8082/)
#   PORT=8083 ./start-dashboard.sh
#   SOURCES="http://127.0.0.1:8090 http://127.0.0.1:8005" ./start-dashboard.sh   (asked in this order)
#   API_KEY_FILE=/path/key.txt ./start-dashboard.sh     (key for the sources; it stays on this machine)
#   FG=1 ./start-dashboard.sh                  (stay in the foreground instead of logging to dashboard/dashboard-linux.out)
# Stop: systemctl --user stop strata-dashboard.scope
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PORT=${PORT:-8082}
HOST=${HOST:-0.0.0.0}
UNIT=${UNIT:-strata-dashboard}
SOURCES=${SOURCES:-"http://127.0.0.1:8090 http://127.0.0.1:8080"}
STRATA_SERVE=${STRATA_SERVE:-$ROOT/Strata-142/serve}        # telemetry.py for the GPU readings (else an own reader)
CONFIG=${CONFIG:-$HERE/configs/T16-142gsc.json}             # its "log" = engine log when no running server names one

# Python with psutil: $PYTHON, else the system python3, else Strata's venv
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in /usr/bin/python3 "$ROOT/Strata/.venv/bin/python" python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import psutil" 2>/dev/null; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "no python3 with psutil found (set PYTHON=...)" >&2; exit 1; }

if ss -Hltn "sport = :$PORT" | grep -q .; then
  echo "port $PORT is already in use:" >&2; ss -Hltnp "sport = :$PORT" >&2; exit 1
fi
if systemctl --user is-active --quiet "$UNIT.scope"; then
  echo "$UNIT.scope is already running (systemctl --user stop $UNIT.scope)" >&2; exit 1
fi

ARGS=(--port "$PORT" --host "$HOST" --strata "$STRATA_SERVE" --engine-config "$CONFIG"
      --state "$ROOT/dashboard/state-linux.json" --settings "$ROOT/dashboard/settings-linux.json")
for s in $SOURCES; do
  if [ -n "${API_KEY_FILE:-}" ]; then ARGS+=(--source "$s,$API_KEY_FILE"); else ARGS+=(--source "$s"); fi
done

cd "$ROOT/dashboard"
CMD=(systemd-run --user --scope --unit="$UNIT" -p MemoryMax=1G "$PY" dashboard.py "${ARGS[@]}")
if [ -n "${FG:-}" ]; then
  exec "${CMD[@]}"
fi
nohup "${CMD[@]}" > "$ROOT/dashboard/dashboard-linux.out" 2>&1 &
sleep 3
if systemctl --user is-active --quiet "$UNIT.scope"; then
  echo "dashboard on port $PORT: http://<host>:$PORT/ (this host: $(hostname -I | xargs)); stop: systemctl --user stop $UNIT.scope; log: dashboard/dashboard-linux.out"
else
  echo "dashboard did not start:" >&2; cat "$ROOT/dashboard/dashboard-linux.out" >&2; exit 1
fi
