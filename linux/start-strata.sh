#!/usr/bin/env bash
# Start Strata with the recommended 4x RTX 3090 config (UD-IQ4_XS, 12 batch slots) in its own systemd scope.
#   ./start-strata.sh                      -> http://127.0.0.1:8090/v1
#   PORT=8005 ./start-strata.sh
#   HOST=0.0.0.0 API_KEY=<secret> ./start-strata.sh   (LAN only together with a key)
#   CONFIG=configs/K-q4kxl-b12.json ./start-strata.sh  (another variant)
# Stop: systemctl --user stop strata.scope
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STRATA=/media/raid0/StrataQ38FN/Strata
CONFIG=$(readlink -f "${CONFIG:-$HERE/configs/G12-pf512-share2.json}")
PORT=${PORT:-8090}
EXTRA=()
[ -n "${HOST:-}" ] && EXTRA+=(--host "$HOST")
[ -n "${API_KEY:-}" ] && EXTRA+=(--api-key "$API_KEY")
cd "$STRATA"
exec systemd-run --user --scope --unit=strata -p MemoryMax=${MEMMAX:-90G} -p MemorySwapMax=0 \
  "$STRATA/.venv/bin/python" "$STRATA/serve/server.py" --engine strata --config "$CONFIG" --port "$PORT" "${EXTRA[@]}"
