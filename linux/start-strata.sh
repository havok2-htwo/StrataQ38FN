#!/usr/bin/env bash
# Start Strata with the recommended 4x RTX 3090 config (UD-IQ4_XS, 16 batch slots, split 12/24/36 + trim, PLE table in RAM, pointer-table MMQ; engine 0.1.43 + our patches, strata-g3-143kt built in Strata-143/build-143 with -DSTRATA_MMQ_TAB=ON) in its own systemd scope.
#   ./start-strata.sh                      -> http://127.0.0.1:8090/v1
#   PORT=8005 ./start-strata.sh
#   HOST=0.0.0.0 API_KEY=<secret> ./start-strata.sh   (LAN only together with a key)
#   CONFIG=configs/K-q4kxl-b12.json ./start-strata.sh  (another variant)
# Stop: systemctl --user stop strata.scope
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STRATA=/media/raid0/StrataQ38FN/Strata
SERVER_DIR=${SERVER_DIR:-/media/raid0/StrataQ38FN/Strata-143}   # server.py matching the engine (0.1.43)
CONFIG=$(readlink -f "${CONFIG:-$HERE/configs/T16-143-64khoca.json}")
PORT=${PORT:-8090}
EXTRA=()
[ -n "${HOST:-}" ] && EXTRA+=(--host "$HOST")
[ -n "${API_KEY:-}" ] && EXTRA+=(--api-key "$API_KEY")
cd "$STRATA"
exec systemd-run --user --scope --unit=strata -p MemoryMax=${MEMMAX:-200G} -p MemorySwapMax=0 \
  "$STRATA/.venv/bin/python" "$SERVER_DIR/serve/server.py" --engine strata --config "$CONFIG" --port "$PORT" "${EXTRA[@]}"
