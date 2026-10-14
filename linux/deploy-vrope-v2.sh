#!/usr/bin/env bash
# Switch production Strata to the per-slot M-RoPE regions: engine strata-g3-143v2 with STRATA_MROPE_SLOTS=1 (image and
# text requests share the batch) and STRATA_ADMIT_SHORT_FIRST=8192 (short prompts are admitted first), server code
# g3-143 08830ddf.  GPU-validated 2026-10-13 21:00-21:22 (linux/results/vision/v2: text A/B GO, engine mode GO).
#   deploy-vrope-v2.sh              switch (Strata is down ~2 min; conversations parked in RAM are lost)
#   deploy-vrope-v2.sh --rollback   back to k5 (T16-143-prod262r-vision-cpu.json) and server cb95419d
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PROD=/media/raid0/StrataQ38FN/Strata-143-prod; URL=http://127.0.0.1:8090
LOG=/media/raid0/StrataQ38FN/Strata/strata-unsloth-ud-iq4_xs.log
case "${1:-}" in
  "") TARGET=T16-143-prod262r-vision-cpu-v2.json; CODE=08830ddf ;;
  --rollback) TARGET=T16-143-prod262r-vision-cpu.json; CODE=cb95419d ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$HERE/configs/$TARGET"
systemctl --user is-active -q strata-bench.scope && { echo "strata-bench.scope is running - stop it first"; exit 1; }
echo "production.json: $(readlink "$HERE/configs/production.json") -> $TARGET; server code -> $CODE"
git -C "$PROD" checkout -q --detach "$CODE"
ln -sfn "$TARGET" "$HERE/configs/production.json"
systemctl --user restart strata.service
t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "strata.service died during the start"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "start timeout (15 min)"; exit 1; }
  sleep 5
done
echo "loaded after $(( $(date +%s) - t0 )) s"
L0=$(grep -an "engine started" "$LOG" | tail -1 | cut -d: -f1)
tail -n +"${L0:-1}" "$LOG" | grep -aE "image position table|WARNING" | head -5 | cut -c1-200
curl -s -m 5 $URL/v1/load; echo
