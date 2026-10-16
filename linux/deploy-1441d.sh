#!/usr/bin/env bash
# Production: overlapped prompt read with the streamed K/V cache (TESTLOG #81), engine strata-g3-1441b-ovkv,
# config = 1441b production + STRATA_PIPE_OVERLAP=1 + STRATA_PIPE_OVERLAP_KVS=1 (split 12,25,37, --kv-resident 32768 as now).
# Validated 2026-10-16 (results/ovkv): token-identical, join pauses 7.7 s -> 0.35 s (32K), 19.3 s -> 0.29 s (100K).
# --rollback: back to 1441b (12,25,37, --kv-resident 32768).
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090
case "${1:-}" in
  "") TARGET=T16-1441d-ov-prod262r-vision-cpu.json ;;
  --rollback) TARGET=T16-1441b-prod262r-vision-cpu.json ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
echo "production.json: $(readlink $R/linux/configs/production.json) -> $TARGET"
ln -sfn "$TARGET" $R/linux/configs/production.json
systemctl --user restart strata.service; t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; exit 1; }; sleep 5; done
echo "loaded after $(( $(date +%s) - t0 )) s"
L=$R/Strata/strata-unsloth-ud-iq4_xs.log; L0=$(grep -an "engine started" $L | tail -1 | cut -d: -f1); tail -n +${L0:-1} $L | grep -aE "prompt buffers|borrows|VRAM free with everything" | cut -c1-160
P=$(systemctl --user show -p MainPID --value strata.service)
for c in $(pgrep -P $P); do tr '\0' '\n' < /proc/$c/environ 2>/dev/null | grep -E "PIPE_OVERLAP|QSA_SLOTS_STREAM" ; readlink /proc/$c/exe; done
curl -s -m 5 $URL/v1/load; echo
