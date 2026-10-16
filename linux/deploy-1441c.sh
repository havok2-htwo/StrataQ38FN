#!/usr/bin/env bash
# Production: stage rebalance 11,24,37 (TESTLOG #79), engine strata-g3-1441c (= 1441b + opt-in round-2 switches, off),
# --kv-resident 28672, STRATA_SPLIT_OWN_RESERVE_MIB=768 (no loan on any card; CUDA1 ~0.65 GiB free once loaded).
# Same env as 1441b otherwise.  Validated 2026-10-16 (results/split3): exact, HTTP +4.9 % short / +3.6 % long.
# --rollback: back to 1441b (12,25,37, --kv-resident 32768).
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090
case "${1:-}" in
  "") TARGET=T16-1441c-s28r-prod262r-vision-cpu.json ;;
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
for c in $(pgrep -P $P); do tr '\0' '\n' < /proc/$c/environ 2>/dev/null | grep -E "QSA_SLOTS_STREAM|SPLIT_OWN_RESERVE" ; readlink /proc/$c/exe; done
curl -s -m 5 $URL/v1/load; echo
