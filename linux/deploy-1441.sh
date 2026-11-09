#!/usr/bin/env bash
# Production on g3-1441 (v0.1.44.1 + our series, engine strata-g3-1441a, STRATA_MSG_CKPT_TAIL=8192); validated in the
# GPU window 2026-10-14 18:48-19:52 (linux/results/r1441). --rollback: back to v3 (0.1.43 line, server 7a49445b).
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090
case "${1:-}" in
  "") TARGET=T16-1441-prod262r-vision-cpu.json; CODE=7f86f97c ;;
  --rollback) TARGET=T16-143-prod262r-vision-cpu-v3.json; CODE=7a49445b ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
echo "production.json: $(readlink $R/linux/configs/production.json) -> $TARGET; server -> $CODE"
git -C $R/Strata-143-prod checkout -q --detach $CODE
ln -sfn "$TARGET" $R/linux/configs/production.json
systemctl --user restart strata.service; t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; exit 1; }; sleep 5; done
echo "loaded after $(( $(date +%s) - t0 )) s"
L=$R/Strata/strata-unsloth-ud-iq4_xs.log; L0=$(grep -an "engine started" $L | tail -1 | cut -d: -f1)
tail -n +$L0 $L | grep -aE "engine started|image position table|WARNING" | head -4 | cut -c1-160
P=$(systemctl --user show -p MainPID --value strata.service); tr '\0' '\n' < /proc/$P/environ | grep -E "STRATA_ADMIT|STRATA_PREFIX"
curl -s -m 5 $URL/v1/load; echo
