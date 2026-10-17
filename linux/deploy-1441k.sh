#!/usr/bin/env bash
# Production: stage 1 fair reads (deploy-1441j.sh) + admission order by estimated NEW tokens (server STRATA_ADMIT_BY_NEW=1,
# TESTLOG #87/#89): follow-up TTFT p50 11.8 -> 3.1 s, sub-agents 3.2 -> 1.9 s, cold first turns 18.7 -> 26.4 s (max wait
# 45 s), replies identical.  Server-only: same engine/config as 1441j, one drop-in line.
# --rollback: back to stage 1 alone (1441j).
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090; SRV=$R/Strata-143-prod
DROP=$HOME/.config/systemd/user/strata.service.d/admit-by-new.conf
[ "$(readlink $R/linux/configs/production.json)" = T16-1441j-fair-prod262r-vision-cpu.json ] \
    || { echo "production is not on 1441j (run deploy-1441j.sh first)"; exit 1; }
[ "$(git -C "$SRV" rev-parse --short=8 HEAD)" = 90f73293 ] || { echo "$SRV is not at 90f73293"; exit 1; }
case "${1:-}" in
  "") printf '[Service]\nEnvironment=STRATA_ADMIT_BY_NEW=1\n' > "$DROP"; echo "STRATA_ADMIT_BY_NEW=1 on" ;;
  --rollback) rm -f "$DROP"; echo "STRATA_ADMIT_BY_NEW off" ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
systemctl --user daemon-reload
systemctl --user restart strata.service; t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; exit 1; }; sleep 5; done
echo "loaded after $(( $(date +%s) - t0 )) s"
P=$(systemctl --user show -p MainPID --value strata.service)
echo "server env: $(tr '\0' '\n' < /proc/$P/environ | grep -E '^STRATA_' | tr '\n' ' ')"
curl -s -m 5 $URL/v1/load; echo
