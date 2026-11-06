#!/usr/bin/env bash
# Production: engine strata-g3-ll10-borrow (Strata-ll10-borrow @324b1d80, sha256 20d43c775ab99baa; patches/v0.1.44.1-ll10-borrow)
# with STRATA_PARK_BORROW=1 only (STRATA_SERVE_STOP_BOUND=0, no CONV_MMAP / GATE_EVICT; MALLOC_MMAP_THRESHOLD_ 4 MiB stays),
# server Strata-143-prod checked out at Strata-srv-1106 @f8fcc276 (10015835 + ConvCacheLog borrow lines) (patches/v0.1.44.1-srv-1106). Deployed 2026-11-06 on the
# owner's go-ahead without a GPU test (window-borrow.sh follows at night).
# --rollback: engine strata-g3-1441j-reads2 (T16-1441j-fair-g8-mmap4m-...) and server 90f73293, i.e. the state of 276dfe4.
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090
case "${1:-}" in
  "") TARGET=T16-ll10-borrow-g8-mmap4m-prod262r-vision-cpu.json; SERVER=f8fcc2762e01d10a9c73e4e055be7df2826ff92a ;;
  --rollback) TARGET=T16-1441j-fair-g8-mmap4m-prod262r-vision-cpu.json; SERVER=90f73293 ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
[ -z "$(git -C $R/Strata-143-prod status --porcelain)" ] || { echo "Strata-143-prod has local changes, not touching it"; exit 1; }
echo "production.json: $(readlink $R/linux/configs/production.json) -> $TARGET"
echo "server: $(git -C $R/Strata-143-prod rev-parse --short HEAD) -> ${SERVER:0:8}"
git -C $R/Strata-143-prod checkout -q --detach $SERVER
ln -sfn "$TARGET" $R/linux/configs/production.json
systemctl --user restart strata.service; t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; exit 1; }; sleep 5; done
echo "loaded after $(( $(date +%s) - t0 )) s"
P=$(systemctl --user show -p MainPID --value strata.service)
for c in $(pgrep -P $P); do
  tr '\0' ' ' < /proc/$c/cmdline | awk '{print $1}'
  tr '\0' '\n' < /proc/$c/environ | grep -E '^(STRATA_PARK_BORROW|STRATA_SERVE_STOP_BOUND|MALLOC_MMAP_THRESHOLD_)=' | tr '\n' ' '; echo
done
