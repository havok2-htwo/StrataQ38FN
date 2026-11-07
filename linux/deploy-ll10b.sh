#!/usr/bin/env bash
# Production: arm N3 of the 2026-11-07 night window (TESTLOG #93) as the default start: engine strata-g3-ll10b
# (Strata-ll10-borrow @23588120, sha256 99bbe1d2304201de; patches/v0.1.44.1-ll10-borrow 0001-0008) with
# STRATA_PARK_BORROW=1 STRATA_CONV_MMAP=1 STRATA_GATE_EVICT=1, STRATA_SERVE_STOP_BOUND at its default (on),
# MALLOC_MMAP_THRESHOLD_ / MALLOC_TRIM_THRESHOLD_ 1 MiB, conversation cache 124 GiB; server Strata-143-prod checked out at
# Strata-srv-1106 @22f4781f (patches/v0.1.44.1-srv-1106 0001-0010; background reads in it stay off: no STRATA_SERVER_READS).
# --prev: the 2026-11-06 state (ll10-borrow binary, borrow only, server f8fcc276); --rollback: 1441j-reads2 + server 90f73293.
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090
case "${1:-}" in
  "") TARGET=T16-ll10b-g8-mmap1m-prod262r-vision-cpu.json; SERVER=22f4781fc60b68eaf9d083841258320271f2cc27 ;;
  --prev) TARGET=T16-ll10-borrow-g8-mmap4m-prod262r-vision-cpu.json; SERVER=f8fcc2762e01d10a9c73e4e055be7df2826ff92a ;;
  --rollback) TARGET=T16-1441j-fair-g8-mmap4m-prod262r-vision-cpu.json; SERVER=90f73293 ;;
  *) echo "usage: $0 [--prev|--rollback]"; exit 2 ;;
esac
[ -z "$(git -C $R/Strata-143-prod status --porcelain)" ] || { echo "Strata-143-prod has local changes, not touching it"; exit 1; }
git -C $R/Strata-143-prod cat-file -e "$SERVER^{commit}" || { echo "server commit $SERVER unknown"; exit 1; }
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
  tr '\0' '\n' < /proc/$c/environ | grep -E '^(STRATA_PARK_BORROW|STRATA_CONV_MMAP|STRATA_GATE_EVICT|STRATA_SERVE_STOP_BOUND|MALLOC_MMAP_THRESHOLD_|MALLOC_TRIM_THRESHOLD_)=' | tr '\n' ' '; echo
done
