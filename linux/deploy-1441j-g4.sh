#!/usr/bin/env bash
# Production: 1441j (stage 1 + admit-by-new) with a 4K checkpoint grid (--prompt-cache-every 4096 --prompt-cache 24),
# TESTLOG #92 (no gain in bench_agentic, +25 GiB RAM peak; live trial for "content first, prompt second" clients).
# --rollback: back to the 16K grid (T16-1441j-fair-prod262r-vision-cpu.json).
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090
case "${1:-}" in
  "") TARGET=T16-1441j-fair-g4-prod262r-vision-cpu.json ;;
  --8k) TARGET=T16-1441j-fair-g8-prod262r-vision-cpu.json ;;
  --rollback) TARGET=T16-1441j-fair-prod262r-vision-cpu.json ;;
  *) echo "usage: $0 [--8k|--rollback]"; exit 2 ;;
esac
echo "production.json: $(readlink $R/linux/configs/production.json) -> $TARGET"
ln -sfn "$TARGET" $R/linux/configs/production.json
systemctl --user restart strata.service; t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; exit 1; }; sleep 5; done
echo "loaded after $(( $(date +%s) - t0 )) s"
P=$(systemctl --user show -p MainPID --value strata.service)
for c in $(pgrep -P $P); do tr '\0' ' ' < /proc/$c/cmdline | grep -o "\-\-prompt-cache[^ ]* [0-9]*" | tr '\n' ' '; done; echo
