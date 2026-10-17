#!/usr/bin/env bash
# Production: fair prompt reads, stage 1 (TESTLOG #88/#89): a long prompt read gives way at a chunk boundary to waiting
# requests with few new tokens (engine STRATA_READ_YIELD=1, server STRATA_YIELD_SHORT=2048 STRATA_YIELDS_MAX=4).
# Engine strata-g3-1441j-reads2 (sha256 b4034791; STRATA_READS unset = stage 2 off), server Strata-143-prod checked out
# at ll8-fairadmit 90f73293 (STRATA_ADMIT_BY_NEW stays off: its own OK pending).
# Validated 2026-10-17 (results/reads2, results/fa): exact everywhere (agent mix 48/48, 16 users 32/32, vision,
# batch_test16); short TTFT next to a cold 125K read 20.2 -> 1.2 s, 16-burst 17.1 -> 4.9 s, agent mix unchanged.
# --rollback: back to 1441g (engine strata-g3-1441f-lanes4, server 7f86f97c, no yield env).
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090; SRV=$R/Strata-143-prod
DROP=$HOME/.config/systemd/user/strata.service.d/read-yield.conf
case "${1:-}" in
  "") TARGET=T16-1441j-fair-prod262r-vision-cpu.json; COMMIT=90f73293 ;;
  --rollback) TARGET=T16-1441g-lanes4-prod262r-vision-cpu.json; COMMIT=7f86f97c ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
git -C "$SRV" diff --quiet HEAD || { echo "$SRV has uncommitted changes: not touching it"; exit 1; }
echo "production.json: $(readlink $R/linux/configs/production.json) -> $TARGET; server $(git -C $SRV rev-parse --short HEAD) -> $COMMIT"
git -C "$SRV" checkout -q --detach "$COMMIT"
if [ -z "${1:-}" ]; then
  printf '[Service]\nEnvironment=STRATA_YIELD_SHORT=2048\nEnvironment=STRATA_YIELDS_MAX=4\n' > "$DROP"
else
  rm -f "$DROP"
fi
systemctl --user daemon-reload
ln -sfn "$TARGET" $R/linux/configs/production.json
systemctl --user restart strata.service; t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; exit 1; }; sleep 5; done
echo "loaded after $(( $(date +%s) - t0 )) s"
P=$(systemctl --user show -p MainPID --value strata.service)
echo "server env: $(tr '\0' '\n' < /proc/$P/environ | grep -E '^STRATA_' | tr '\n' ' ')"
for c in $(pgrep -P $P); do tr '\0' '\n' < /proc/$c/environ 2>/dev/null | grep -E "READ_YIELD|SPEC_LANES=|READS=" ; readlink /proc/$c/exe; done
L=$R/Strata/strata-unsloth-ud-iq4_xs.log; L0=$(grep -an "engine started" $L | tail -1 | cut -d: -f1)
tail -n +${L0:-1} $L | grep -aoE "read_yield=[0-9]+|yield_row=[0-9]+" | sort -u | tr '\n' ' '; echo
curl -s -m 5 $URL/v1/load; echo
