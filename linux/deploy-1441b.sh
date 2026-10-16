#!/usr/bin/env bash
# Production on g3-1441-ll (= 1441a + the opt-in low-level levers, engine strata-g3-1441b: CUDA 13.0, upstream build
# defaults incl. MMQ_KQUANTS / Q6K_EXPERTS), switches STRATA_QSA_SLOTS_STREAM, STRATA_ATTN_VBATCH, STRATA_GR_FAST,
# STRATA_GR_DOWN_PF=2, STRATA_GR_NORM_FAST, STRATA_EARLY_SAMPLE, STRATA_MMVF_ROWS.  Validated 2026-10-16
# (linux/results/ll/w2, w3; TESTLOG #77).  Server code unchanged (7f86f97c).  --rollback: back to 1441a.
set -euo pipefail
R=/media/raid0/StrataQ38FN; URL=http://127.0.0.1:8090
case "${1:-}" in
  "") TARGET=T16-1441b-prod262r-vision-cpu.json ;;
  --rollback) TARGET=T16-1441-prod262r-vision-cpu.json ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
echo "production.json: $(readlink $R/linux/configs/production.json) -> $TARGET"
ln -sfn "$TARGET" $R/linux/configs/production.json
systemctl --user restart strata.service; t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "died"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "timeout"; exit 1; }; sleep 5; done
echo "loaded after $(( $(date +%s) - t0 )) s"
P=$(systemctl --user show -p MainPID --value strata.service)
for c in $(pgrep -P $P); do tr '\0' '\n' < /proc/$c/environ 2>/dev/null | grep -E "QSA_SLOTS_STREAM|ATTN_VBATCH|GR_FAST|EARLY_SAMPLE" ; readlink /proc/$c/exe; done
curl -s -m 5 $URL/v1/load; echo
