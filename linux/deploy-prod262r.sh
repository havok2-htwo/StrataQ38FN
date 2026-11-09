#!/usr/bin/env bash
# Switch production Strata to T16-143-prod262r: RAM-first conversation cache (96 GiB / 320 slots, 16 GiB RAM floor,
# no parking under 1024 tokens), SSD spill from 8192 tokens (300 GB/day), glibc returning freed cache segments,
# MemoryMax 224G, GGUF expert page cache dropped after each load (strata-reclaim.service).
#   deploy-prod262r.sh              switch (Strata is down ~2-3 min; conversations parked in RAM are lost)
#   deploy-prod262r.sh --rollback   back to T16-143-prod262 (48 GiB / 48 slots; the new units stay, they fit both;
#                                   the unit before the switch is kept as ~/.config/systemd/user/strata.service.pre-prod262r)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; U=~/.config/systemd/user; SPILL=/media/raid0/strata-kvspill
NEW=T16-143-prod262r.json; OLD=T16-143-prod262.json; URL=http://127.0.0.1:8090
case "${1:-}" in
  "") TARGET=$NEW ;;
  --rollback) TARGET=$OLD ;;
  *) echo "usage: $0 [--rollback]"; exit 2 ;;
esac
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$HERE/configs/$TARGET"
sh -n "$HERE/strata-reclaim-gguf.sh"
[ -d "$SPILL" ] && [ -f "$HERE/systemd/strata-reclaim.service" ] && [ -f /media/raid0/StrataQ38FN/Strata-143-prod/serve/server.py ]
systemctl --user is-active -q strata-bench.scope && { echo "strata-bench.scope (soak) is still running - stop it first"; exit 1; }
echo "production.json: $(readlink "$HERE/configs/production.json") -> $TARGET"

# units first (a running unit is not affected until its next start); keep the installed unit once for a full undo
[ -f "$U/strata.service.pre-prod262r" ] || cp "$U/strata.service" "$U/strata.service.pre-prod262r"
cp "$HERE/systemd/strata.service" "$HERE/systemd/strata-reclaim.service" "$U/"
systemctl --user daemon-reload

systemctl --user stop strata.service
trap 'echo "deploy step failed - starting strata.service again"; systemctl --user start strata.service' ERR
for i in $(seq 30); do curl -s -m 2 $URL/health >/dev/null || break; sleep 2; done
curl -s -m 2 $URL/health >/dev/null && { echo "port 8090 still answers after the stop"; exit 1; }
# The SSD tier's fingerprint covers the engine argv and the STRATA_* env (STRATA_CONV_TIER_* excepted): sets written
# under the other config are foreign - never restored, not counted against the cap, kept until their TTL. Keep `budget`
# (today's written bytes).
n=$(find "$SPILL" -maxdepth 1 \( -name '*.ses' -o -name '*.meta' -o -name '.*.tmp' \) | wc -l)
echo "kvspill: removing $n files ($(du -sh "$SPILL" | cut -f1) before)"
find "$SPILL" -maxdepth 1 \( -name '*.ses' -o -name '*.meta' -o -name '.*.tmp' \) -delete
ln -sfn "$TARGET" "$HERE/configs/production.json"
trap - ERR
systemctl --user start strata.service

t0=$(date +%s)
until curl -s -m 3 $URL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata.service || { echo "strata.service died during the start"; journalctl --user -u strata.service -n 30 --no-pager; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "start timeout (15 min)"; exit 1; }
  sleep 5
done
echo "loaded after $(( $(date +%s) - t0 )) s; MemoryMax $(systemctl --user show -P MemoryMax strata.service)"
set +e +o pipefail   # diagnostics only from here on
LOG=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['log'])" "$HERE/configs/$TARGET")
L0=$(grep -an "engine started" "$LOG" | tail -1 | cut -d: -f1)
tail -n +"${L0:-1}" "$LOG" | grep -aE "memory guard|conversation cache|conversation tier" | head -8 | cut -c1-200

# strata-reclaim.service waits for /health plus 30 s, then reclaims
sleep 20
for i in $(seq 60); do [ "$(systemctl --user is-active strata-reclaim.service)" = activating ] || break; sleep 5; done
journalctl --user -u strata-reclaim.service -n 3 --no-pager -q | tail -2
CG=/sys/fs/cgroup$(systemctl --user show -P ControlGroup strata.service)
echo "strata.service $(awk '{printf "%.1f", $1/2^30}' $CG/memory.current) GiB; MemAvailable $(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB"
curl -s -m 5 $URL/metrics | python3 -c "import sys,json; c=json.load(sys.stdin)['conversation_cache']; print('conversation cache:', c['budget_mib'], 'MiB,', c['slots'], 'slots, tier', 'on' if c.get('tier') else 'off')"
