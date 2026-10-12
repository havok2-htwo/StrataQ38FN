#!/usr/bin/env bash
# forced-spill multi-turn arm: start a config (kvrun.sh, KEEP=1, port 8095), run bench_multiturn (greedy, reply hashes),
# snapshot GET /metrics' conversation_cache, the tier folder and the cgroup, stop.
#   mt.sh <label> <config.json> <users> <turns> <doc> [extra engine args...]
set -uo pipefail
LABEL=$1; CFG=$2; U=$3; T=$4; DOC=$5; shift 5
OUT=/media/raid0/StrataQ38FN/linux/results/kvtier/stage3
KEEP=1 "$OUT/kvrun.sh" "$LABEL" "$CFG" 0 "$@" | grep -E "ready|died|timeout|busy"
systemctl --user is-active -q "strata-kt3-$LABEL.scope" || exit 1
t0=$(date +%s)
/media/raid0/StrataQ38FN/Strata/.venv/bin/python -I /media/raid0/StrataQ38FN/linux/bench_multiturn.py --url http://127.0.0.1:8095 \
  --label "kt3-$LABEL-mt-u$U" --users "$U" --turns "$T" --doc "$DOC" --temperature 0 --out "$OUT/multiturn.jsonl" > "$OUT/mt-$LABEL.txt" 2>&1
echo "bench wall $(( $(date +%s) - t0 )) s" >> "$OUT/mt-$LABEL.txt"
sleep 3
curl -s http://127.0.0.1:8095/metrics | python3 -c "import json,sys; c=json.load(sys.stdin)['conversation_cache']; print(json.dumps(c, indent=1))" > "$OUT/metrics-$LABEL.json"
tail -9 "$OUT/mt-$LABEL.txt"
echo "cache restores: $(grep -c 'conversation cache: restored' $OUT/engine-$LABEL.log)  from disk: $(grep -c 'restored .*from=disk' $OUT/engine-$LABEL.log)"
grep "conversation tier:" "$OUT/engine-$LABEL.log" | tail -1
du -sb /media/raid0/strata-kvspill 2>/dev/null
CG=/sys/fs/cgroup$(systemctl --user show -P ControlGroup "strata-kt3-$LABEL.scope")
echo "cgroup current $(cat $CG/memory.current) peak $(cat $CG/memory.peak)"; grep -E '^(anon|file|shmem|file_dirty|file_writeback) ' $CG/memory.stat
systemctl --user stop "strata-kt3-$LABEL.scope"
