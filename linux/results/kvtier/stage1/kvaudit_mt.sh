#!/usr/bin/env bash
# multi-turn arm: start a config (kvaudit_run.sh, no warm, KEEP=1), run bench_multiturn, keep the engine log, stop.
#   kvaudit_mt.sh <label> <config.json> <users> <turns>
set -uo pipefail
LABEL=$1; CFG=$2; U=$3; T=$4
OUT=/media/raid0/StrataQ38FN/linux/results/kvtier/stage1
KEEP=1 "$OUT/kvaudit_run.sh" "$LABEL" "$CFG" 0 | grep -E "ready|^[0-3], |died|timeout|busy"
systemctl --user is-active -q "strata-kt-$LABEL.scope" || exit 1
/media/raid0/StrataQ38FN/Strata/.venv/bin/python -I /media/raid0/StrataQ38FN/linux/bench_multiturn.py --url http://127.0.0.1:8095 \
  --label "kv-$LABEL-mt-u$U" --users "$U" --turns "$T" --temperature 0 --out "$OUT/multiturn.jsonl" > "$OUT/mt-$LABEL.txt" 2>&1
tail -12 "$OUT/mt-$LABEL.txt"
grep -E "conversation cache|reused" "$OUT/engine-$LABEL.log" | grep -cE "restored" | sed 's/^/restores: /'
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
CG=/sys/fs/cgroup$(systemctl --user show -P ControlGroup "strata-kt-$LABEL.scope")
echo "cgroup current $(cat $CG/memory.current) peak $(cat $CG/memory.peak)"; grep -E '^(anon|file|shmem|inactive_file|active_file) ' $CG/memory.stat
systemctl --user stop "strata-kt-$LABEL.scope"
