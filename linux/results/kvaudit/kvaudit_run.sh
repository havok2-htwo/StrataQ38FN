#!/usr/bin/env bash
# KV-capacity audit: start one config through serve/server.py (own scope), snapshot VRAM/RAM idle, optionally warm it
# with a short 16-user burst (batch-window captures, prompt buffers) and snapshot again, then stop.
#   kvaudit_run.sh <label> <config.json> [warm=0|1] [extra engine args appended...]
set -uo pipefail
LABEL=$1; SRC=$(readlink -f "$2"); WARM=${3:-1}; shift 3 || shift $#
OUT=/media/raid0/StrataQ38FN/linux/results/kvaudit
STRATA=/media/raid0/StrataQ38FN/Strata
SERVER_DIR=/media/raid0/StrataQ38FN/Strata-143
PORT=${PORT:-8095}
UNIT=strata-kh-$LABEL
CFG=$OUT/config-$LABEL.json
ENGLOG=$OUT/engine-$LABEL.log
rm -f "$ENGLOG"
python3 - "$SRC" "$CFG" "$ENGLOG" "$PORT" "$@" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["log"] = sys.argv[3]; c["port"] = int(sys.argv[4])
extra = sys.argv[5:]
# extra args: "--flag value" pairs replace an existing flag's value, else are appended
a = c["args"]; i = 0
while i < len(extra):
    f = extra[i]; v = extra[i+1] if i+1 < len(extra) and not extra[i+1].startswith("--") else None
    if f in a:
        k = a.index(f)
        if v is not None: a[k+1] = v
    else:
        a.append(f); 
        if v is not None: a.append(v)
    i += 2 if v is not None else 1
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "$used" -ge 1000 ]; then echo "GPU busy ($used MiB)"; exit 1; fi
cd "$SERVER_DIR" || exit 1
systemd-run --user --scope -q -p MemoryMax=${MEMMAX:-200G} -p MemorySwapMax=0 --unit="$UNIT" \
  "$STRATA/.venv/bin/python" "$SERVER_DIR/serve/server.py" --engine strata --config "$CFG" --port "$PORT" --host 0.0.0.0 \
  > "$OUT/server-$LABEL.out" 2>&1 < /dev/null &
sleep 5
t0=$(date +%s)
until curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"loaded": *true'; do
  if ! systemctl --user is-active -q "$UNIT.scope"; then echo "[$LABEL] died during start"; tail -20 "$OUT/server-$LABEL.out"; tail -20 "$ENGLOG"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 900 ]; then echo "[$LABEL] start timeout"; systemctl --user stop "$UNIT.scope"; exit 1; fi
  sleep 5
done
echo "[$LABEL] ready after $(( $(date +%s) - t0 )) s"
sleep 10
snap() {   # snap <tag>
  local f=$OUT/snap-$LABEL-$1.txt
  { echo "# $(date -Iseconds) $LABEL $1"
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
    nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv
    CG=/sys/fs/cgroup$(systemctl --user show -P ControlGroup "$UNIT.scope")
    echo "cgroup $CG memory.current $(cat $CG/memory.current) memory.peak $(cat $CG/memory.peak 2>/dev/null)"
    grep -E '^(anon|file|kernel|shmem|unevictable|mapped_file|file_mapped) ' $CG/memory.stat
    for p in $(cat $CG/cgroup.procs); do echo "pid $p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-80)"; grep -E 'VmRSS|VmLck|VmPin|RssAnon|RssFile|RssShmem' /proc/$p/status | tr '\n' ' '; echo; done
    grep -E 'MemTotal|MemAvailable|Mlocked|Unevictable' /proc/meminfo
    curl -s "http://127.0.0.1:$PORT/v1/status" | python3 -c "import json,sys;d=json.load(sys.stdin);print('status context',d.get('context'),'concurrency',d.get('concurrency'),'cache_max_tokens',d.get('cache_max_tokens'))"
  } > "$f" 2>&1
  cat "$f"
}
snap idle
if [ "$WARM" = "1" ]; then
  nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,nounits -lms 500 > "$OUT/gpumem-$LABEL-warm.csv" 2>/dev/null &
  SMI=$!
  "$STRATA/.venv/bin/python" -I /media/raid0/StrataQ38FN/linux/bench_concurrency.py --url "http://127.0.0.1:$PORT" --label "kv-$LABEL" \
    --levels 16 --rounds 1 --max-tok 128 --prompt-tok ${PROMPT_TOK:-1000} --temperature 0 --out "$OUT/sweep-$LABEL.jsonl" --req-out "" > "$OUT/warm-$LABEL.txt" 2>&1
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  tail -3 "$OUT/warm-$LABEL.txt"
  python3 - "$OUT/gpumem-$LABEL-warm.csv" <<'PY'
import csv,sys
m={}
for r in list(csv.reader(open(sys.argv[1])))[1:]:
    try: i=int(r[1]); u=int(r[2])
    except: continue
    m[i]=max(m.get(i,0),u)
print("peak used MiB during warm:", m)
PY
  sleep 5
  snap warm
fi
[ "${KEEP:-0}" = "1" ] || systemctl --user stop "$UNIT.scope"
