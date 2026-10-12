#!/usr/bin/env bash
# Overnight stability soak of a config through the HTTP server: mixed load in cycles until HOURS elapse.
#   soak.sh <config.json> [HOURS=5] [LONG_TOK=100000]
# Per cycle: 16 users short, 16 users mixed lengths (MIX 128..512, 4 rounds), 24 agents x 4 turns, one long request,
# 8 cancelled streams. Every 60 s: health, GPU memory, scope memory, engine RSS, error counts -> results/soak/samples.csv
cd "$(dirname "$0")"
CFG=$(readlink -f "$1"); HOURS=${2:-5}; LONG_TOK=${3:-100000}
export SERVER_DIR=${SERVER_DIR:-/media/raid0/StrataQ38FN/Strata-143}
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; URL=http://127.0.0.1:8090; OUT=results/soak
KEEP=1 ./run_strata_variant.sh soak-warm "$CFG" 1 --rounds 1 --max-tok 8 --warmup 0 >/dev/null 2>&1
systemctl --user is-active -q strata-bench.scope || { echo "server did not start"; exit 1; }
LOG=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['log'])" "$CFG")
L0=$(grep -n "engine started" "$LOG" | tail -1 | cut -d: -f1)
CG=/sys/fs/cgroup$(systemctl --user show -P ControlGroup strata-bench.scope)
echo "ts,healthy,gpu0,gpu1,gpu2,gpu3,cg_current_gb,engine_rss_gb,err,watchdog,terminate,cuda_err" > $OUT/samples.csv
( while systemctl --user is-active -q strata-bench.scope; do
    h=$(curl -s -m 5 $URL/health | grep -c '"loaded": *true')
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ',' )
    c=$(awk '{printf "%.1f", $1/1e9}' $CG/memory.current 2>/dev/null)
    pid=$(pgrep -o -x strata-g3-143k4 2>/dev/null || pgrep -o -f "engine/strata-g3" 2>/dev/null | head -1)
    r=$(ps -o rss= -p ${pid:-1} 2>/dev/null | awk '{printf "%.1f", $1/1e6}')
    t=$(tail -n +$L0 "$LOG")
    e=$(echo "$t" | grep -c "^ERR\| ERR "); w=$(echo "$t" | grep -ci "watchdog"); k=$(echo "$t" | grep -ci "terminate called"); u=$(echo "$t" | grep -ciE "cuda error|illegal address")
    echo "$(date +%FT%T),$h,${g%,},$c,$r,$e,$w,$k,$u" >> $OUT/samples.csv
    sleep 60; done ) &
SAMPLER=$!
END=$(( $(date +%s) + HOURS*3600 )); cyc=0
while [ $(date +%s) -lt $END ] && systemctl --user is-active -q strata-bench.scope; do
  cyc=$((cyc+1)); echo "=== cycle $cyc $(date +%T)"
  $PY -I bench_concurrency.py --url $URL --label soak-c$cyc-short --levels 16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0.7 --warmup 0 --uid-offset $((cyc*100000)) --out $OUT/sweep.jsonl 2>&1 | grep -E "^ +16 "
  $PY -I bench_concurrency.py --url $URL --label soak-c$cyc-mix --levels 16 --rounds 4 --max-tok 512 --max-tok-min 128 --temperature 0.7 --warmup 0 --uid-offset $((cyc*100000+50000)) --out $OUT/sweep.jsonl 2>&1 | grep -E "^ +16 "
  $PY -I bench_multiturn.py --url $URL --label soak-c$cyc-mt --users 24 --turns 4 --temperature 0.7 --out $OUT/multiturn.jsonl 2>&1 | grep -E "^wall|follow-up"
  $PY -I bench_concurrency.py --url $URL --label soak-c$cyc-long --levels 1 --rounds 1 --max-tok 256 --prompt-tok $LONG_TOK --temperature 0 --warmup 0 --timeout 1800 --uid-offset $((cyc*100000+90000)) --out $OUT/sweep.jsonl 2>&1 | grep -E "^ +1 "
  $PY -I - <<'PYC'
import json, threading, urllib.request
def cancel(i):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": f"Write a long essay number {i} about rivers."}],
                       "max_tokens": 400, "stream": True}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", body,
                                                          {"Content-Type": "application/json"}), timeout=60)
        r.read(800); r.close()            # read a few chunks, then drop the connection mid-stream
    except Exception as e:
        print("cancel err", e)
ts = [threading.Thread(target=cancel, args=(i,)) for i in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]; print("8 cancelled streams done")
PYC
done
echo "=== end $(date +%T), cycles $cyc, server active: $(systemctl --user is-active strata-bench.scope)"
kill $SAMPLER 2>/dev/null
cp "$LOG" $OUT/engine-soak.full.log 2>/dev/null; tail -n +$L0 "$LOG" > $OUT/engine-soak.run.log
tail -3 $OUT/samples.csv
echo SOAK DONE
