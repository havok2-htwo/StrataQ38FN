#!/usr/bin/env bash
# Overnight stability soak of a config through the HTTP server: mixed load in cycles until HOURS elapse.
#   soak.sh <config.json> [HOURS=5] [LONG_TOK=100000] [MEMMAX=200G] [OUT=results/soak] [BURST=e.g. 70000,200000]
# Per cycle: 16 users short, 16 users mixed lengths (MIX 128..512, 4 rounds), 24 agents x 4 turns (new conversations
# every cycle), from cycle 3 on 8 agents revisiting cycle n-2's conversations (RAM or SSD-tier restore), one long
# request, 8 cancelled streams. After the cycles, per BURST size: 16 parallel requests of that many prompt tokens (slot
# images outside the cache budget at once). Every 60 s: health, GPU memory, scope memory, engine RSS, error counts ->
# OUT/samples.csv; every 1 s: MemAvailable + scope memory/events -> OUT/mem.csv; every 10 s: conversation cache and
# SSD tier counters -> OUT/cache.csv. After the start, linux/strata-reclaim-gguf.sh drops the GGUF expert page cache.
cd "$(dirname "$0")"
CFG=$(readlink -f "$1"); HOURS=${2:-5}; LONG_TOK=${3:-100000}; export MEMMAX=${4:-224G}; OUT=${5:-results/soak}; BURST=${6:-}
export SERVER_DIR=${SERVER_DIR:-/media/raid0/StrataQ38FN/Strata-143-prod}
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; URL=http://127.0.0.1:8090
mkdir -p "$OUT"; T0=$(date '+%F %T')
systemctl --user is-active -q strata.service && { echo "strata.service is running - stop it first"; exit 1; }
KEEP=1 ./run_strata_variant.sh soak-warm "$CFG" 1 --rounds 1 --max-tok 8 --warmup 0 >/dev/null 2>&1
systemctl --user is-active -q strata-bench.scope || { echo "server did not start"; exit 1; }
echo "reclaim: $(./strata-reclaim-gguf.sh)"
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
echo "ts,mem_avail_gb,cg_current_gb,anon_gb,file_gb,shmem_gb,ev_max,ev_oom,ev_oom_kill" > $OUT/mem.csv
( while systemctl --user is-active -q strata-bench.scope; do
    m=$(awk '/^MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo)
    st=$(awk '/^(anon|file|shmem) /{printf "%.2f,", $2/2^30}' $CG/memory.stat 2>/dev/null)
    ev=$(awk '/^(max|oom|oom_kill) /{printf "%s,", $2}' $CG/memory.events 2>/dev/null)
    echo "$(date +%T),$m,$(awk '{printf "%.2f", $1/2^30}' $CG/memory.current 2>/dev/null),${st}${ev%,}" >> $OUT/mem.csv
    sleep 1; done ) &
MEMSAMP=$!
echo "ts,parked,bytes_gb,evictions,restores,tier_spills,tier_restores,tier_written_gb,drops_small,drops_ram,drops_busy,drops_budget,tier_pending" > $OUT/cache.csv
( while systemctl --user is-active -q strata-bench.scope; do
    curl -s -m 5 $URL/metrics | python3 -c "
import sys, json, time
c = json.load(sys.stdin)['conversation_cache']; t = c.get('tier', {})
print(','.join(map(str, [time.strftime('%H:%M:%S'), c['parked'], round(c['bytes'] / 1e9, 2), c['evictions'], c['restores'],
      t.get('spills'), t.get('restores'), round(t.get('written_bytes', 0) / 1e9, 1), t.get('drops_small'), t.get('drops_ram'),
      t.get('drops_busy'), t.get('drops_budget'), t.get('pending')])))" >> $OUT/cache.csv 2>/dev/null
    sleep 10; done ) &
CACHESAMP=$!
END=$(( $(date +%s) + HOURS*3600 )); cyc=0
while [ $(date +%s) -lt $END ] && systemctl --user is-active -q strata-bench.scope; do
  cyc=$((cyc+1)); echo "=== cycle $cyc $(date +%T)"
  $PY -I bench_concurrency.py --url $URL --label soak-c$cyc-short --levels 16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0.7 --warmup 0 --uid-offset $((cyc*100000)) --out $OUT/sweep.jsonl 2>&1 | grep -E "^ +16 "
  $PY -I bench_concurrency.py --url $URL --label soak-c$cyc-mix --levels 16 --rounds 4 --max-tok 512 --max-tok-min 128 --temperature 0.7 --warmup 0 --uid-offset $((cyc*100000+50000)) --out $OUT/sweep.jsonl 2>&1 | grep -E "^ +16 "
  $PY -I bench_multiturn.py --url $URL --label soak-c$cyc-mt --users 24 --turns 4 --temperature 0.7 --uid-offset $((cyc*1000)) --out $OUT/multiturn.jsonl 2>&1 | grep -E "^wall|follow-up"
  [ $cyc -ge 3 ] && $PY -I bench_multiturn.py --url $URL --label soak-c$cyc-revisit --users 8 --turns 1 --temperature 0.7 --uid-offset $(((cyc-2)*1000)) --out $OUT/multiturn.jsonl 2>&1 | grep -E "^ +1 |^wall"
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
for b in ${BURST//,/ }; do
  systemctl --user is-active -q strata-bench.scope || break
  echo "=== burst 16 x $b tokens $(date +%T)"
  $PY -I bench_concurrency.py --url $URL --label soak-burst-$b --levels 16 --rounds 1 --max-tok 1500 --prompt-tok $b --temperature 0.7 --warmup 0 --timeout 3600 --uid-offset $((900000+b)) --out $OUT/sweep.jsonl 2>&1 | grep -E "^ +16 "
  echo "    min MemAvailable since start: $(tail -n +2 $OUT/mem.csv | cut -d, -f2 | sort -n | head -1) GB"
done
echo "=== end $(date +%T), cycles $cyc, server active: $(systemctl --user is-active strata-bench.scope)"
kill $SAMPLER $MEMSAMP $CACHESAMP 2>/dev/null
echo "memory: MemAvailable min $(tail -n +2 $OUT/mem.csv | cut -d, -f2 | sort -n | head -1) GB, scope max $(tail -n +2 $OUT/mem.csv | cut -d, -f3 | sort -n | tail -1) GiB, events (max,oom,oom_kill) $(tail -1 $OUT/mem.csv | cut -d, -f7-)"
echo "cache:  $(head -1 $OUT/cache.csv)"; echo "        $(tail -1 $OUT/cache.csv)"
echo "oomd:   $(journalctl -u systemd-oomd --since "$T0" --no-pager -q 2>/dev/null | grep -ci kill) kill lines"
cp "$LOG" $OUT/engine-soak.full.log 2>/dev/null; tail -n +$L0 "$LOG" > $OUT/engine-soak.run.log
tail -3 $OUT/samples.csv
systemctl --user stop strata-bench.scope
echo SOAK DONE
