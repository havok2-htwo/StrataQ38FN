#!/bin/sh
# Series H3 (2026-10-08): partial prefix reuse when the context diverges (late / 75% / 50%), on the optimized config.
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized --workload diverge"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }

while ! grep -q SERIES-H2-DONE logs/series_h2.txt 2>/dev/null; do sleep 20; done

run H15-div-base
run H16-div-boundary --env STRATA_CACHE_MESSAGE_BOUNDARY=1
run H17-div-every8k  --add prompt-cache-every=8192 --add prompt-cache=12
run H18-div-both     --add prompt-cache-every=8192 --add prompt-cache=12 --env STRATA_CACHE_MESSAGE_BOUNDARY=1
echo SERIES-H3-DONE $(date +%H:%M:%S)
