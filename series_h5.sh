#!/bin/sh
# Series H5 (2026-10-08): largest sensible context on the optimized config (live 131072).
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized --workload longctx --decode-runs 3"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }

while ! grep -q SERIES-H4-DONE logs/series_h4.txt 2>/dev/null; do sleep 20; done

run H30-ctx128k
run H31-ctx192k        --set max-context=196608
run H32-ctx262k        --set max-context=262144
run H33-ctx192k-kvres  --set max-context=196608 --add kv-resident=32768 --env STRATA_KV_PREFETCH=1
run H34-ctx262k-kvres  --set max-context=262144 --add kv-resident=32768 --env STRATA_KV_PREFETCH=1
echo SERIES-H5-DONE $(date +%H:%M:%S)
