#!/bin/sh
# Series H2 (2026-10-08): smallest safe --vram-reserve-mib on the optimized config (live: 1200 -> ~730-840 MiB free).
# The engine warns below 256 MiB free (WDDM paging can stall a verify graph for good) and aims at ~512 MiB.
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }

# wait for series H to finish (it uses the GPU)
while ! grep -q SERIES-H-DONE logs/series_h.txt 2>/dev/null; do sleep 20; done

run H12-res800 --decode-runs 3 --set vram-reserve-mib=800
run H13-res600 --decode-runs 3 --set vram-reserve-mib=600
run H14-res400 --decode-runs 3 --set vram-reserve-mib=400
echo SERIES-H2-DONE $(date +%H:%M:%S)
