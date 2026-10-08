#!/bin/sh
# Series H4 (2026-10-08): speculative decoding on the optimized config (live: --spec 4 --spec-min-p 0.5).
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized --decode-runs 3"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }

while ! grep -q SERIES-H3-DONE logs/series_h3.txt 2>/dev/null; do sleep 20; done

run H20-spec4-p05                                       # reference
run H21-minp03  --tune '{"spec_min_p": 0.3}'
run H22-minp07  --tune '{"spec_min_p": 0.7}'
run H23-spec3   --set spec=3
run H24-spec5   --set spec=5
run H25-spec6   --set spec=6
run H26-spec4-p05-again                                 # drift check
echo SERIES-H4-DONE $(date +%H:%M:%S)
