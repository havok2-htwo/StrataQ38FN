#!/bin/sh
# Series H7 (2026-10-08): speculative decoding OFF vs ON (all earlier arms ran --spec 4 + MTP).
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized --decode-runs 3"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }
while ! grep -q SERIES-H6-DONE logs/series_h6.txt 2>/dev/null; do sleep 20; done
run H50-spec-on                                   # reference: --spec 4 + MTP
run H51-spec-off  --set spec=0                    # no drafting: one token per window
echo SERIES-H7-DONE $(date +%H:%M:%S)
