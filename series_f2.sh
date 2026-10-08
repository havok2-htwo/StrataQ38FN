#!/bin/sh
# Series F2 (2026-10-08): combined config from series E + adaptive-tier tuning (E4) + routing trace (E0).
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
mkdir -p traces
COMBO="--set ple-row-cache=33554432 --drop resident-budget-gib --add resident-experts --add conversation-cache-mib=6144 --add conversation-cache-slots=3"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $COMBO --decode-runs 3 "$@" 2>&1 | tee "logs/E-$name.txt"; }

run F0-combo
run F1-lag2      --env STRATA_ADAPT_LAG=2
run F2-async     --add adapt-async=1
run F3-swaps48   --add adapt-swaps=48
run F4-decay85   --add adapt-decay=0.85
run F5-static    --add adapt-every=0
run F6-trace     --add dump-routing=X:/dev/StrataQ38FN/traces/F6-routing.bin
run F0-again
echo SERIES-F2-DONE $(date +%H:%M:%S)
