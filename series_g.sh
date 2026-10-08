#!/bin/sh
# Series G (2026-10-08): confirm --adapt-async 1 and tune it (swaps are now off the critical path).
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
COMBO="--set ple-row-cache=33554432 --drop resident-budget-gib --add resident-experts --add conversation-cache-mib=6144 --add conversation-cache-slots=3"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $COMBO --decode-runs 3 "$@" 2>&1 | tee "logs/E-$name.txt"; }

run G1-async        --add adapt-async=1
run G2-combo
run G3-async        --add adapt-async=1
run G4-async-sw192  --add adapt-async=1 --add adapt-swaps=192
run G5-async-every2 --add adapt-async=1 --add adapt-every=2
run G6-async-dec85  --add adapt-async=1 --add adapt-decay=0.85
echo SERIES-G-DONE $(date +%H:%M:%S)
