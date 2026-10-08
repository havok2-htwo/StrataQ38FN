#!/bin/sh
# Series E (2026-10-08): memory-hierarchy A/B on UD-IQ4_XS, see ANALYSE-Speicherhierarchie-XS.md section 4.
# Requires the live stack to be stopped (Stop-All.bat). Each arm restarts the engine on port 8090.
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
run() { echo "===== $1 $(date +%H:%M:%S)"; $PY agent_bench.py "$@" 2>&1 | tee "logs/E-$1.txt"; }

run A0-live                                                    # baseline = live config
run E1a-ple32m   --set ple-row-cache=33554432                  # frees ~8.8 GiB commit
run E1b-ple16m   --set ple-row-cache=16777216
run E2-park      --set ple-row-cache=33554432 --add conversation-cache-mib=4096 --add conversation-cache-slots=2
run E3a-resexp   --set ple-row-cache=33554432 --drop resident-budget-gib --add resident-experts --env STRATA_LOOKAHEAD=0
run E3b-resexp-la --set ple-row-cache=33554432 --drop resident-budget-gib --add resident-experts
run A0-again                                                   # drift check
echo SERIES-E-DONE $(date +%H:%M:%S)
