#!/bin/bash
cd /media/raid0/StrataQ38FN/linux/results/overlap/m3
./run_m3.sh off strata-g3-overlap "--compare smoke-on.json"
./run_m3.sh ref strata-g3-mmqtab "--compare smoke-on.json"
./run_m3.sh padup strata-g3-overlap "--env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_PADUP=1 --compare smoke-on.json"
./run_m3.sh wait2 strata-g3-overlap "--env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_WAIT=2 --compare smoke-on.json"
./run_m3.sh on-1325 strata-g3-overlap "--env STRATA_PIPE_OVERLAP=1 --split 13,25,37"
./run_m3.sh off-1325 strata-g3-overlap "--split 13,25,37 --compare on-1325.json"
echo CHAIN1 DONE >> summary.txt
