#!/bin/bash
cd /media/raid0/StrataQ38FN/linux/results/overlap/m3
./run_m3.sh final-on strata-g3-overlap "--env STRATA_PIPE_OVERLAP=1 --compare smoke-on.json"
./run_m3.sh final-off strata-g3-overlap "--compare smoke-on.json"
echo CHAIN3 DONE >> summary.txt
