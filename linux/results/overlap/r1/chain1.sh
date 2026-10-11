#!/bin/bash
cd /media/raid0/StrataQ38FN/linux/results/overlap/r1
./run_r1.sh fault-d-cuda strata-g3-overlap "--env STRATA_PIPE_OVERLAP=1 --expect-fault --bg 4 --bg-new 200 --env STRATA_PIPE_OVERLAP_TEST_FAULT=cuda"
./run_r1.sh smoke-on strata-g3-overlap "--env STRATA_PIPE_OVERLAP=1"
./run_r1.sh off strata-g3-overlap "--env STRATA_PIPE_OVERLAP=0 --solo-json smoke-on.json --compare smoke-on.json"
./run_r1.sh padup strata-g3-overlap "--env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_PADUP=1 --solo-json smoke-on.json --compare off.json"
./run_r1.sh wait2 strata-g3-overlap "--env STRATA_PIPE_OVERLAP=1 --env STRATA_PIPE_OVERLAP_FORCE_WAIT=2 --solo-json smoke-on.json --compare off.json"
./run_bt16.sh bt16-off bt16-on
echo CHAIN1-DONE >> summary.txt
