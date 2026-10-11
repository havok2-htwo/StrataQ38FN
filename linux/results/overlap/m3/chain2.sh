#!/bin/bash
cd /media/raid0/StrataQ38FN/linux/results/overlap/m3
F="--env STRATA_PIPE_OVERLAP=1 --expect-fault --bg 4 --bg-new 200"
./run_m3.sh fault-a-poll strata-g3-overlap "$F --env STRATA_PIPE_OVERLAP_TEST_FAULT=poll"
./run_m3.sh fault-b-throw strata-g3-overlap "$F --env STRATA_PIPE_OVERLAP_TEST_FAULT=throw"
./run_m3.sh fault-c-stage strata-g3-overlap "$F --env STRATA_PIPE_OVERLAP_TEST_FAULT=stage"
./run_m3.sh fault-e-stall strata-g3-overlap "$F --env STRATA_PIPE_OVERLAP_TEST_FAULT=stall --env STRATA_WATCHDOG_S=10 --env STRATA_PIPE_OVERLAP_JOIN_S=2"
echo CHAIN2 DONE >> summary.txt
