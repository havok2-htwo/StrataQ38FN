#!/bin/bash
# (1) Stage-3 premise: k one-chunk pieces pipelined through the 4 stages (--prefill 144: a 144k-token read = k chunks)
# (2) Track H suspect: every prefill stage thread inherits the host thread's single CPU (0); --aux-cpus 64-127 moves them
P=/media/raid0/StrataQ38FN/linux/results/batchadmit/pipecheck
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142fc.json
run() {  # name prefill extra singles
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}' || { echo GPUs busy; return; }
  systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-ba-pipecheck-$1 \
    --setenv=PC_PREFILL=$2 --setenv=PC_EXTRA="$3" --setenv=PC_SINGLE=$4 \
    python3 $P/pipe_driver.py --config $CFG --out $P/$1 --reps 4 --phases idle > $P/$1.driver.out 2>&1
  echo "$1 rc=$? $(date +%T)"
}
run pf144 144 "" 151,295,439,583,1159
run pf144-aux 144 "--aux-cpus 64-127" 151,295,439,583,1159
run pf512-aux 512 "--aux-cpus 64-127" 150,300,450,600,1200,2400
