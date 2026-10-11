#!/bin/bash
cd /media/raid0/StrataQ38FN/linux/results/overlap/exactness-r1
until grep -q '^DONE bt16-off' summary.txt; do sleep 5; done
./run.sh ov-on ov-off ov-ref ov-padup ov-wait1 ov-wait2 ov-on-1325 ov-off-1325 ov-padup-1325 ov-wait1-1325
./run.sh soak-on soak-off
./run.sh fault-a fault-b fault-c fault-d fault-e
./run.sh bt16-off-1325 il-off-1325 il-on
echo CHAINB-DONE >> summary.txt
