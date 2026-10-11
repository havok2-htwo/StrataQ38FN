#!/bin/bash
cd /media/raid0/StrataQ38FN/linux/results/overlap/exactness-r1
until grep -q '^CHAINB-DONE' summary.txt; do sleep 5; done
./run.sh soak2-on soak2-off
./e2e.sh
echo CHAINC-DONE >> summary.txt
