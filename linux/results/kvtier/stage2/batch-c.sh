#!/bin/bash
D=/media/raid0/StrataQ38FN/linux/results/kvtier/stage2
cd $D
C=/media/raid0/StrataQ38FN/linux/results/rebase143/exact/cmp.py
./gc.sh gc-off off.json
./gc.sh gc-ont ont.json
./gc.sh gc-on4 onv.json --batch 4 --batch-groups 2
./gc.sh gc-off4 off.json --batch 4 --batch-groups 2
python3 $C gc-ont.json gc-off.json
python3 $C gc-on4.json gc-off4.json
echo BATCH-C DONE
