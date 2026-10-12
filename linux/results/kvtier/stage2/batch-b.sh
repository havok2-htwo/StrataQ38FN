#!/bin/bash
D=/media/raid0/StrataQ38FN/linux/results/kvtier/stage2
cd $D
./run_exact.sh t07:onv ov1:onv il:off bt16:off il:ont
./gc.sh gc-off off.json
./gc.sh gc-ont ont.json
./gc.sh gc-on4 onv.json --batch 4 --batch-groups 2
./gc.sh gc-off4 off.json --batch 4 --batch-groups 2
python3 /media/raid0/StrataQ38FN/linux/results/rebase143/exact/cmp.py gc-ont.json gc-off.json
python3 /media/raid0/StrataQ38FN/linux/results/rebase143/exact/cmp.py gc-on4.json gc-off4.json
echo BATCH-B DONE
