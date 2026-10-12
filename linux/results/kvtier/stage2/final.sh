#!/bin/bash
# stage 2a final gates + measurements on engine/strata-g3-143k3 (the commit's build)
D=/media/raid0/StrataQ38FN/linux/results/kvtier/stage2; L=/media/raid0/StrataQ38FN/linux; K=$L/results/kvaudit
C=$L/results/rebase143/exact/cmp.py
cd $D
./run_exact.sh il:onv bt16:onv t07:onv ov1:onv il:off bt16:off il:ont
./gc.sh gc-off off.json
./gc.sh gc-ont ont.json
python3 $C gc-ont.json gc-off.json
echo "=== kvaudit $(date +%T)"
for spec in "kt2-ref24 T16-143-64khoca128 24" "kt2-s24 T16-143-64khocas128 24" "kt2-ref48 T16-143-64khoca128 48" "kt2-s48 T16-143-64khocas128 48"; do
  set -- $spec
  echo "--- $1 ($2, $3 users x 4 turns) $(date +%T)"
  $K/kvaudit_mt.sh $1 $L/configs/$2.json $3 4
  sleep 10
done
echo "=== ab_quick $(date +%T)"
SERVER_DIR=/media/raid0/StrataQ38FN/Strata-143 $L/ab_quick.sh kt2 $L/configs/T16-143-64khoca.json $L/configs/T16-143-64khocas.json
echo FINAL DONE $(date +%T)
