#!/bin/bash
L=/media/raid0/StrataQ38FN/linux; K=$L/results/kvaudit
cd $L
echo "=== ab_quick REPS=3 $(date +%T)"
REPS=3 SERVER_DIR=/media/raid0/StrataQ38FN/Strata-143 ./ab_quick.sh kt2r $L/configs/T16-143-64khoca.json $L/configs/T16-143-64khocas.json 2>&1 | grep -E "vs T16|agg .* steady" | tail -8
echo "=== kvaudit repeat $(date +%T)"
for spec in "kt2-s24b T16-143-64khocas128 24" "kt2-ref24b T16-143-64khoca128 24"; do
  set -- $spec
  echo "--- $1 ($2, $3 users x 4 turns) $(date +%T)"
  $K/kvaudit_mt.sh $1 $L/configs/$2.json $3 4 | grep -E "^ +[1-4] |wall|restores"
  sleep 10
done
echo FINAL2 DONE $(date +%T)
