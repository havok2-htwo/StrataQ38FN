#!/usr/bin/env bash
# STRATA_PIPE_OVERLAP_MAX_QUEUE: exactness (batch_overlap_test vs the exactness-r1 dump), then HTTP 16/24 users.
cd "$(dirname "$0")"
S=/media/raid0/StrataQ38FN/Strata; X=results/overlap/ovq; E=results/overlap/exactness-r1
for q in 0; do
  env STRATA_MMQ_DIRECT=1 STRATA_IQ_MT_MIN=1 BATCH_TEST_LOG=$X/q$q.engine.log systemd-run --user --scope -q -p MemoryMax=90G \
    --unit=strata-ov-q$q python3 $S/tools/batch_overlap_test.py --exe $S/engine/strata-g3-ovq --config configs/T16-mmqtab.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow" --env STRATA_PIPE_OVERLAP=1 \
    --env STRATA_PIPE_OVERLAP_MAX_QUEUE=$q --solo-json $E/ov-on.json --compare $E/ov-on.json --dump $X/q$q.json > $X/q$q.out 2>&1
  echo "exactness q$q rc=$? $(grep -cE 'DIFFERS|DID NOT' $X/q$q.out) differ, overlapped $(grep -c 'overlapped read of' $X/q$q.engine.log), drained-queued $(grep -c 'admissions queued behind' $X/q$q.engine.log)"
  tail -3 $X/q$q.out
done
R=./run_strata_variant.sh
for rep in 1 2; do for cfg in T16-ovq0 T16-ovq1; do
  $R strata-$cfg-short-r$rep configs/$cfg.json 16,24 --rounds 2 --max-tok 512 --prompt-tok 100 2>&1 | tail -3
  $R strata-$cfg-long-r$rep configs/$cfg.json 16,24 --rounds 2 --max-tok 512 2>&1 | tail -3
  for c in short long; do f=results/engine-strata-$cfg-$c-r$rep.log; n=$(grep -n "engine started" $f | tail -1 | cut -d: -f1)
    echo "$cfg $c r$rep: overlapped $(tail -n +$n $f | grep -c 'overlapped read of'), drained (queued) $(tail -n +$n $f | grep -c 'admissions queued behind')"; done
done; done
echo SERIES OVQ DONE
