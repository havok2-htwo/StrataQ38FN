#!/usr/bin/env bash
# MAX_QUEUE with the server's waiting=N (Strata 645b4bce): HTTP 16/24 users, 2 repetitions, + 8 users.
cd "$(dirname "$0")"
R=./run_strata_variant.sh
for rep in 1 2; do for cfg in T16-ovq0 T16-ovq1; do
  $R strata-$cfg-w-short-r$rep configs/$cfg.json 8,16,24 --rounds 2 --max-tok 512 --prompt-tok 100 2>&1 | tail -4
  $R strata-$cfg-w-long-r$rep configs/$cfg.json 8,16,24 --rounds 2 --max-tok 512 2>&1 | tail -4
  for c in short long; do f=results/engine-strata-$cfg-w-$c-r$rep.log; n=$(grep -n "engine started" $f | tail -1 | cut -d: -f1)
    echo "$cfg $c r$rep: overlapped $(tail -n +$n $f | grep -c 'overlapped read of'), drained (queued) $(tail -n +$n $f | grep -c 'admissions queued behind')"; done
done; done
for cfg in T16-overlap-off T16-overlap; do
  $R strata-$cfg-u8-short configs/$cfg.json 8 --rounds 2 --max-tok 512 --prompt-tok 100 2>&1 | tail -2
  $R strata-$cfg-u8-long configs/$cfg.json 8 --rounds 2 --max-tok 512 2>&1 | tail -2
done
echo SERIES OVQ2 DONE
