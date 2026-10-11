#!/usr/bin/env bash
# M5 of upstream/DESIGN-overlap.md: overlapped admission (STRATA_PIPE_OVERLAP=1) vs off, same binary
# (engine/strata-g3-overlap), through the HTTP server, 3 repetitions, short (~120 tok) and long (~1.2K tok) prompts.
#   ARMS="T16-overlap-off T16-overlap" REPS=3 ./series_overlap.sh
cd "$(dirname "$0")"
R=./run_strata_variant.sh
S="--rounds 2 --max-tok 512 --prompt-tok 100"; L="--rounds 2 --max-tok 512"
for rep in $(seq 1 ${REPS:-3}); do
  for cfg in ${ARMS:-T16-overlap-off T16-overlap}; do
    $R strata-$cfg-short-r$rep configs/$cfg.json 1,16,24 $S 2>&1 | tail -4
    grep -c "overlapped read of" results/engine-strata-$cfg-short-r$rep.log 2>/dev/null | sed 's/^/overlapped reads: /'
    $R strata-$cfg-long-r$rep configs/$cfg.json 1,16,24 $L 2>&1 | tail -4
    grep -c "overlapped read of" results/engine-strata-$cfg-long-r$rep.log 2>/dev/null | sed 's/^/overlapped reads: /'
  done
done
echo SERIES OVERLAP DONE
