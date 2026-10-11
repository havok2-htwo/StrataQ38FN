#!/usr/bin/env bash
# Quick A/B (~4-5 min per arm): one server start per config, 16 users, greedy (temperature 0: an exact change gives
# the same tokens, so only the speed differs), short then 1.2K prompts on the same server.  At 16 users three
# repetitions differed by +-0.3 % (TESTLOG #41), so one run per arm is enough; repeat only to confirm a new default.
#   ab_quick.sh <tag> <config.json>...        (SERVER_DIR, LEVELS=16 as in run_strata_variant.sh; MIX=128: reply
#   lengths spread over 128..512 per request, so the slots do not finish in lockstep - closer to real traffic)
cd "$(dirname "$0")"
TAG=$1; shift
for cfg in "$@"; do
  n=$(basename "$cfg" .json)
  KEEP=1 ./run_strata_variant.sh ab-$TAG-$n-short "$cfg" ${LEVELS:-16} --rounds ${ROUNDS:-2} --max-tok 512 --prompt-tok 100 --temperature 0 ${MIX:+--max-tok-min $MIX} 2>&1 | tail -2
  "/media/raid0/StrataQ38FN/Strata/.venv/bin/python" -I bench_concurrency.py --url http://127.0.0.1:${PORT:-8090} --label ab-$TAG-$n-long \
    --levels ${LEVELS:-16} --rounds ${ROUNDS:-2} --max-tok 512 --temperature 0 --warmup 0 ${MIX:+--max-tok-min $MIX} --out results/sweep.jsonl 2>&1 | tail -1 | tee results/ab-$TAG-$n-long.txt
  cp "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['log'])" "$cfg")" results/engine-ab-$TAG-$n.log 2>/dev/null
  systemctl --user stop strata-bench.scope
done
