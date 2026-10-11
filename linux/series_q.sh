#!/usr/bin/env bash
# Batching quick-win experiments (2026-10-11): fewer/larger pipeline groups, higher decode share, data-parallel 2x2.
cd "$(dirname "$0")"
R=./run_strata_variant.sh
S="--rounds 2 --max-tok 512 --prompt-tok 100"
L="--rounds 2 --max-tok 512"
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python

$R strata-Q2-b16-groups2-short configs/Q2-b16-groups2.json 16,24 $S 2>&1 | tail -2
$R strata-Q2-b16-groups2-long configs/Q2-b16-groups2.json 16 $L 2>&1 | tail -1
$R strata-Q2-b12-groups2-short configs/Q2-b12-groups2.json 12,24 $S 2>&1 | tail -2
$R strata-Q3-b12-share4-short configs/Q3-b12-share4.json 12,24 $S 2>&1 | tail -2
$R strata-Q3-b12-share4-long configs/Q3-b12-share4.json 12 $L 2>&1 | tail -1

# Experiment 1: two independent instances on GPUs 0+1 and 2+3, one scope (MemoryMax 90G)
systemctl --user reset-failed strata-dp.scope 2>/dev/null
systemd-run --user --scope --unit=strata-dp -p MemoryMax=90G -p MemorySwapMax=0 \
  ./dp_launch.sh "$PWD/configs/Q1-dp2-inst0.json:8091" "$PWD/configs/Q1-dp2-inst1.json:8092" \
  > results/dp2-launch.out 2>&1 < /dev/null &
sleep 10
t0=$(date +%s)
until curl -s -m 3 http://127.0.0.1:8091/health | grep -q '"loaded": *true' && \
      curl -s -m 3 http://127.0.0.1:8092/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata-dp.scope || { echo "dp2 died"; tail -20 results/server-dp-809*.out; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo "dp2 start timeout"; exit 1; }
  sleep 5
done
echo "[dp2] ready after $(( $(date +%s) - t0 )) s"
U=http://127.0.0.1:8091,http://127.0.0.1:8092
$PY -I bench_concurrency.py --url $U --label strata-Q1-dp2-short --levels 1,12,24,32 $S --out results/sweep.jsonl 2>&1 | tail -5
$PY -I bench_concurrency.py --url $U --label strata-Q1-dp2-long --levels 16 $L --out results/sweep.jsonl 2>&1 | tail -1
grep -hE "expert cache [0-9]+ slots|of its [0-9]+ profiled|resident RAM mode" /media/raid0/StrataQ38FN/Strata/strata-dp2-*.log | cut -c1-160
grep -E "^(anon|file) " /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/strata-dp.scope/memory.stat
systemctl --user stop strata-dp.scope
echo SERIES Q DONE
