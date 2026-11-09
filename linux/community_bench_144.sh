#!/usr/bin/env bash
# Same-day comparison for an upstream community report (docs/COMMUNITY_BENCHMARKS.md): our production engine + server
# (0.1.43 + patches, 16 slots) against stock 0.1.44 (8 slots, its own serve/server.py), both text only, same requests.
# Takes production (strata.service) down for the whole run (~20 min) and starts it again on exit, also on Ctrl-C.
#   community_bench_144.sh            both arms, ours first
#   ARMS=stock community_bench_144.sh one arm only (ours|stock)
# Per arm: short prompts (100 tokens) at 1, 8 and 16 users, 1.2K prompts at 16 users (2 x 512 tokens per user,
# greedy), then 16 agents x 3 turns multi-turn (capped at MT_MAX_S seconds). Results: results/sweep.jsonl,
# results/multiturn.jsonl and results/<label>.txt with labels cb144-<arm>-{short,long,mt}.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
TAG=${TAG:-cb144}
ARMS=${ARMS:-ours stock}
MT_MAX_S=${MT_MAX_S:-240}
export PORT=${PORT:-8095} BIND=127.0.0.1 MEMMAX=${MEMMAX:-224G}
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python

back_to_production() {
  systemctl --user stop strata-bench.scope 2>/dev/null
  systemctl --user start strata.service
  echo "[$TAG] production started again $(date +%T)"
}
trap back_to_production EXIT

echo "[$TAG] production down $(date +%T)"
systemctl --user stop strata.service

for arm in $ARMS; do
  case $arm in
    ours)  cfg=configs/CB144-ours-prod-text.json;    srv=/media/raid0/StrataQ38FN/Strata-143-prod ;;
    stock) cfg=configs/CB144-stock-0.1.44-p8.json;   srv=/media/raid0/StrataQ38FN/Strata-144 ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  t0=$(date +%s)
  SERVER_DIR=$srv KEEP=1 ./run_strata_variant.sh "$TAG-$arm-short" "$cfg" 1,8,16 \
    --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 2>&1 | tail -6
  SERVER_DIR=$srv ATTACH=1 KEEP=1 ./run_strata_variant.sh "$TAG-$arm-long" "$cfg" 16 \
    --rounds 2 --max-tok 512 --temperature 0 --warmup 0 2>&1 | tail -3
  timeout "$MT_MAX_S" "$PY" -I bench_multiturn.py --url "http://127.0.0.1:$PORT" --label "$TAG-$arm-mt" \
    --users 16 --turns 3 --out results/multiturn.jsonl 2>&1 | tail -4
  systemctl --user stop strata-bench.scope
  echo "[$TAG] arm $arm took $(( $(date +%s) - t0 )) s"
done
