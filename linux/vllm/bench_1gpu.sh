#!/usr/bin/env bash
# The RTX PRO 6000 comparison protocol against a running OpenAI-compatible server (vLLM or Strata), same as the
# Strata runs of 2026-10-15 (results/sweep-rtxpro6000.jsonl): greedy, 2 rounds per user, 512 tokens out,
#   short (~100-token prompt) and long (~1.2K) at 1, 8 and 16 users,
#   plus a prefill probe: one user, ~4K and ~16K-token prompts, 16 tokens out, 3 rounds (prompt tok/s from TTFT).
#   LABEL=vllm031-1gpu VLLM=1 ./bench_1gpu.sh                     (vLLM on port 8003, thinking off via the template)
#   LABEL=strata-s3 URL=http://127.0.0.1:8090 ./bench_1gpu.sh      (Strata)
#   LEVELS=1,8,16  OUT=results/sweep-rtxpro6000.jsonl  PREFILL=4000,16000 (empty: skip)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
PY=${PY:-/media/data/miniconda3/envs/StrataQ38FN/Strata/.venv/bin/python}
URL=${URL:-http://127.0.0.1:8003}
: "${LABEL:?set LABEL}"
LEVELS=${LEVELS:-1,8,16}
OUT=${OUT:-results/sweep-rtxpro6000.jsonl}
PREFILL=${PREFILL-4000,16000}
ARGS=(--url "$URL" --rounds 2 --max-tok 512 --temperature 0 --out "$OUT")
[ -n "${VLLM:-}" ] && ARGS+=(--vllm --model qwen3.8-flash-next)

"$PY" bench_concurrency.py "${ARGS[@]}" --levels "$LEVELS" --prompt-tok 100 --label "$LABEL-short"
"$PY" bench_concurrency.py "${ARGS[@]}" --levels "$LEVELS" --warmup 0 --label "$LABEL-long"
for n in ${PREFILL//,/ }; do
  "$PY" bench_concurrency.py "${ARGS[@]}" --levels 1 --rounds 3 --max-tok 16 --prompt-tok "$n" \
    --warmup 0 --label "$LABEL-prefill$n"
done
echo "BENCH_DONE $LABEL"
