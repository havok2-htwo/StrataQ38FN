#!/usr/bin/env bash
# Our community benchmark (as community_bench_144.sh) against vLLM 0.31 with Qwen3.8-Flash-Next W4A16 (VnimanieAI),
# TP=4 + expert parallel, FP8 n-gram table in pinned RAM (vllm/run_vllm031_flashnext.sh). Production is stopped for
# the run and started again on exit (also on Ctrl-C). Results: results/cbvllm-*.txt, sweep.jsonl, multiturn.jsonl.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
TAG=${TAG:-cbvllm}; PORT=${PORT:-8096}; URL=http://127.0.0.1:$PORT; MT_MAX_S=${MT_MAX_S:-240}
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; LOG=results/$TAG-vllm-server.log
back() { [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null; sleep 5; pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null
  for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
  systemctl --user start strata.service; echo "[$TAG] production started again $(date +%T)"; }
trap back EXIT
echo "[$TAG] production down $(date +%T)"; systemctl --user stop strata.service
for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
PORT=$PORT SPEC=0 QWEN4_PLE_FP8_SCALE=1.995e-4 EXTRA="--enable-expert-parallel" vllm/run_vllm031_flashnext.sh > "$LOG" 2>&1 &
VPID=$!; t0=$(date +%s)
until curl -s -m 3 $URL/v1/models | grep -q qwen3.8-flash-next; do
  kill -0 $VPID 2>/dev/null || { echo "[$TAG] vLLM died, see $LOG"; tail -20 "$LOG"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "[$TAG] vLLM start timeout"; exit 1; }; sleep 10; done
echo "[$TAG] vLLM ready after $(( $(date +%s) - t0 )) s"
B="$PY -I bench_concurrency.py --url $URL --model qwen3.8-flash-next --vllm --out results/sweep.jsonl"
$B --label $TAG-short --levels 1,8,16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 | tee results/$TAG-short.txt | tail -6
$B --label $TAG-long --levels 16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0 | tee results/$TAG-long.txt | tail -3
timeout "$MT_MAX_S" $PY -I bench_multiturn.py --url $URL --model qwen3.8-flash-next --vllm --label $TAG-mt --users 16 --turns 3 \
  --out results/multiturn.jsonl 2>&1 | tee results/$TAG-mt.txt | tail -4
