#!/bin/sh
# bench_agentic.py (defaults: 2 sessions, 6K system, 20K context, 10 main steps, sub-agents at 2 and 6, chat beside it)
# on one RTX PRO 6000: first against the Strata server already running on :8090 (S8 production-style config), then vLLM
# 0.31 with 128K context. Results: results/agentic-rtxpro6000.jsonl.   VLLM_MODEL=<W4A16 snapshot> ./agentic_1gpu.sh
R=/media/data/miniconda3/envs/StrataQ38FN
PY=$R/Strata/.venv/bin/python
OUT=results/agentic-rtxpro6000.jsonl
cd "$R/linux" || exit 1
echo "ARM_START strata $(date +%T)"
"$PY" -I bench_agentic.py --url http://127.0.0.1:8090 --label strata1441-s8-agentic --out "$OUT"
echo "ARM_DONE strata $(date +%T)"
systemctl --user stop strata-s8.scope; sleep 8
[ -n "${VLLM_MODEL:-}" ] || { echo ALL_AGENTIC_DONE; exit 0; }
log=/tmp/claude-1000/vllm-agentic.log
systemctl --user reset-failed vllm-1gpu.scope 2>/dev/null
MODEL="$VLLM_MODEL" MAX_LEN=131072 EXTRA="--enable-prompt-tokens-details" "$R/linux/vllm/run_vllm031_flashnext_1gpu.sh" > "$log" 2>&1 &
until grep -q -E 'Application startup complete|Engine core initialization failed' "$log"; do sleep 10; done
grep -q 'Application startup complete' "$log" || { echo "ARM_FAILED vllm"; tail -5 "$log"; exit 1; }
echo "ARM_START vllm $(date +%T) $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$log")"
"$PY" -I bench_agentic.py --url http://127.0.0.1:8003 --model qwen3.8-flash-next --vllm --label vllm031-agentic --out "$OUT"
echo "ARM_DONE vllm $(date +%T)"
systemctl --user stop vllm-1gpu.scope
echo ALL_AGENTIC_DONE
