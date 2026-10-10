#!/usr/bin/env bash
# Qwen3.8-Flash-Next W4A16 (VnimanieAI) on vLLM 0.29.0, TP=4 over the four RTX 3090s.
# The 102 GB bf16 PLE n-gram table is loaded as FP8 (patch/sitecustomize.py) and kept in pinned host RAM via UVA
# (--cpu-offload-params ngram_embedding: ~12.8 GB per GPU, ~51 GB in total); everything else stays in VRAM.
#   QWEN4_PLE_FP8_SCALE  global FP8 scale = absmax(table) / 448 (from ple_scan.py)
#   PORT, MAX_LEN, MAX_SEQS, GPU_UTIL, SPEC (0 = no MTP, N = MTP with N tokens), EXTRA
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=$(ls -d /media/raid0/Models/models--VnimanieAI--Qwen3.8-Flash-Next-W4A16/snapshots/*/ | head -1)
: "${QWEN4_PLE_FP8_SCALE:?set QWEN4_PLE_FP8_SCALE (absmax/448)}"
PORT=${PORT:-8003}
MAX_LEN=${MAX_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-64}
GPU_UTIL=${GPU_UTIL:-0.94}
SPEC=${SPEC:-0}
export QWEN4_PLE_FP8_SCALE
export PYTHONPATH="$HERE/patch${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/media/raid0/Models
SPEC_ARGS=()
if [ "$SPEC" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"qwen4_exp_mtp\",\"num_speculative_tokens\":$SPEC}")
fi
exec /home/administrator/G3_VLLM/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 4 \
  --cpu-offload-gb 14 --cpu-offload-params ngram_embedding \
  --language-model-only \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --host 127.0.0.1 --port "$PORT" \
  "${SPEC_ARGS[@]}" ${EXTRA:-}
