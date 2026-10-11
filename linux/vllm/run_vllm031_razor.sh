#!/usr/bin/env bash
# Qwen3.8-Flash-Next RAZOR-65B-A6B-NVFP4 (cleyesode: 256 of 512 experts kept, NVFP4 experts, FP8 n-gram table) on
# vLLM 0.31, 4x RTX 3090: TP=4 + expert parallel, NVFP4 via Marlin (weight-only on Ampere), n-gram table in pinned
# host RAM (vLLM's EngramConfig) with patch031's byte-wise lookup (Triton on sm_86 cannot load e4m3).
#   PORT, MAX_LEN, MAX_SEQS, GPU_UTIL, SPEC (0 = no MTP), BATCHED, EXTRA
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=$(ls -d /media/raid0/Models/models--cleyesode--Qwen3.8-Flash-Next-RAZOR-65B-A6B-NVFP4/snapshots/*/ | head -1)
PORT=${PORT:-8003}
MAX_LEN=${MAX_LEN:-262144}
MAX_SEQS=${MAX_SEQS:-12}
GPU_UTIL=${GPU_UTIL:-0.92}
SPEC=${SPEC:-2}
BATCHED=${BATCHED:-2048}
export QWEN4_PLE_AMPERE_LOOKUP=1
export PYTHONPATH="$HERE/patch031${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False,pinned_max_round_threshold_mb:1024}
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export HF_HOME=/media/raid0/Models HUGGINGFACE_HUB_CACHE=/media/raid0/Models
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
SPEC_ARGS=()
if [ "$SPEC" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"qwen4_exp_mtp\",\"num_speculative_tokens\":$SPEC}")
fi
exec /home/administrator/G3_VLLM/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3.8-flash-next-razor65 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --language-model-only \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCHED" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching --no-enable-flashinfer-autotune \
  --host 127.0.0.1 --port "$PORT" \
  "${SPEC_ARGS[@]}" ${EXTRA:-}
