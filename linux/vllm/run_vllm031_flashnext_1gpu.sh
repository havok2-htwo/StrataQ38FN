#!/usr/bin/env bash
# Qwen3.8-Flash-Next W4A16 (VnimanieAI) on vLLM 0.31.0 on ONE RTX PRO 6000 Blackwell (96 GB), in its own systemd scope.
# Same patch as the 4x RTX 3090 run (run_vllm031_flashnext.sh): the 102 GB bf16 PLE n-gram table is cast to FP8 and kept
# in exactly pinned host RAM (~51 GB, patch031/sitecustomize.py); experts and everything else stay in VRAM.
# TP=1, so no --enable-expert-parallel (that was only needed because 640 / TP4 is not divisible by the group size).
#   MODEL=/path/to/snapshot ./run_vllm031_flashnext_1gpu.sh     (the folder with config.json and the safetensors)
#   QWEN4_PLE_FP8_SCALE  global FP8 scale = absmax(table) / 448; default 1.995e-4 = ple_scan.py on this checkpoint
#   PORT, MAX_LEN, MAX_SEQS, GPU_UTIL, SPEC (0 = no MTP, N = MTP with N tokens), MEMMAX, EXTRA
# Stop: systemctl --user stop vllm-1gpu.scope
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL:?set MODEL to the W4A16 snapshot folder (config.json + model-*.safetensors)}"
[ -f "$MODEL/config.json" ] || { echo "no config.json in $MODEL" >&2; exit 1; }
VENV=${VENV:-/home/administrator/G3_VLLM/venv}
PORT=${PORT:-8003}
MAX_LEN=${MAX_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-64}
GPU_UTIL=${GPU_UTIL:-0.92}
SPEC=${SPEC:-0}
export QWEN4_PLE_FP8_SCALE=${QWEN4_PLE_FP8_SCALE:-1.995e-4}
export PYTHONPATH="$HERE/patch031${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# No system CUDA toolkit on this host. FlashInfer JIT-builds kernels on first use (nvcc, ninja, curand.h, and
# $CUDA_HOME/lib64/libcudart.so + lib64/stubs/libcuda.so to link): /home/administrator/G3_VLLM/cuda is a symlink tree
# in that layout over the pip CUDA 13 packages in Strata's venv (nvidia/cu13: nvcc, cudart, cublas, cccl, curand).
export CUDA_HOME=${CUDA_HOME:-/home/administrator/G3_VLLM/cuda}
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
SPEC_ARGS=()
if [ "$SPEC" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"qwen4_exp_mtp\",\"num_speculative_tokens\":$SPEC}")
fi
exec systemd-run --user --scope --unit=vllm-1gpu -p MemoryMax=${MEMMAX:-115G} -p MemorySwapMax=0 \
  "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 1 \
  --language-model-only \
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --host 127.0.0.1 --port "$PORT" \
  "${SPEC_ARGS[@]}" ${EXTRA:-}
