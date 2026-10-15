#!/usr/bin/env bash
# Build the Strata engine for an RTX PRO 6000 Blackwell (sm_120) on a host without a system CUDA toolkit
# (as done on 2026-10-15, docs: RTXPRO6000.md). CUDA 13 comes from pip into Strata's venv; CMake and FlashInfer
# expect the classic toolkit layout, so $CUDA_ROOT is a symlink tree over the pip packages (nothing is copied).
#   ./build-engine-sm120.sh Strata-1441        -> Strata-1441/build-sm120/strata   (0.1.44.1, what S1-S7 use)
#   ./build-engine-sm120.sh Strata-143         -> Strata-143/build-sm120/strata    (0.1.43)
# Env: ROOT (StrataQ38FN folder), CUDA_ROOT (symlink tree, default /home/administrator/G3_VLLM/cuda), JOBS
set -euo pipefail
TREE=${1:?usage: build-engine-sm120.sh <Strata tree folder, e.g. Strata-1441>}
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
V=$ROOT/Strata/.venv
CU=$V/lib/python3.12/site-packages/nvidia/cu13
CUDA_ROOT=${CUDA_ROOT:-/home/administrator/G3_VLLM/cuda}
JOBS=${JOBS:-$(nproc)}

# 1. CUDA 13 toolchain from pip (nvcc 13.4, cudart, cublas, cccl; curand for vLLM/FlashInfer's JIT)
if [ ! -x "$CU/bin/nvcc" ]; then
  "$V/bin/python" -m pip install "nvidia-cuda-nvcc==13.4.92" "nvidia-cuda-runtime==13.4.92" nvidia-cuda-cccl \
    nvidia-cublas nvidia-curand
fi

# 2. classic layout: bin, include, lib64 with unversioned .so names, lib64/stubs/libcuda.so (the driver's)
mkdir -p "$CUDA_ROOT/lib64/stubs"
ln -sfn "$CU/bin" "$CUDA_ROOT/bin"
ln -sfn "$CU/include" "$CUDA_ROOT/include"
[ -e "$CU/nvvm" ] && ln -sfn "$CU/nvvm" "$CUDA_ROOT/nvvm"
for f in "$CU"/lib/*; do
  b=$(basename "$f"); ln -sfn "$f" "$CUDA_ROOT/lib64/$b"
  case $b in *.so.*) ln -sfn "$f" "$CUDA_ROOT/lib64/${b%%.so.*}.so" ;; esac
done
ln -sfn /lib/x86_64-linux-gnu/libcuda.so "$CUDA_ROOT/lib64/stubs/libcuda.so"

# 3. configure + build only the engine. cmake from the pip package (the venv's bin/cmake shim points to the old host);
#    ggml from our own llama.cpp fork in Strata/third_party (as the 3090 builds), MMQ pointer table on.
CM=$V/lib/python3.12/site-packages/cmake/data/bin/cmake
cd "$ROOT/$TREE"
"$CM" -S . -B build-sm120 -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER="$CU/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_FLAGS="-I$CU/include" \
  -DSTRATA_MMQ_TAB=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=OFF \
  -DSTRATA_GGML_DIR="$ROOT/Strata/third_party/llama.cpp"
PATH=$CU/bin:$PATH make -C build-sm120 -j"$JOBS" strata
echo "built $ROOT/$TREE/build-sm120/strata (run it with LD_LIBRARY_PATH=$CU/lib, configs: \"lib_dirs\")"
