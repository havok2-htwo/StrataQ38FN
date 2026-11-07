#!/usr/bin/env bash
# Build a v0.2.12-based engine next to production without disturbing it.
#   build-212.sh <worktree> <binary name under Strata/engine/> [extra cmake args...]
# Same flags as strata-g3-ll10b (CUDA 13.0, sm_86, native/Q6K experts, KQUANTS, MMQ_TAB, tests), Ninja-free (Makefiles).
# Guard: hard MemoryMax (never MemoryHigh, see the oomd note), -j3, nice 19, CPUs 0-63 (production's aux CPUs are 65-127),
# so host MemAvailable stays above production's 16 GiB conversation-cache floor. JOBS / MEMMAX override.
set -euo pipefail
WT=$(realpath "$1"); NAME=$2; shift 2
JOBS=${JOBS:-3}; MEMMAX=${MEMMAX:-6G}
OUT=/media/raid0/StrataQ38FN/Strata/engine/$NAME
BUILD=$WT/build
LOG=/media/raid0/StrataQ38FN/linux/results/build-$NAME.log
mkdir -p "$(dirname "$LOG")"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if (( avail < 22 )); then echo "MemAvailable ${avail} GiB < 22 GiB: not building next to production" >&2; exit 2; fi
run() { systemd-run --user --scope --quiet -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 \
          nice -n 19 taskset -c 0-63 "$@"; }
{
  echo "== $(date -Is) build $NAME from $WT @ $(git -C "$WT" rev-parse --short HEAD) (jobs $JOBS, MemoryMax $MEMMAX)"
  run cmake -S "$WT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
    -DSTRATA_MMQ_TAB=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@"
  if grep -q "STRATA_MMQ_TAB off" "$BUILD/CMakeFiles/CMakeOutput.log" 2>/dev/null; then
    echo "WARNING: CMake turned STRATA_MMQ_TAB off"; fi
  run cmake --build "$BUILD" --target strata -j "$JOBS"
  install -m 755 "$BUILD/strata" "$OUT"
  sha256sum "$OUT"
  echo "== $(date -Is) done"
} 2>&1 | tee "$LOG"
