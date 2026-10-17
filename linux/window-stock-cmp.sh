#!/usr/bin/env bash
# Window (~18 min, production down; worst case with every timeout hit ~90 min): the STOCK arm of the 2026-10-17
# comparison (window-cmp-vllm.sh + window-agentic-cmp.sh) - upstream Strata v0.1.44.1 (tag 296b59d2) as released:
# its own engine (Strata-1441-stock/build/strata, built with setup.py's release flags, see the build note below) and
# its own server (Strata-1441-stock/serve/server.py), config configs/CMPV-stock-0.1.44.1-text.json = CMPV-1441g-text
# with only the upstream env vars, parallel 8 (stock's cap on a layer split without --batch-mtp: kVerifyMaxT = 8,
# generate.cpp clamps --batch to it; serve/server.py PARALLEL_MAX = 8), no drop-in env (STRATA_ADMIT_SHORT_FIRST /
# STRATA_PREFIX_DIAG are ours). Same benches and arguments as our arm:
#   bench_concurrency.py short (100-token prompt) at 1/2/3/4/8/16 users and 1.2K prompts at 1/4/16, 2 x 512 tokens per
#   user, greedy; then a fresh server for bench_agentic.py (2 coding sessions, 6K system prompt, 20K context, ephemeral
#   reminders, 3x3 sub-agents, a chat user, greedy) and bench_multiturn.py (16 agents x 3 turns, 8K doc + 3K/turn, T=0.7).
# Isolated on 127.0.0.1:8095 in the strata-bench scope (MemoryMax 224G). Production is started again at the end, also
# on errors/Ctrl-C/SIGTERM: each timeout-wrapped step runs in the background and is waited for (step), because GNU
# timeout puts itself in its own process group, so a foreground 'timeout ...' never sees the terminal's Ctrl-C and
# the window would just go on with the next step; the INT/TERM trap stops the current step and exits at once. Results: results/cmpv-stock-{short,long}.{txt,out}, cmpa-stock-{agentic,mt}.txt, rows in
# sweep.jsonl / agentic.jsonl / multiturn.jsonl (labels cmpv-stock-*, cmpa-stock-*), log results/cmps-window.log.
#
# Build note (not done by this script): the engine must exist first -
#   git -C /media/raid0/StrataQ38FN/Strata-1441 worktree add --detach /media/raid0/StrataQ38FN/Strata-1441-stock v0.1.44.1
#   cmake -G Ninja -S Strata-1441-stock -B Strata-1441-stock/build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
#     -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
#     -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 -DSTRATA_GGML_DIR=/media/raid0/StrataQ38FN/Strata/third_party/llama.cpp
#   cmake --build Strata-1441-stock/build --target strata -j 64
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; SP=8095; SURL=http://127.0.0.1:$SP
STOCK=/media/raid0/StrataQ38FN/Strata-1441-stock; CFG=configs/CMPV-stock-0.1.44.1-text.json
CFGABS=/media/raid0/StrataQ38FN/linux/$CFG; ENGLOG=/media/raid0/StrataQ38FN/Strata/strata-cmpv-stock-1441.log
log() { echo "[$(date +%T)] $*" | tee -a results/cmps-window.log; }
gpus_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }

# preflight (before production goes down): a stock engine of the tag, its server, the config
[ -x $STOCK/build/strata ] || { echo "no stock engine at $STOCK/build/strata (build note above)"; exit 1; }
[ "$(git -C $STOCK rev-parse HEAD 2>/dev/null)" = 296b59d2f73c8f04a805df29c2fa97655adace78 ] || { echo "$STOCK is not at v0.1.44.1"; exit 1; }
git -C $STOCK diff --quiet HEAD || { echo "$STOCK has local changes"; exit 1; }
[ -f $CFGABS ] || { echo "no $CFG"; exit 1; }
# only upstream settings: nothing of ours from the calling shell (the config's "env" is all the engine gets)
for v in $(compgen -e | grep '^STRATA_'); do unset "$v"; done

back() { systemctl --user stop strata-bench.scope 2>/dev/null; gpus_free; systemctl --user start strata.service; log "production started again"; }
CUR=   # pid of the running step's timeout (it forwards TERM to the step's whole process group)
on_sig() { trap '' INT TERM; log "interrupted"; [ -n "$CUR" ] && { kill -TERM "$CUR"; wait "$CUR"; } 2>/dev/null; exit 130; }
trap back EXIT; trap on_sig INT TERM
step() { local t=$1; shift; timeout "$t" "$@" & CUR=$!; wait "$CUR"; local rc=$?; CUR=; return $rc; }   # step SECS CMD...
englog_cut() {   # this engine start's part of the (shared, cumulative) engine log
  [ -f $ENGLOG ] || return 0; local n; n=$(grep -n 'engine started' $ENGLOG | tail -1 | cut -d: -f1)
  tail -n +"${n:-1}" $ENGLOG > "results/engine-$1.run.log"; }

log "window start: stopping strata.service"; systemctl --user stop strata.service; gpus_free || { log "GPUs not free"; exit 1; }

# part 1: bench_concurrency, short then long on the same server (as cmpv-strata)
step 1500 env SERVER_DIR=$STOCK PORT=$SP BIND=127.0.0.1 MEMMAX=224G KEEP=1 \
  ./run_strata_variant.sh cmpv-stock-short $CFG 1,2,3,4,8,16 \
  --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 > results/cmpv-stock-short.out 2>&1; log "stock short done (rc $?)"
step 600 env SERVER_DIR=$STOCK PORT=$SP ATTACH=1 KEEP=1 \
  ./run_strata_variant.sh cmpv-stock-long $CFG 1,4,16 \
  --rounds 2 --max-tok 512 --temperature 0 --warmup 0 > results/cmpv-stock-long.out 2>&1; log "stock long done (rc $?)"
systemctl --user stop strata-bench.scope; gpus_free

# part 2: a fresh server (empty conversation cache, as cmpa-strata), bench_agentic + bench_multiturn
systemctl --user reset-failed strata-bench.scope 2>/dev/null
( cd $STOCK && systemd-run --user --scope --unit=strata-bench -p MemoryMax=224G -p MemorySwapMax=0 \
    $PY $STOCK/serve/server.py --engine strata --config $CFGABS --port $SP --host 127.0.0.1 \
    > /media/raid0/StrataQ38FN/linux/results/cmpa-stock-server.out 2>&1 < /dev/null & )
sleep 5; t0=$(date +%s); up=1
until curl -s -m 3 $SURL/health | grep -q '"loaded": *true'; do
  systemctl --user is-active -q strata-bench.scope || { log "stock server died during start"; up=0; break; }
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "stock start timeout"; up=0; break; }; sleep 5; done
if [ $up = 1 ]; then
  log "stock ready after $(( $(date +%s) - t0 )) s"
  step 1500 $PY -I bench_agentic.py --url $SURL --label cmpa-stock-agentic --out results/agentic.jsonl \
    > results/cmpa-stock-agentic.txt 2>&1; log "stock agentic done (rc $?)"
  step 900 $PY -I bench_multiturn.py --url $SURL --users 16 --turns 3 --label cmpa-stock-mt --out results/multiturn.jsonl \
    > results/cmpa-stock-mt.txt 2>&1; log "stock multiturn done (rc $?)"
  englog_cut cmpa-stock
fi
systemctl --user stop strata-bench.scope; gpus_free
log "window done"
