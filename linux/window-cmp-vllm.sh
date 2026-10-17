#!/usr/bin/env bash
# Window (~35 min, production down): Strata 1441g (production engine + server, text only) against vLLM 0.31 W4A16
# without MTP (TP=4 + expert parallel, as 2026-10-15) and with MTP (SPEC=2, 8 seqs, as TESTLOG #16), at 1/2/3/4/8/16
# users, short (100-token) and 1.2K prompts, 2 x 512 tokens per user, greedy. Same requests in every arm.
# Production is started again at the end, also on errors/Ctrl-C. Results: results/cmpv-*, sweep.jsonl.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; M=qwen3.8-flash-next; VP=8096; VURL=http://127.0.0.1:$VP
log() { echo "[$(date +%T)] $*" | tee -a results/cmpv-window.log; }
gpus_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
stop_vllm() { [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null; sleep 5
  pkill -f "vllm.entrypoints.openai.api_server.*--port $VP" 2>/dev/null; sleep 3
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $VP" 2>/dev/null; gpus_free; VPID=; }
back() { systemctl --user stop strata-bench.scope 2>/dev/null; stop_vllm; systemctl --user start strata.service; log "production started again"; }
trap back EXIT
start_vllm() { local arm=$1; shift
  env PORT=$VP QWEN4_PLE_FP8_SCALE=1.995e-4 "$@" vllm/run_vllm031_flashnext.sh > results/cmpv-$arm-server.log 2>&1 &
  VPID=$!; local t0; t0=$(date +%s)
  until curl -s -m 3 $VURL/v1/models | grep -q $M; do
    kill -0 $VPID 2>/dev/null || { log "vLLM ($arm) died"; tail -15 results/cmpv-$arm-server.log; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "vLLM ($arm) start timeout"; return 1; }; sleep 10; done
  log "vLLM ($arm) ready after $(( $(date +%s) - t0 )) s"; }
VB() { local arm=$1 short=$2 long=$3
  timeout 600 $PY -I bench_concurrency.py --url $VURL --model $M --vllm --out results/sweep.jsonl --label cmpv-$arm-short \
    --levels $short --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 > results/cmpv-$arm-short.txt 2>&1; log "$arm short done"
  timeout 600 $PY -I bench_concurrency.py --url $VURL --model $M --vllm --out results/sweep.jsonl --label cmpv-$arm-long \
    --levels $long --rounds 2 --max-tok 512 --temperature 0 --warmup 0 > results/cmpv-$arm-long.txt 2>&1; log "$arm long done"; }

log "window start: stopping strata.service"; systemctl --user stop strata.service; gpus_free || { log "GPUs not free"; exit 1; }

# arm 1: Strata 1441g, production engine and server (Strata-143-prod) with the service's drop-in env, text only
export STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1
SERVER_DIR=/media/raid0/StrataQ38FN/Strata-143-prod PORT=8095 BIND=127.0.0.1 MEMMAX=224G KEEP=1 \
  ./run_strata_variant.sh cmpv-strata-short configs/CMPV-1441g-text.json 1,2,3,4,8,16 \
  --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 > results/cmpv-strata-short.out 2>&1; log "strata short done"
SERVER_DIR=/media/raid0/StrataQ38FN/Strata-143-prod PORT=8095 ATTACH=1 KEEP=1 \
  ./run_strata_variant.sh cmpv-strata-long configs/CMPV-1441g-text.json 1,4,16 \
  --rounds 2 --max-tok 512 --temperature 0 --warmup 0 > results/cmpv-strata-long.out 2>&1; log "strata long done"
systemctl --user stop strata-bench.scope; gpus_free

# arm 2: vLLM without MTP (as 2026-10-15)
if start_vllm nomtp SPEC=0 EXTRA="--enable-expert-parallel"; then VB vllm-nomtp 1,2,3,4,8,16 1,4,16; fi
stop_vllm
# arm 3: vLLM with MTP (TESTLOG #16 settings; at most 8 concurrent sequences)
if start_vllm mtp SPEC=2 MAX_SEQS=8 GPU_UTIL=0.93 EXTRA="--max-num-batched-tokens 1024"; then VB vllm-mtp 1,2,3,4,8 1,4,8; fi
log "window done"
