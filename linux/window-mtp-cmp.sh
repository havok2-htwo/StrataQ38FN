#!/usr/bin/env bash
# Third window (~15 min, production down): vLLM 0.31 W4A16 WITH MTP (SPEC=2, 8 seqs, expert parallel - without EP the
# W4A16 MoE refuses TP=4: intermediate size per partition 160 is not divisible by group_size 128), the same benches as
# window-cmp-vllm.sh (1/2/3/4/8 users short, 1/4/8 long) and window-agentic-cmp.sh (bench_agentic, multiturn 16x3).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; M=qwen3.8-flash-next; VP=8096; VURL=http://127.0.0.1:$VP
log() { echo "[$(date +%T)] $*" | tee -a results/cmpm-window.log; }
gpus_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
stop_vllm() { [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null; sleep 5
  pkill -f "vllm.entrypoints.openai.api_server.*--port $VP" 2>/dev/null; sleep 3
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $VP" 2>/dev/null; gpus_free; VPID=; }
back() { stop_vllm; systemctl --user start strata.service; log "production started again"; }
trap back EXIT
log "window start: stopping strata.service"; systemctl --user stop strata.service; gpus_free || { log "GPUs not free"; exit 1; }
try() { local tag=$1; shift
  env PORT=$VP QWEN4_PLE_FP8_SCALE=1.995e-4 "$@" vllm/run_vllm031_flashnext.sh > results/cmpm-$tag-server.log 2>&1 &
  VPID=$!; local t0; t0=$(date +%s)
  until curl -s -m 3 $VURL/v1/models | grep -q $M; do
    kill -0 $VPID 2>/dev/null || { log "vLLM ($tag) died: $(grep -a -m1 -o 'ValueError: .\{0,160\}' results/cmpm-$tag-server.log)"; stop_vllm; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "vLLM ($tag) start timeout"; stop_vllm; return 1; }; sleep 10; done
  log "vLLM ($tag) ready after $(( $(date +%s) - t0 )) s"; }
# 64K context first (bench_agentic's main agent needs it); fall back to 32K (then the main agent cannot run)
if try mtp64k SPEC=2 MAX_SEQS=8 GPU_UTIL=0.93 MAX_LEN=65536 EXTRA="--enable-expert-parallel --max-num-batched-tokens 1024" \
   || try mtp32k SPEC=2 MAX_SEQS=8 GPU_UTIL=0.93 MAX_LEN=32768 EXTRA="--enable-expert-parallel --max-num-batched-tokens 1024"; then
  B="$PY -I bench_concurrency.py --url $VURL --model $M --vllm --out results/sweep.jsonl"
  timeout 600 $B --label cmpv-vllm-mtp-short --levels 1,2,3,4,8 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 > results/cmpv-vllm-mtp-short.txt 2>&1; log "mtp short done"
  timeout 600 $B --label cmpv-vllm-mtp-long --levels 1,4,8 --rounds 2 --max-tok 512 --temperature 0 --warmup 0 > results/cmpv-vllm-mtp-long.txt 2>&1; log "mtp long done"
  timeout 1500 $PY -I bench_agentic.py --url $VURL --model $M --vllm --label cmpa-vllm-mtp-agentic --out results/agentic.jsonl > results/cmpa-vllm-mtp-agentic.txt 2>&1; log "mtp agentic done (rc $?)"
  timeout 900 $PY -I bench_multiturn.py --url $VURL --model $M --vllm --users 16 --turns 3 --label cmpa-vllm-mtp-mt --out results/multiturn.jsonl > results/cmpa-vllm-mtp-mt.txt 2>&1; log "mtp multiturn done (rc $?)"
fi
log "window done"
