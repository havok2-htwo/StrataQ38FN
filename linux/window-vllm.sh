#!/usr/bin/env bash
# Window: Strata off, vLLM 0.31 + Qwen3.8-Flash-Next W4A16 (TP=4, expert parallel, FP8 n-gram table in pinned RAM),
#   arm "plain": prefix cache on            -> community bench (short 1/8/16, 1.2K 16, multiturn 16x3) + bench_agentic
#   arm "kvram": + 80 GiB KV tier in RAM    -> multiturn 16x3 + bench_agentic
# Production (strata.service) is started again at the end, also on errors/Ctrl-C. Results: results/cbvllm-*.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PORT=8096; URL=http://127.0.0.1:$PORT; PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; M=qwen3.8-flash-next
log() { echo "[$(date +%T)] $*" | tee -a results/cbvllm-window.log; }
gpus_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
stop_vllm() { [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null; sleep 5
  pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null; sleep 3
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null; gpus_free; VPID=; }
back() { stop_vllm; systemctl --user start strata.service; log "production started again"; }
trap back EXIT
start_vllm() { local arm=$1 extra=$2
  PORT=$PORT SPEC=0 QWEN4_PLE_FP8_SCALE=1.995e-4 EXTRA="--enable-expert-parallel $extra" \
    vllm/run_vllm031_flashnext.sh > results/cbvllm-$arm-server.log 2>&1 &
  VPID=$!; local t0; t0=$(date +%s)
  until curl -s -m 3 $URL/v1/models | grep -q $M; do
    kill -0 $VPID 2>/dev/null || { log "vLLM ($arm) died"; tail -15 results/cbvllm-$arm-server.log; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { log "vLLM ($arm) start timeout"; return 1; }; sleep 10; done
  log "vLLM ($arm) ready after $(( $(date +%s) - t0 )) s"; }
B="$PY -I bench_concurrency.py --url $URL --model $M --vllm --out results/sweep.jsonl"
MT="$PY -I bench_multiturn.py --url $URL --model $M --vllm --users 16 --turns 3 --out results/multiturn.jsonl"
AG="$PY -I bench_agentic.py --url $URL --model $M --vllm --out results/agentic.jsonl"

log "window start: stopping strata.service"; systemctl --user stop strata.service; gpus_free || { log "GPUs not free"; exit 1; }
if start_vllm plain ""; then
  $B --label cbvllm-plain-short --levels 1,8,16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 > results/cbvllm-plain-short.txt 2>&1; log "plain short done"
  $B --label cbvllm-plain-long --levels 16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0 > results/cbvllm-plain-long.txt 2>&1; log "plain long done"
  timeout 900 $MT --label cbvllm-plain-mt > results/cbvllm-plain-mt.txt 2>&1; log "plain multiturn done"
  timeout 2400 $AG --label cbvllm-plain-agentic > results/cbvllm-plain-agentic.txt 2>&1; log "plain agentic done"
fi
stop_vllm
KV='--kv-transfer-config {"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":85899345920}}'
if start_vllm kvram "$KV"; then
  timeout 900 $MT --label cbvllm-kvram-mt > results/cbvllm-kvram-mt.txt 2>&1; log "kvram multiturn done"
  timeout 2400 $AG --label cbvllm-kvram-agentic > results/cbvllm-kvram-agentic.txt 2>&1; log "kvram agentic done"
fi
trap - EXIT; back
until curl -s -m 3 http://127.0.0.1:8090/health | grep -q '"loaded": *true'; do sleep 5; done; log "production loaded - Strata benches on :8090"
S="$PY -I bench_concurrency.py --url http://127.0.0.1:8090 --out results/sweep.jsonl"
$S --label cbs1441-short --levels 1,8,16 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 > results/cbs1441-short.txt 2>&1; log "strata short done"
$S --label cbs1441-long --levels 16 --rounds 2 --max-tok 512 --temperature 0 --warmup 0 > results/cbs1441-long.txt 2>&1; log "strata long done"
timeout 900 $PY -I bench_multiturn.py --url http://127.0.0.1:8090 --users 16 --turns 3 --label cbs1441-mt --out results/multiturn.jsonl > results/cbs1441-mt.txt 2>&1; log "strata multiturn done"
timeout 2400 $PY -I bench_agentic.py --url http://127.0.0.1:8090 --label cbs1441-agentic --out results/agentic.jsonl > results/cbs1441-agentic.txt 2>&1; log "strata agentic done"
log "window done"
