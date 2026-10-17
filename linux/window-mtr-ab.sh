#!/usr/bin/env bash
# Window (~30 min, production down): (1) overlap on/off A/B for the multiturn follow-up TTFT regression
# (results/mt-regression-2026-10-17.md): arms A = CMPV-1441g-text (production minus vision), B = A without
# STRATA_PIPE_OVERLAP/_KVS, order ABBA, a fresh server per run on 127.0.0.1:8095 with the service's drop-in env;
# per run bench_multiturn 16x3 (greedy) and bench_agentic (greedy) - TTFT p50/p95/max per turn and role.
# (2) vLLM 0.31 W4A16 start probe at --max-model-len 262144 (does it accept a full-context sequence?).
# Waits for production to be loaded before stopping it; starts production again on exit (production.json untouched).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; SP=8095; SURL=http://127.0.0.1:$SP; VP=8096
STRATA=/media/raid0/StrataQ38FN/Strata; SRV=/media/raid0/StrataQ38FN/Strata-143-prod; M=qwen3.8-flash-next
log() { echo "[$(date +%T)] $*" | tee -a results/mtr-window.log; }
gpus_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
prod_loaded() { for i in $(seq 120); do curl -s -m 3 http://127.0.0.1:8090/health | grep -q '"loaded": *true' && return 0; sleep 5; done; return 1; }
stop_bench() { systemctl --user stop strata-bench.scope 2>/dev/null; pkill -f "vllm.entrypoints.openai.api_server.*--port $VP" 2>/dev/null; sleep 3
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $VP" 2>/dev/null; gpus_free; }
back() { stop_bench; systemctl --user start strata.service; log "production started again"; }
trap back EXIT
prod_loaded || log "warning: production was not loaded before the window"
log "window start: stopping strata.service"; systemctl --user stop strata.service; gpus_free || { log "GPUs not free"; exit 1; }
run_arm() { local arm=$1 rep=$2 cfg=/media/raid0/StrataQ38FN/linux/configs/MTR-$1.json L=mtr-$1-$2
  systemctl --user reset-failed strata-bench.scope 2>/dev/null
  ( cd $SRV && STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1 systemd-run --user --scope --unit=strata-bench -p MemoryMax=224G \
      -p MemorySwapMax=0 $STRATA/.venv/bin/python $SRV/serve/server.py --engine strata --config $cfg --port $SP --host 127.0.0.1 \
      > /media/raid0/StrataQ38FN/linux/results/$L-server.out 2>&1 < /dev/null & )
  local t0; t0=$(date +%s)
  until curl -s -m 3 $SURL/health | grep -q '"loaded": *true'; do
    [ $(( $(date +%s) - t0 )) -gt 600 ] && { log "$L start timeout"; stop_bench; return 1; }; sleep 5; done
  log "$L ready after $(( $(date +%s) - t0 )) s"
  timeout 900 $PY -I bench_multiturn.py --url $SURL --users 16 --turns 3 --temperature 0 --label $L-mt --out results/multiturn.jsonl \
    > results/$L-mt.txt 2>&1; log "$L multiturn done: $(tail -1 results/$L-mt.txt)"
  timeout 1500 $PY -I bench_agentic.py --url $SURL --label $L-agentic --out results/agentic.jsonl > results/$L-agentic.txt 2>&1
  log "$L agentic done: $(tail -1 results/$L-agentic.txt | cut -c1-90)"
  stop_bench; }
run_arm A 1; run_arm B 1; run_arm B 2; run_arm A 2
# vLLM full-context probe
env PORT=$VP QWEN4_PLE_FP8_SCALE=1.995e-4 SPEC=0 MAX_LEN=262144 EXTRA="--enable-expert-parallel" vllm/run_vllm031_flashnext.sh \
  > results/mtr-vllm262k-server.log 2>&1 &
t0=$(date +%s); res=timeout
while [ $(( $(date +%s) - t0 )) -lt 600 ]; do
  curl -s -m 3 http://127.0.0.1:$VP/v1/models | grep -q $M && { res="started"; break; }
  pgrep -f "vllm.entrypoints.openai.api_server.*--port $VP" >/dev/null || { res="refused/died"; break; }; sleep 10; done
log "vLLM 262144 probe: $res after $(( $(date +%s) - t0 )) s; $(grep -a -o -m1 -E '(ValueError|RuntimeError): .{0,200}' results/mtr-vllm262k-server.log)"
grep -a -m2 -E 'GPU KV cache size|Maximum concurrency' results/mtr-vllm262k-server.log | tee -a results/mtr-window.log
log "window done"
