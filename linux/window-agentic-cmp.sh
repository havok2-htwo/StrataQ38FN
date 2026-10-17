#!/usr/bin/env bash
# Second window (~30 min, production down), after window-cmp-vllm.sh: the agentic workloads on the same three arms.
#   bench_agentic.py (2 coding sessions, 6K system prompt, 20K context, ephemeral reminders, 3x3 parallel sub-agents,
#   a chat user beside them, greedy) and bench_multiturn.py (16 agents x 3 turns, 8K document + 3K per turn, T=0.7).
# Arms: Strata 1441g (production engine + server, text only) and vLLM 0.31 W4A16 without MTP (EP, 131K context); the
# MTP arm is in window-mtp-cmp.sh. Production is started again at the end, also on errors/Ctrl-C. Results: results/cmpa-*.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=/media/raid0/StrataQ38FN/Strata/.venv/bin/python; M=qwen3.8-flash-next; VP=8096; VURL=http://127.0.0.1:$VP; SP=8095
STRATA=/media/raid0/StrataQ38FN/Strata; SRV=/media/raid0/StrataQ38FN/Strata-143-prod
log() { echo "[$(date +%T)] $*" | tee -a results/cmpa-window.log; }
gpus_free() { for i in $(seq 90); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 2; done; return 1; }
stop_vllm() { [ -n "${VPID:-}" ] && kill "$VPID" 2>/dev/null; sleep 5
  pkill -f "vllm.entrypoints.openai.api_server.*--port $VP" 2>/dev/null; sleep 3
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $VP" 2>/dev/null; gpus_free; VPID=; }
back() { systemctl --user stop strata-bench.scope 2>/dev/null; stop_vllm; systemctl --user start strata.service; log "production started again"; }
trap back EXIT
start_vllm() { local arm=$1; shift
  env PORT=$VP QWEN4_PLE_FP8_SCALE=1.995e-4 "$@" vllm/run_vllm031_flashnext.sh > results/cmpa-$arm-server.log 2>&1 &
  VPID=$!; local t0; t0=$(date +%s)
  until curl -s -m 3 $VURL/v1/models | grep -q $M; do
    kill -0 $VPID 2>/dev/null || { log "vLLM ($arm) died"; tail -15 results/cmpa-$arm-server.log; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "vLLM ($arm) start timeout"; return 1; }; sleep 10; done
  log "vLLM ($arm) ready after $(( $(date +%s) - t0 )) s"; }
work() { local arm=$1 url=$2; shift 2
  timeout 1500 $PY -I bench_agentic.py --url $url "$@" --label cmpa-$arm-agentic --out results/agentic.jsonl \
    > results/cmpa-$arm-agentic.txt 2>&1; log "$arm agentic done (rc $?)"
  timeout 900 $PY -I bench_multiturn.py --url $url "$@" --users 16 --turns 3 --label cmpa-$arm-mt --out results/multiturn.jsonl \
    > results/cmpa-$arm-mt.txt 2>&1; log "$arm multiturn done (rc $?)"; }

log "window start: stopping strata.service"; systemctl --user stop strata.service; gpus_free || { log "GPUs not free"; exit 1; }

# arm 1: Strata 1441g on an isolated port, the service's drop-in env
systemctl --user reset-failed strata-bench.scope 2>/dev/null
( cd $SRV && STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1 systemd-run --user --scope --unit=strata-bench -p MemoryMax=224G \
    -p MemorySwapMax=0 $STRATA/.venv/bin/python $SRV/serve/server.py --engine strata --config /media/raid0/StrataQ38FN/linux/configs/CMPV-1441g-text.json \
    --port $SP --host 127.0.0.1 > /media/raid0/StrataQ38FN/linux/results/cmpa-strata-server.out 2>&1 < /dev/null & )
t0=$(date +%s)
until curl -s -m 3 http://127.0.0.1:$SP/health | grep -q '"loaded": *true'; do
  [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "strata start timeout"; break; }; sleep 5; done
log "strata ready after $(( $(date +%s) - t0 )) s"
work strata http://127.0.0.1:$SP
systemctl --user stop strata-bench.scope; gpus_free

if start_vllm nomtp SPEC=0 MAX_LEN=131072 EXTRA="--enable-expert-parallel"; then work vllm-nomtp $VURL --model $M --vllm; fi
stop_vllm
log "window done"
