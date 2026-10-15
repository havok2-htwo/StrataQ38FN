#!/bin/sh
# Slot scaling on one RTX PRO 6000 (2026-10-15): Strata with 18/20/24/32 batch slots (--batch-mtp), each measured with as
# many users as slots, then vLLM 0.31 (max-num-seqs 64) at the same user counts. Same protocol as bench_1gpu.sh (greedy,
# 2 rounds, 512 tokens, short ~100 and long ~1.2K prompts); results go to results/sweep-rtxpro6000.jsonl.
#   ./slot_scaling_1gpu.sh            (VLLM_MODEL=<W4A16 snapshot> for the vLLM arm; empty skips it)
R=/media/data/miniconda3/envs/StrataQ38FN
PY=$R/Strata/.venv/bin/python
OUT=results/sweep-rtxpro6000.jsonl

wait_ready() {  # log pattern unit
  i=0
  until grep -q -E "$2" "$1"; do
    i=$((i+1))
    # the scope registers a moment after systemd-run is backgrounded: only treat "inactive" as dead after 30 s
    if [ $i -gt 150 ] || { [ $i -gt 6 ] && ! systemctl --user is-active --quiet "$3.scope"; }; then return 1; fi
    sleep 5
  done
}

sweep() {  # url label levels [extra args]
  url=$1; label=$2; levels=$3; shift 3
  cd "$R/linux" || exit 1
  "$PY" bench_concurrency.py --url "$url" --levels "$levels" --rounds 2 --max-tok 512 --prompt-tok 100 \
    --temperature 0 --label "$label-short" --out "$OUT" "$@"
  "$PY" bench_concurrency.py --url "$url" --levels "$levels" --rounds 2 --max-tok 512 \
    --temperature 0 --warmup 0 --label "$label-long" --out "$OUT" "$@"
}

for arm in "S5 18" "S6 20" "S7 24" "S4 32"; do
  set -- $arm; tag=$1; n=$2
  cfg=$(ls "$R"/linux/configs/"$tag"-rtxpro6000-batch"$n"-mtp.json)
  unit=strata-b$n; log=/tmp/claude-1000/$unit-server.log
  systemctl --user reset-failed "$unit.scope" 2>/dev/null
  cd "$R/Strata" || exit 1
  systemd-run --user --scope --unit="$unit" -p MemoryMax=110G -p MemorySwapMax=0 \
    "$PY" "$R/Strata-1441/serve/server.py" --engine strata --config "$cfg" --port 8090 > "$log" 2>&1 &
  if wait_ready "$log" '^ready:' "$unit"; then
    echo "ARM_READY strata-b$n vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    sweep http://127.0.0.1:8090 "rtxpro6000-b${n}mtp" "$n"
    echo "ARM_DONE strata-b$n"
  else
    echo "ARM_FAILED strata-b$n"; tail -5 "$log"
  fi
  systemctl --user stop "$unit.scope"; sleep 5
done

if [ -n "${VLLM_MODEL:-}" ]; then
  log=/tmp/claude-1000/vllm-scaling.log
  systemctl --user reset-failed vllm-1gpu.scope 2>/dev/null
  MODEL="$VLLM_MODEL" "$R/linux/vllm/run_vllm031_flashnext_1gpu.sh" > "$log" 2>&1 &
  if wait_ready "$log" 'Application startup complete' vllm-1gpu; then
    echo "ARM_READY vllm"
    sweep http://127.0.0.1:8003 vllm031-1gpu-scale 18,20,24,32 --vllm --model qwen3.8-flash-next
    echo "ARM_DONE vllm"
  else
    echo "ARM_FAILED vllm"; tail -5 "$log"
  fi
  systemctl --user stop vllm-1gpu.scope
fi
echo ALL_SCALING_DONE
