#!/usr/bin/env bash
# Remaining Strata measurements (2026-10-10): Q4_K_XL, 3 GPUs, prefill vs prompt length, MTP off, users 24-40.
cd "$(dirname "$0")"
R=./run_strata_variant.sh
B="--max-tok 512"
# prefill sweep: one request per length, 32 output tokens; TTFT / prompt tokens = prefill rate
prefill() { for n in 1600 6500 13000 24000; do
  KEEP=1 $R "$1-pf$n" "$2" 1 --rounds 1 --max-tok 32 --prompt-tok $n --warmup 0 2>&1 | tail -1
done; systemctl --user stop strata-bench.scope; }

$R strata-K-q4kxl-solo configs/K-q4kxl-solo.json 1 --rounds 4 $B 2>&1 | tail -1
$R strata-K-q4kxl-b12-short configs/K-q4kxl-b12.json 12,24 --rounds 2 $B --prompt-tok 100 2>&1 | tail -2
$R strata-K-q4kxl-b12-long configs/K-q4kxl-b12.json 12 --rounds 2 $B 2>&1 | tail -1
$R strata-T3-solo configs/T3-solo.json 1 --rounds 4 $B 2>&1 | tail -1
$R strata-T3-b12-short configs/T3-b12.json 12,18 --rounds 2 $B --prompt-tok 100 2>&1 | tail -2
$R strata-M-nomtp configs/M-nomtp.json 1 --rounds 4 $B 2>&1 | tail -1
$R strata-U-b12-users configs/G12-pf512-share2.json 24,28,32,36,40 --rounds 2 $B --prompt-tok 100 2>&1 | tail -5
prefill strata-P-pf512 configs/G12-pf512-share2.json
prefill strata-P-auto configs/P-auto.json
prefill strata-P-auto32k configs/P-auto32k.json
echo SERIES DONE
