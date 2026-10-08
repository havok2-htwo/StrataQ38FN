#!/bin/sh
# Series H6 (2026-10-08): batching without --mtp-q4 (H10/H11 died: "MTP admission for slot 0 failed: unsupported native MMVQ GGML type").
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }
while ! grep -q SERIES-H5-DONE logs/series_h5.txt 2>/dev/null; do sleep 20; done
run H40-solo-nomtpq4   --workload parallel --drop mtp-q4
run H41-batch2-nomtpq4 --workload parallel --drop mtp-q4 --cfg parallel=2 --add batch-mtp
echo SERIES-H6-DONE $(date +%H:%M:%S)
