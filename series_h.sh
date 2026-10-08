#!/bin/sh
# Series H (2026-10-08): on top of the optimized config - KV to SSD (session files), pcie_frac, VRAM for experts,
# KV streaming, batching.  Base: Strata/strata-unsloth-ud-iq4_xs.json.optimized
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }

run H1-ses-X  --workload session --server-arg=--slot-save-path --server-arg=X:/dev/StrataQ38FN/sessions-x
run H2-ses-C  --workload session --server-arg=--slot-save-path --server-arg=C:/StrataQ38FN-sessions
run H3-opt    --decode-runs 3
run H4-pcie07 --decode-runs 3 --tune '{"pcie_frac": 0.7}'
run H5-pcie085 --decode-runs 3 --tune '{"pcie_frac": 0.85}'
run H6-pcie10 --decode-runs 3 --tune '{"pcie_frac": 1.0}'
run H7-res1000 --decode-runs 3 --set vram-reserve-mib=1000
run H8-kvres  --decode-runs 3 --add kv-resident=32768 --env STRATA_KV_PREFETCH=1
run H9-par-solo --workload parallel
run H10-batch2  --workload parallel --cfg parallel=2 --add batch-mtp
run H11-batch2-agent --decode-runs 3 --cfg parallel=2 --add batch-mtp
echo SERIES-H-DONE $(date +%H:%M:%S)
