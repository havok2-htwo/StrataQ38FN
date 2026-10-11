#!/bin/sh
# Series U (2026-10-11): engine upgrade 0.1.40.1 -> 0.1.42 on UD-IQ4_XS, one request at a time (batch_bench.py --suite engine).
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY batch_bench.py $name --suite engine "$@" 2>&1; }

run U0-v14001       --strata Strata                              # production engine (reference)
run U1-v142         --strata Strata-142                          # new defaults: tail skip, measured PCIe share, CPU prompt share
run U2-v142-old     --strata Strata-142 --env STRATA_ROUTE_TAIL_SKIP=0 --env STRATA_PCIE_FRAC_DEFAULT=old \
                    --env STRATA_PREFILL_CPU_SHARE=0             # 0.1.42 with the answer-changing defaults off
run U3-v142-pf512   --strata Strata-142 --set prefill=512        # Linux finding: short prompts faster with 512 chunks
run U0b-v14001      --strata Strata                              # reference again (drift)
echo SERIES-U-DONE
