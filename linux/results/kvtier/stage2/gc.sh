#!/bin/bash
# batch_groups_cache_test (two long conversations alternating over 3 rounds through the pipelined slots) in its own
# scope: gc.sh <name> <config.json> [extra engine args]; answers -> <name>.json, engine log -> <name>.engine.log
D=/media/raid0/StrataQ38FN/linux/results/kvtier/stage2; NAME=$1; CFG=$(readlink -f "$2"); shift 2
X="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 ${*:---batch 2 --batch-groups 2}"
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for i in $(seq 150); do gpu_free && break; sleep 2; done
gpu_free || { echo "$NAME: GPUs busy"; exit 9; }
ENVS=$(python3 -c "import json,sys; e=json.load(open(sys.argv[1])).get('env',{}); e.update(STRATA_MMQ_DIRECT='1', STRATA_IQ_MT_MIN='1'); print(' '.join(f'{k}={v}' for k,v in e.items()))" "$CFG")
echo "$NAME start $(date +%T) extra: $X env: $ENVS"
cd /media/raid0/StrataQ38FN/Strata-143
systemd-run --user --scope -q -p MemoryMax=200G --unit=strata-kt2-$NAME \
  $(for kv in $ENVS; do echo --setenv=$kv; done) --setenv=SMOKE_LOG=$D/$NAME.engine.log --setenv="STRATA_EXTRA_ARGS=$X" \
  /media/raid0/StrataQ38FN/Strata/.venv/bin/python tools/batch_groups_cache_test.py "$CFG" $D/$NAME.json > $D/$NAME.out 2>&1
echo "$NAME rc=$? $(date +%T)"; tail -4 $D/$NAME.out
