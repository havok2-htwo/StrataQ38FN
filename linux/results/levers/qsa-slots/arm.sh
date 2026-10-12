#!/bin/bash
# one arm in its own scope: arm.sh <outdir> <name> <config.json> <command...>
# env = the config's env + STRATA_MMQ_DIRECT=1 STRATA_IQ_MT_MIN=1 (exact); engine log -> <outdir>/<name>.engine.log
D=$1; NAME=$2; CFG=$3; shift 3
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for i in $(seq 150); do gpu_free && break; sleep 2; done
gpu_free || { echo "$NAME: GPUs busy"; exit 9; }
ENVS=$(python3 -c "import json,sys; e=json.load(open(sys.argv[1])).get('env',{}); e.update(STRATA_MMQ_DIRECT='1', STRATA_IQ_MT_MIN='1'); print(' '.join(f'{k}={v}' for k,v in e.items()))" "$CFG")
echo "$NAME start $(date +%T) env: $ENVS"
env $ENVS BATCH_TEST_LOG=$D/$NAME.engine.log systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-l5-$NAME \
  --setenv=BATCH_TEST_LOG=$D/$NAME.engine.log $(for kv in $ENVS; do echo --setenv=$kv; done) \
  "$@" > $D/$NAME.out 2>&1
rc=$?
echo "$NAME rc=$rc $(date +%T)"
exit $rc
