#!/bin/bash
# lever 4 (batch self-commit, STRATA_BATCH_SELF_COMMIT=1; the multi-slot GDN arms (*_on, *_cv, STRATA_GDN_SLOTS) ran
# with the build of with-gdn-slots.patch, which was dropped - no gain): arms one at a time,
# each in its own scope.  Gates: the exactness tools with the new binary; timing: pf_driver.py (= profile2/prof_driver.py
# with PF_STAMP=fixed: identical prompts across arms) in decode mode, reference and new back to back.
D=/media/raid0/StrataQ38FN/linux/results/levers/self-commit
T=/media/raid0/StrataQ38FN/Strata-142/tools
EXE=/media/raid0/StrataQ38FN/Strata/engine/strata-g3-142lv
CFG=/media/raid0/StrataQ38FN/linux/configs/T16-142fcb.json
EXTRA="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --aux-cpus 65-127 --layer-split 12,24,37"
CFGENV="STRATA_MMQ_DIRECT=1 STRATA_SLOT_COPY_DIRECT=1 STRATA_PIPE_BYIELD_NODRAIN=1 STRATA_ADMIT_MERGE_FIRST=1 STRATA_CKPT_POOL=1 STRATA_CKPT_THREADS=1 STRATA_SLOT_CKPT_MOVE=1 STRATA_SLOT_COPY_BATCHSYNC=1 STRATA_SPLIT_MTP_BATCH=1 STRATA_IQ_MT_MIN=1"
SC="STRATA_BATCH_SELF_COMMIT=1"
ON="STRATA_BATCH_SELF_COMMIT=1 STRATA_GDN_SLOTS=1"
gpu_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{b=1} END{exit b}'; }
for arm in "$@"; do
  envs=""; pd=""
  case $arm in
    bt16_sc) envs="$SC"; cmd="python3 $T/batch_test16.py --exe $EXE --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra \"$EXTRA\" --dump $D/bt16-sc.json" ;;
    bt32_sc) envs="$SC"; cmd="python3 $T/batch_test32.py --exe $EXE --config $CFG --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra \"$EXTRA\" --dump $D/bt32-sc.json" ;;
    il_sc)   envs="$SC"; cmd="python3 $T/batch_interleave_test.py --exe $EXE --config $CFG --extra \"$EXTRA\"" ;;
    ov_sc)   envs="$SC"; cmd="python3 $T/batch_overlap_test.py --exe $EXE --config $CFG --extra \"$EXTRA\" --env STRATA_PIPE_OVERLAP=1 --env STRATA_BATCH_SELF_COMMIT=1 --dump $D/ov-sc.json" ;;
    ov_ref)  envs=""; cmd="python3 $T/batch_overlap_test.py --exe ${EXE%lv}fc --config $CFG --extra \"$EXTRA\" --env STRATA_PIPE_OVERLAP=1 --dump $D/ov-ref.json" ;;
    bt32_off) envs=""; cmd="python3 $T/batch_test32.py --exe $EXE --config $CFG --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra \"$EXTRA\" --dump $D/bt32-off.json" ;;
    ov_off)  envs=""; cmd="python3 $T/batch_overlap_test.py --exe $EXE --config $CFG --extra \"$EXTRA\" --env STRATA_PIPE_OVERLAP=1 --dump $D/ov-off.json --compare $D/ov-sc.json" ;;
    bt16_on) envs="$ON"; cmd="python3 $T/batch_test16.py --exe $EXE --config $CFG --batch 16 --n 16 --max-new 64 --long-tokens 1200 --extra \"$EXTRA\" --dump $D/bt16-on.json" ;;
    bt32_on) envs="$ON"; cmd="python3 $T/batch_test32.py --exe $EXE --config $CFG --batch 32 --n 32 --max-new 64 --long-tokens 1200 --extra \"$EXTRA\" --dump $D/bt32-on.json" ;;
    il_ref)  envs=""; cmd="python3 $T/batch_interleave_test.py --exe ${EXE%lv}fc --config $CFG --extra \"$EXTRA\"" ;;
    il_ref_min) CFGENV="STRATA_MMQ_DIRECT=1 STRATA_IQ_MT_MIN=1"; envs=""; cmd="python3 $T/batch_interleave_test.py --exe ${EXE%lv}fc --config $CFG --extra \"$EXTRA\"" ;;
    il_on_min)  CFGENV="STRATA_MMQ_DIRECT=1 STRATA_IQ_MT_MIN=1"; envs="$ON"; cmd="python3 $T/batch_interleave_test.py --exe $EXE --config $CFG --extra \"$EXTRA\"" ;;
    il_on)   envs="$ON"; cmd="python3 $T/batch_interleave_test.py --exe $EXE --config $CFG --extra \"$EXTRA\"" ;;
    ov_on)   envs="$ON"; cmd="python3 $T/batch_overlap_test.py --exe $EXE --config $CFG --extra \"$EXTRA\" --env STRATA_PIPE_OVERLAP=1 --env STRATA_BATCH_SELF_COMMIT=1 --env STRATA_GDN_SLOTS=1 --dump $D/ov-on.json" ;;
    d16_ref) pd="--config $D/cfg-ref.json --mode decode --batch 16 --seconds 30" ;;
    d16_sc)  pd="--config $D/cfg-lv.json --mode decode --batch 16 --seconds 30 --env $SC" ;;
    d16_on)  pd="--config $D/cfg-lv.json --mode decode --batch 16 --seconds 30 --env STRATA_BATCH_SELF_COMMIT=1 --env STRATA_GDN_SLOTS=1" ;;
    d32_ref) pd="--config $D/cfg-ref.json --mode decode --batch 32 --seconds 30" ;;
    d32_sc)  pd="--config $D/cfg-lv.json --mode decode --batch 32 --seconds 30 --env $SC" ;;
    d32_on)  pd="--config $D/cfg-lv.json --mode decode --batch 32 --seconds 30 --env STRATA_BATCH_SELF_COMMIT=1 --env STRATA_GDN_SLOTS=1" ;;
    p16_ref) pd="--config $D/cfg-ref.json --mode decode --batch 16 --seconds 30 --env STRATA_VERIFY_PROFILE=1" ;;
    p16_on)  pd="--config $D/cfg-lv.json --mode decode --batch 16 --seconds 30 --env STRATA_VERIFY_PROFILE=1 --env STRATA_BATCH_SELF_COMMIT=1 --env STRATA_GDN_SLOTS=1" ;;
    p16_sc)  pd="--config $D/cfg-lv.json --mode decode --batch 16 --seconds 30 --env STRATA_VERIFY_PROFILE=1 --env $SC" ;;
    p32_sc)  pd="--config $D/cfg-lv.json --mode decode --batch 32 --seconds 30 --env STRATA_VERIFY_PROFILE=1 --env $SC" ;;
    p16_cv)  pd="--config $D/cfg-lv.json --mode decode --batch 16 --seconds 30 --env STRATA_VERIFY_PROFILE=1 --env $SC --env STRATA_GDN_SLOTS=2" ;;
    p32_ref) pd="--config $D/cfg-ref.json --mode decode --batch 32 --seconds 30 --env STRATA_VERIFY_PROFILE=1" ;;
    p32_on)  pd="--config $D/cfg-lv.json --mode decode --batch 32 --seconds 30 --env STRATA_VERIFY_PROFILE=1 --env STRATA_BATCH_SELF_COMMIT=1 --env STRATA_GDN_SLOTS=1" ;;
    *) echo "unknown $arm"; continue ;;
  esac
  for i in $(seq 150); do gpu_free && break; sleep 2; done
  gpu_free || { echo "$arm: GPUs busy"; continue; }
  echo "$arm start $(date +%T)"
  if [ -n "$pd" ]; then
    PF_STAMP=fixed systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-lv-sc-$arm \
      python3 $D/pf_driver.py --out $D/$arm $pd > $D/$arm.driver.out 2>&1
  else
    env $CFGENV $envs BATCH_TEST_LOG=$D/$arm.engine.log systemd-run --user --scope -q -p MemoryMax=90G --unit=strata-lv-sc-$arm \
      --setenv=BATCH_TEST_LOG=$D/$arm.engine.log $(for kv in $CFGENV $envs; do echo --setenv=$kv; done) \
      bash -c "$cmd" > $D/$arm.out 2>&1
  fi
  echo "$arm rc=$? $(date +%T)"
  sleep 5
done
