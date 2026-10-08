#!/bin/sh
# Series I (2026-10-08): untested levers from the lever hunt (GPU kernels, CPU pool, PCIe mode, prefill chunk, drafting, profile, server).
cd "$(dirname "$0")"
PY=Strata/.venv/Scripts/python.exe
B="--base strata-unsloth-ud-iq4_xs.json.optimized --decode-runs 3"
run() { name=$1; shift; echo "===== $name $(date +%H:%M:%S)"; $PY agent_bench.py $name $B "$@" 2>&1 | tee "logs/E-$name.txt"; }
# H7 also waits on H6 and uses port 8090 -> chain behind H7, not H6
while ! grep -q SERIES-H7-DONE logs/series_h7.txt 2>/dev/null; do sleep 20; done

run I00-ref                                                         # reference
# --- GPU floor: gfx1151 bitwise kernels, never tried on CUDA (arch_defaults.cpp:14-37)
run I01-v2tsum    --env STRATA_EXPERT_V2=1 --env STRATA_TSUM=1      # grouped VRAM-hit kernel for IQ3_S/IQ4_NL (iq_kernels.cu:3387)
run I02-v2        --env STRATA_EXPERT_V2=1                          # V2 without TSUM
run I03-lfuse     --env STRATA_LFUSE=1                              # fewer launches, shared expert + KV append
run I04-qfuse     --env STRATA_QFUSE=1                              # q8_1 images written by producers
run I05-gdnsplit  --env STRATA_GDN_SPLIT=1                          # GDN step over 4 blocks per head
run I06-plebatch  --env STRATA_PLE_BATCH=1                          # verify-window PLE K/V at once (decode opt-in, verify.cpp:748)
run I07-mmvf      --env STRATA_MMVF_ROWS=1                          # bf16 multi-row GEMV
run I08-lanecell  --env STRATA_ATTN_LANECELL=1                      # decode attention, one cell per thread
run I09-q6p       --env STRATA_Q6_PACKED=1                          # packed Q6_K head (extra copy, ~+500 MiB -> watch slots)
run I10-hcq8      --env STRATA_HC_Q8=1                              # HC projections as Q8_0 (bit-changing, ~+670 MiB)
# --- CPU pool (Zen 5, SMT siblings idle today)
run I11-pw6       --add pool-workers=6                              # fewer barrier participants
run I12-pw11      --add pool-workers=11                             # 4 unpinned extra workers
run I13-pw15      --add pool-workers=15                             # 8 unpinned extra workers (may lose)
run I14-mtmin1    --env STRATA_IQ_MT_MIN=1                          # single-token groups via Strata AVX-512 instead of ggml AVX2
run I15-gather    --env STRATA_IQ_GATHER=1                          # AVX-512 gather (slower on Zen 4, untested on Zen 5)
run I16-pf4k      --env STRATA_IQ_PREFETCH=4096                     # prefetch distance (default 2048)
run I17-pf1k      --env STRATA_IQ_PREFETCH=1024
# --- PCIe experts: copy engine / direct instead of in-graph copy kernel (generate.cpp:6201-6205); grep log for 'no progress'
run I18-dma       --add pcie-mode=dma --workload session
run I19-dmab      --add pcie-mode=dma --env STRATA_DMA_BATCH=2 --workload session
run I20-direct    --add pcie-mode=direct
# --- prefill
run I21-pftiming  --env STRATA_PREFILL_TIMING=1                     # diagnostic: PLE ms of cold vs re-read (do not compare tok/s)
run I22-ref-long  --workload longctx                                # long-context reference
run I23-chunk16k  --set prefill=auto:16384 --workload longctx       # check 'prompt chunk auto:' line
run I24-chunk32k  --set prefill=auto:32768 --workload longctx
run I25-k8v4-long --set kv=k8v4 --workload longctx                  # commit/VRAM; check longctx answer
run I26-sr0       --add short-read=0                                # short tails via batched path
run I27-sr16      --add short-read=16
# --- drafting
run I28-vocab-en  --add mtp-draft-vocab=X:/dev/StrataQ38FN/Strata/data/draft_vocab_en.bin   # smaller head (control for a later de arm)
run I29-hnorm     --add mtp-hnorm=stream                            # draft norm per stream (acceptance probe)
run I30-chain3    --add lookup-chain=3                              # S becomes 7 automatically
# --- start-up warmth: expert profile
run I31-coderprof --set expert-profile=X:/dev/StrataQ38FN/Strata/data/expert-profile-coder.bin
run I32-profsave  --add expert-profile-save=X:/dev/StrataQ38FN/expert-profile-learned.bin --add expert-profile-save-every=1 --workload session
if [ -f expert-profile-learned.bin ]; then
run I33-proflearn --set expert-profile=X:/dev/StrataQ38FN/expert-profile-learned.bin   # compare hit rate of requests 1-3
fi
# --- server
run I34-loopstop  --cfg 'reasoning_loop_recovery="stop"'           # no-regression check only
run I35-slots6    --set conversation-cache-slots=6                  # expect neutral on this workload
run I36-ref-again                                                   # drift check
echo SERIES-I-DONE $(date +%H:%M:%S)
