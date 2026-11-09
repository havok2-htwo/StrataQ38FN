<!-- 2026-11-09, read-only code/log investigation (workflow v31-decode-kernel-levers), nothing measured yet. Production = v3.1 on strata-212q50 (Strata-212q50 @85277f09); note: bench binaries for a window must come from Strata-212q50/build, not Strata-212p/build. -->

# Kernel plan for Strata v3.1 on 4x RTX 3090 (sm_86)

## Bottom line

Not much is left to gain. Almost every v3.1 tensor type already runs on a tuned, exact path. There is one real kernel gap: the Q5_0 expert down projection, used in 33 of 48 layers. There is also one config lever, the layer split. Together they plausibly win about +3–7% aggregate at 8–12 users and about +1–4% at 1–4 users. None of this is measured yet. Everything else is 0–2% or unknown.

Typical load is at most 7 concurrent users, so the 8-user gap is the one that matters most. In the same window, v3.1 was 6.8% slower than v3.0 at 8 users (373.2 vs 400.4 tok/s) and 2.2% slower at 2 users (230.8 vs 236.1). At 1 user v3.1 is already faster. It is fine to stop after window 1 if the measured levers come in under 3%.

The rules stay as they are for every arm:
- all 24576 experts in VRAM;
- no card lends prompt-path slots;
- 4 of 4 lanes on;
- 12 slots;
- the 1.5 GiB reserve is never touched;
- tests only on 127.0.0.1:8091;
- no deploy without "deploy ja";
- coordinate the window with the docs/bench session first.

## Levers, best first (expected gain relative to effort)

**P1. Layer split 14,26,37 → 13,25,37** (config only, exact)
- **Why:** above 4 users, decode runs on the pipelined batch path. A group-step takes 5.100 ms on v3.1 against 4.716 ms on v3.0 (engine-V.log:882, engine-A.log:880). That is +8.1%, close to the 14/13 ratio of the slowest stage's layer counts. Earlier runs on the 212q engine: 12,25,37 gave 5.72 ms per step against 6.00 ms for 14,26,37, but 13,25,36 gave 5.98 ms, so the rebalance does not always help. CUDA3, which also carries the Q6_K head, may become the next bottleneck. With 13,25,37, CUDA2 becomes the heaviest expert stage (32.13 MB per expert set against 31.74 MB today).
- **Gain:** +2–5% aggregate at 8–12 users, neutral at 1–4 users.
- **Effort:** one window arm, about 4 minutes. It was never run with 12 slots. VRAM estimate: CUDA2 keeps about 3.1 GiB free at the buffer decision, against the 2.23 GiB it needs, so no loan is expected. This is untested.
- **Measure:** the 'strata batch (pipelined …)' ms per group-step and the short aggregate at 1/4/8 users, against a 14,26,37 arm in the same window. Accept only if the log confirms all three rule points and greedy output is identical.

**P2. Q5_0 down kernel (K1)** (iq_kernels.cu only, rebuild, intended bit-exact)
- **What:**
  - add `S27<6>`/`Split<6>`: load = qs/qh words + d; apply = the existing dp4a chain and `d5*d8*(float)(sumi-16*sumu)`;
  - set `kSplit<6> = true`;
  - add `TD == 6` to the 8-lane branch at :2300, so it uses `row_dot_40_sub8`;
  - give d_type 6 `d_rows = 32` at :6486.
- **Why:** `Fmt<6>` gives K = 40, the same row shape as IQ4_NL. Today the type takes the generic `native_down_kernel<6>` (:4301, :2205-2229):
  - one warp per row, 8 rows per block;
  - the weights are decoded again for every entry;
  - activations are read from global memory;
  - only 62.5% of lane slots are used;
  - 4x the blocks of the IQ4_NL path.

  v3.0 had only 5 layers on this path; v3.1 has 33. Upstream has the same gap, and there is nothing to port from there.
- **Gain:** +1–4% at 1–8 users. The gain is largest in lane and solo windows (2–5 entries per group, where decode-once applies) and smaller in 1-row batch groups. It is not enough to close the whole 8-user gap; part of that gap is the split.
- **Effort:** about 40–80 lines plus parity cases. Optional second step, also exact: hoist `sumu` or pre-subtract 16 with `__vsub4`, which halves the dp4a count.
- **Exactness:** the apply is a single product, like IQ4_NL, and `row_dot_40_sub8` already rebuilds row_dot's xor-16/8 addition tree. It still has to pass parity, because of the Q5_1 FMA precedent (:909-911). `kSplit<6>` also switches `launch_gu`/`launch_mmvq` for Q5_0, so parity-check those too.
- **Go/no-go:** in window 1, measure the down-phase GB/s of Q5_0 layer 0 against IQ4_NL layer 7 at the same shape. Write the patch only if Q5_0 reaches at most about 80% of IQ4_NL.

**P3. Q6_K dense small-batch tables for sm_86** (L2/D1–D3/K2 merged; env-only first; bit-exact)
- **Why:** v3.1's dense attention projections and shared-expert gate/up are Q6_K; v3.0 was all Q8_0. The sm_86 Q6_K interleaved-table row (`kIlRows_86`, native_mmvq.cu:3680) was copied from other cards and never measured on a 3090. It also predates the B6 sm_86 default (195d3791 vs 2725c23f).
  - There is no sm_86 table for 5–8 columns, no 1-column Q6_K row, and the hoisted-load (PF) variants have a table only for sm_80.
  - `STRATA_B6_MMVQ_ROWS=1` was tuned with Q8_0 dense weights.
- **Gain:** 0–2% decode, mostly at 1–4 users.
- **Effort:**
  - env only: `STRATA_MMVQ_IL_ROWS` for Q6_K at 2–4 columns, `STRATA_MMVQ_IL_PF=3/4`, `STRATA_B6_MMVQ_ROWS=4/8`;
  - K2 (adding `{86, kIlRows58_86}`) only if the 5–8 column cells come out mostly non-zero.
- **Measure:** `mmvq_il_parity --bench --emit-table --pf 0,3,4` and `b6_mmvq_bench`. Promote a cell only if it gains at least 3% at 2–5 columns on the qkv/gate/q shapes.

**P4. Opt-in expert layouts: `STRATA_EXP_MODE=8`, `STRATA_EXPDEC=1`** (env only; documented bitwise)
- **Why:** neither was ever measured on sm_86. Both replace the GU_SWIGLU fusion. Neither covers Q5_0 down, and EXPDEC also skips the IQ4_XS gate/up.
- **Gain:** unknown, about −3% to +4%.
- **Effort:** bench rows only. Q5_0 and IQ4_XS instances (L3) are worth writing only after P2, and only if the bench shows the gate/up side winning.

**P5. `STRATA_IQ_STAGE_TABLES=1`**
- One bench row only. This also turns on grid staging for IQ3_XXS. It was 3–13% slower on IQ3_S on a 3090 Ti, so the most likely result is ≤ 0.

## Excluded

| Lever | Reason |
|---|---|
| Packed Q6_K for dense (L4) | Needs extra VRAM against the reserve, or an in-place repack that breaks prompt-path readers; no evidence of gain |
| `STRATA_HC_Q8` | Not bit-exact. Gain unknown, ~0.45 GiB/card VRAM saved. Only with explicit owner OK plus identity and eval gate |
| Upstream v0.2.13 / origin/main kernel ports | No decode kernel change for these types; device code identical |
| Upstream launcher defaults | Already set or deliberately pinned |
| Inert switches | ADAPT_*, EXPERT_V2, KQ_SPLIT, TSUM, Q8/Q6_PACKED, PF_FUSED have no effect here or break the VRAM/exactness rules |
| MTP acceptance, verify-window time, prompt path | Already at parity |
| 16-user, long and multiturn gaps | Caused by 12 vs 16 slots, which stays as is |

## GPU windows (8091 only; production restored at the end; no deploy)

**Prep (no GPU, about 1 h)**
- Write `linux/window-kern.sh`, copied from `window-v31.sh`: same isolation, guard and restore timer. Phase A runs as GPU_TESTS steps; the phase C arms are built from `configs/T16-212q-v31-prod262r-vision-cpu.json`.
- Binaries come from `/media/raid0/StrataQ38FN/Strata-212p/build`. They have the same decode kernels as production; the only change since is a test file.
- Shard: `/media/raid0/Models/Strata-GGUF/unsloth-UD-IQ4_XS-v31/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf`.

**Window 1 (no rebuild, about 27 min of production off)**

- **T+0:** stop strata.service and wait until the GPUs are free.
- **T+0:30 to T+8, phase A: micro-benches, one per card in parallel.**
  - **CUDA0, `native_expert_bench` with `STRATA_GU_SWIGLU=1`**
    - Layers 0,4,7,8,9,30 cover all six v3.1 type pairs. Weight the results 20/13/8/2/1/4.
    - Shapes:
      - G=20, T=1, `BENCH_CAP=20` (batch);
      - G=30, T=5, cap 50 (lane);
      - G=45, T=1, cap 50.
    - Modes 0/8 and 0/7 (P4).
    - 4 alternating runs with `STRATA_EXPDEC` unset / =1.
    - One run with `STRATA_IQ_STAGE_TABLES=1`.
    - The down-phase GB/s of layer 0 (Q5_0) against layer 7 (IQ4_NL) gives the P2 go/no-go.
    - v3.0 shard 00002, layers 0,2,4, modes 0/0, as a reference.
  - **CUDA1:** `b6_mmvq_bench 500 0 {1,4,8}`.
  - **CUDA2:** `MMVQ_IL_ONLY=Q6_K mmvq_il_parity --bench --emit-table --pf 0,3,4`. It must end with OK.
  - **CUDA3:** `q8_gemv_bench --cols 1..8 --bw 936` as the Q8_0 reference. Optionally `ncu --section SpeedOfLight` on native_down at layer 0 vs layer 7; skip it if profiling permission is denied.
- **T+8: decision rules.**
  - An expert switch qualifies only with an identical hash, 0 differing, and at least 5% lower weighted expert time.
  - A dense cell qualifies with at least 3% at T 2–5.
- **T+9 to T+25, phase C: engine arms, about 4 min each.**
  - Each arm runs `bench_concurrency --levels 1,4,8 --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0`, then the placement/loan/lanes/VRAM greps.
  - **C0:** v3.1 production config plus `STRATA_DISPATCH_REPORT`.
  - **C1:** C0 with `layer_split` 13,25,37 (P1).
  - **C2:** C0 plus the phase A winners (`STRATA_MMVQ_IL_ROWS` string / `B6_MMVQ_ROWS` / `EXP_MODE`). Drop this arm if nothing qualified.
  - **C3:** C0 plus `STRATA_VERIFY_PROFILE=1`. This gives the per-stage ms table: whether S1 or CUDA3 is the bottleneck, and the split between routed experts and Q6_K dense. It sizes P2 and P3. Use it for attribution only; the profiler turns off some fusions, so it is not a speed arm.
- **T+25:** start production and verify that it loaded.
- **Afterwards (no GPU)**
  - Exactness: the per-request sha256 in `sweep.requests.jsonl` must match C0. Also compare C0 with the 11-09 V run, to see whether greedy output is repeatable across runs at all.
  - Speed: 8-user aggregate, per-user decode p50, rows/s per group-step.
  - A single run resolves about 1.2%. Only a gain of at least 3% earns an ABBA follow-up. A split change then goes into `deploy-v31.sh`, which fixes `SPLIT=14,26,37` today, and only after "deploy ja".

**Window 2 (only if P2 is a go)**
- Prep without a GPU: implement K1 in a separate build directory of 212q50, CUDA 13.0, same CMakeCache. Build `strata`, `iq_multi_parity`, `native_grouped_parity`, `native_expert_parity` and `native_expert_bench`.
- **Parity:** Q5_0 down at n_ff 640, T=1..8, 1–6 entries per group, old against new, memcmp equal. Include the `launch_gu`/mmvq Q5_0 paths. Any differing byte means the patch is dropped or reworked.
- **Kernel time:** `native_expert_bench` layers 0,4,7 at the phase A shapes, before and after.
- **Engine:** ABAB on 8091 against the current engine, 1/2/4/8/12 users, greedy replies byte-identical, on the winning split from window 1.
- **Then:** the optional `sumu`/pre-subtract-16 variant, with the same gates. Q5_0 and IQ4_XS instances for mode 8 / EXPDEC (L3) only if phase A showed the gate/up side winning.
- An upstream PR for the Q5_0 Split is possible, as havok2-htwo noreply, with your approval of the text before posting.

All the evidence above comes from the window results in `/media/raid0/StrataQ38FN/linux/results/v31-1109-2004/` (cfg/, engine-V.log, engine-A.log, A-short.txt, window.log) and from `/media/raid0/StrataQ38FN/Strata-212q50/src/kernels/cuda/iq_kernels.cu` and `native_mmvq.cu`.