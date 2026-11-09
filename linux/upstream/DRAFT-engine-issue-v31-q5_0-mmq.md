# DRAFT upstream issue (not posted): CUDA prompt reads of UD-IQ4_XS-v31

Status: draft only. Posting needs the owner's text approval and goes out as havok2-htwo (noreply).
Source: window v31 (linux/results/v31-1109-1754, 2026-11-09) and a read-only code investigation of Strata-212p + origin/main 504b1433.

---

**DRAFT upstream issue — do not post.** It still needs your approval as text and must be posted as havok2-htwo.

**Title:** CUDA: UD-IQ4_XS-v31 prompt read ~2.4x slower than v3.0 — Q5_0 down layers (33/48) leave MMQ because STRATA_ORCA_Q4KS_MMQ is OFF by default

**Body:**

> On CUDA (4x RTX 3090, sm_86), Unsloth UD-IQ4_XS-v31 reads prompts about 2.4x slower than UD-IQ4_XS v3.0 on the same engine. Decode is unchanged.
>
> **Repro** (one binary built from v0.2.12 with patches; the gating code below is identical on origin/main 504b1433). Both packs are fully VRAM-resident, and no card lends prompt buffers.
>
> | | v3.0 | v3.1 |
> |---|---|---|
> | cold prompt, 151522 tokens | 25413 ms (5962.3 tok/s) | 60991 ms (2484.3 tok/s) |
> | decode p50, 1 user | 145.6 tok/s | 150.1 tok/s |
> | 16 users, ~1K-token prompts, agg | 418.6 tok/s | 238.9 tok/s |
>
> **Engine log, v3.1:**
> ```
> strata: prompt MMQ reads resident experts in place (STRATA_MMQ_DIRECT=1): 15 of 48 layers
> strata: prompt MMQ sorts the full expert groups (STRATA_MMQ_GROUP_SORT=1): 15 of 48 layers
> strata serve: layer split: 100% of the experts resident, the prompt path's streamed ring 96 slots
> ```
> (v3.0 prints 48 of 48 for both lines.)
>
> **Per-layer expert type mix (gate/up, down):**
> - v3.1: IQ3_XXS/Q5_0 ×20, IQ4_XS/Q5_0 ×13, IQ3_XXS/IQ4_NL ×8, IQ4_XS/IQ4_NL ×4, IQ2_S/Q2_0 ×2, IQ2_S/IQ4_NL ×1
> - v3.0: IQ3_S/IQ4_NL ×43, IQ3_S/Q8_0 ×4, IQ4_XS/Q8_0 ×1
>
> The 15 layers that pass are exactly the layers whose down is not Q5_0.
>
> **Cause:**
> - `mmq::supported()` accepts `GGML_TYPE_Q5_0` only under `#ifdef STRATA_ORCA_Q4KS_MMQ` (`src/prefill/moe_mmq.cu`). The option defaults to OFF.
> - `fits()` rejects the type before the "no tile" message, so nothing is logged.
> - `mmq_plan()` needs both products of a layer to fit. So the 33 Q5_0-down layers leave MMQ entirely, gate/up included.
> - On sm_86 the fused native path is off by default, and on CUDA it has no Q5_0 down. These layers therefore go to per-expert dequantize-to-FP16 + cuBLAS.
> - The DIRECT and GROUP_SORT counts only count MMQ-plan layers, which is why the log shows 15/48.
>
> **Suggested fix:**
> - Enable the Q5_0 MMQ instance by default on CUDA, or under a type-specific option. The pinned ggml already has Q5_0 MMQ, and its Ampere config is the same as IQ4_NL's.
> - Optionally add `q5_0` to `_strata_mmq_tab_types` / `run_tab` so these layers are read in place as well. The down over-read at 640 columns is 88 B, under the 256 B slot guard. `tests/cuda/prefill_mmq_tab_test.cu`, which currently asserts that Q5_0 has no pointer table, would need updating.
> - Until then, a note on CUDA prompt speed next to `UNSLOTH_V31_ENGINE` / the v3.1 model entries would help.
>
> AMD already has a fix for this cost (63f06385 for gfx1151, e77bd43b for gfx12). I have not yet run the `prefill_mmq_v31_test --bench` comparison with the flag ON, and I can post those numbers.

---

Notes:
- The two arms used different `--layer-split` values (12,25,37 vs 14,26,37). Neither lent prompt buffers, so I don't think this affects the result, but it is the one difference beyond the pack.
- I left the "~5 GB FP16 traffic per layer per chunk" figure out because it is only an upper bound: each expert is dequantized once per super-chunk.

Files:
- `/media/raid0/StrataQ38FN/linux/results/v31-1109-1754/summary.txt`
- `/media/raid0/StrataQ38FN/linux/results/v31-1109-1754/engine-VR.log`
- `/media/raid0/StrataQ38FN/linux/results/v31-1109-1754/engine-A.log`
- `/tmp/claude-1000/-media-raid0-StrataQ38FN/7178d4be-6501-48a4-a47e-407568957fcc/scratchpad/sim.py` (re-runs the per-layer predicate)

---

## Analysis (for us)

**Root cause (verified)**

The production binary `/media/raid0/StrataQ38FN/Strata/engine/strata-212q-9743c131` (sha256 9743c131…, the same file as `/media/raid0/StrataQ38FN/Strata-212p/build/strata`) was built without the Q5_0 prompt-path MMQ instance:

- `/media/raid0/StrataQ38FN/Strata-212p/build/CMakeCache.txt:1011` has `STRATA_ORCA_Q4KS_MMQ:BOOL=OFF`. That is also the default (`CMakeLists.txt:142`).
- `nm -C` on the binary shows no `mul_mat_q_case` or `mul_mat_q_tab_case` for type 6 (Q5_0).
- `mmq::supported()` (`src/prefill/moe_mmq.cu:322-325`) accepts Q5_0 only under `#ifdef STRATA_ORCA_Q4KS_MMQ`. The launch case at `moe_mmq.cu:515-517` has the same guard.
- `fits()` (`moe_mmq.cu:350`) returns false when `!supported`, before the #420 "no tile" message. So the log says nothing about these layers.
- `mmq_plan()` (`src/prefill/prefill.cpp:1537`) needs both `fits(gate/up, 1280)` and `fits(down, 2560)`. Otherwise the whole layer gets `p.fallback`, so both products leave MMQ.
- All 33 v3.1 layers with a Q5_0 down (20 × IQ3_XXS/Q5_0 + 13 × IQ4_XS/Q5_0) leave MMQ entirely. Their IQ3_XXS/IQ4_XS gate/up leave with them.
- The DIRECT and GROUP_SORT counts both skip `!p.layer[l]` (`prefill.cpp:1588`, `1612`; prints at `1602`, `1617`). So "15 of 48" is a symptom, not the cause.
- Re-running the predicate on the pack's `native_experts.txt` gives exactly 15/48 for v3.1: layers 7, 8, 9, 10, 11, 14, 21, 23, 30, 32, 34, 35, 42, 43, 44, which are the layers whose down is not Q5_0. For v3.0 it gives 48/48. This matches `engine-A.log` (48/48) and `engine-VR.log` (15/48).

**What fails, per layer**

| layer pair (gate/up, down) | v3.1 count | MMQ plan | DIRECT (in place) | GROUP_SORT |
|---|---|---|---|---|
| IQ3_XXS, Q5_0 | 20 | fail (`supported(Q5_0)` = false) | skipped | skipped |
| IQ4_XS, Q5_0 | 13 | fail | skipped | skipped |
| IQ3_XXS, IQ4_NL | 8 | pass | pass | pass |
| IQ4_XS, IQ4_NL | 4 | pass | pass | pass |
| IQ2_S, Q2_0 | 2 | pass | pass | pass |
| IQ2_S, IQ4_NL | 1 | pass | pass | pass |

v3.0 for comparison: IQ3_S/IQ4_NL × 43, IQ3_S/Q8_0 × 4, IQ4_XS/Q8_0 × 1. All 48 pass.

Where the 33 layers go instead:
- **Not the fused kernels.** `fused::native_supported()` (`src/prefill/moe_fused_iq.cu:2041-2086`) is off on sm_86 by default; by default it is on only for cc 12.0 cards with 80+ SMs. On CUDA, `d_covered` (`:1912`) covers only Q2_0 and IQ4_NL as downs. Q5_0 is handled only under `__HIPCC__`.
- **They take the FP16 path** (`prefill.cpp:6046-6074`, layer-major at `7286-7290`). For each routed expert: `iq_dequant_gu_f16` and `iq_dequant_f16` to FP16, a cuBLAS FP16 GEMM for gate/up, `swiglu_interleaved`, and a cuBLAS FP16 GEMM for down. The dequant ring has 2 entries (DQ=2, `prefill.cpp:459`). The weights are read from resident slots: 100% of experts were resident and no card lent prompt buffers.
- **Dense Q6_K projections are not a cause.** On CUDA, Q6_K and Q8_0 both go through `Gemm::native` (`src/prefill/gemm.cu:1353`): `dequant_f16` in `src/kernels/cuda/dequant_bf16.cu`, same template, then cuBLAS FP16. Dense MMQ is HIP-only (`gemm.cu:1258`).
- **Decode is unaffected.** It does not use this path.

**Measured impact** (window v31w11091754, 4 × RTX 3090, same binary)

- A = v3.0, `--layer-split 12,25,37`. VR = v3.1, `--layer-split 14,26,37` plus `STRATA_SPLIT_OWN_RESERVE_MIB=1024`. No prompt-buffer loans in either arm.
- Cold read of 151522 tokens: A takes 25413 ms (5962.3 tok/s), VR takes 60991 ms (2484.3 tok/s). That is 2.40× slower. Client TTFT is 25.58 s vs 61.16 s.
- Decode p50 with 1 user: 145.6 (A) vs 150.1 (VR) tok/s, so equal.
- Multiturn wall time: 91.7 vs 163.7 s. Follow-up TTFT p50: 1.72 vs 4.37 s.
- Long prompts, 16 users: 418.6 vs 238.9 agg tok/s (−42.9 %). TTFT p50/p95: 1.84/6.79 vs 8.07/28.79 s.
- Short prompts, 16 users: 538.0 vs 334.2 agg tok/s (−37.9 %).

**Fix options** (none built or measured)

1. **Build flag only: `-DSTRATA_ORCA_Q4KS_MMQ=ON`.**
   - Effort: rebuild, then `prefill_mmq_v31_test --require-q5_0` and `prefill_mmq_v31_test --bench`, then one GPU window.
   - Risk: low. The kernel already exists in the pinned ggml (`mmq-instance-q5_0.cu`, `load_tiles_q5_0` at `mmq-load-tiles.cuh:313`, the same D4 `vec_dot_q8_0_q8_1_mma` as IQ4_NL). Its Ampere config (`mmq-config-ampere.cuh:70-85`) is identical to IQ4_NL's. Numbers change at rounding level only, because activations go through q8_1 instead of FP16, as on v3.0.
   - Expected: the MMQ plan covers 48/48. DIRECT stays 15/48 (the other 33 use a gather copy). GROUP_SORT 48/48 is a prediction: `whole_tiles(Q5_0)` was not checked.
   - Binary and VRAM cost is about 1 MB per GPU (an estimate, not measured).
2. **Option 1 plus Q5_0 in the pointer-table (in-place) allowlist** (about 10 lines).
   - Changes:
     - `CMakeLists.txt` near line 2538: `if(STRATA_ORCA_Q4KS_MMQ) list(APPEND _strata_mmq_tab_types q5_0) endif()`.
     - Add a guarded `STRATA_TAB_CASE(GGML_TYPE_Q5_0)` to `run_tab` (`moe_mmq.cu:263-281`).
     - Update `tests/cuda/prefill_mmq_tab_test.cu:536-541`, which currently fails the test if Q5_0 has a pointer table. Add a Q5_0 2560×640 down shape with an expected over-read of 88, and add the pairs IQ3_XXS/Q5_0 and IQ4_XS/Q5_0.
   - Checks that hold by arithmetic: the over-read is 4 × 22 = 88 B, under `kSlotGuard` = 256. The blob sizes match the [gate|up|down] layout: 2,380,800 and 2,867,200.
   - Risk: low to moderate, because it is a new instance. About 4 MiB per GPU (estimate). Expected effect: DIRECT 48/48.
3. **A new kernel is not needed.** The other route would be a CUDA port of upstream's gfx12 fused Q5_0 down (e77bd43b). That is much more work, and fused native is off by default on sm_86 anyway. Not recommended.

**Env switch that helps now:** none. The gate is compile-time.
- `STRATA_PF_FUSED=1` would not help: on CUDA the fused path has no Q5_0 down.
- `STRATA_PF_FUSED_Q50` is gfx12 (HIP) only.
- The only immediate mitigation is to keep production on v3.0 with this build.

**Upstream status** (origin `Niko1221/Strata` fetched read-only, origin/main 504b1433 = v0.2.13.1-1)

- Not fixed.
  - The option is still OFF (`CMakeLists.txt:154`).
  - Q5_0 is behind the same ifdef (`moe_mmq.cu:327-329`, `519-521`).
  - The pointer-table list still has the same nine types (`cmake/backends/cuda.cmake:194`).
  - Between v0.2.12 and origin/main, the only commit that touches the MMQ sources is 2351f2cc (Q2 blob geometry). It does not change which types are accepted.
- Upstream fixed the Q5_0-down prompt cost on AMD only:
  - 63f06385: gfx1151, on by default.
  - e77bd43b: gfx12, opt-in `STRATA_PF_FUSED_Q50=1`; on R9700, prompt +44 % at 2K, +40 % at 8K, +13 % at 32K.
- Upstream's own `tests/cuda/prefill_mmq_v31_test.cpp` (5d56ff8c) documents the CUDA limitation. But `setup.py:434` (`UNSLOTH_V31_ENGINE = (0, 1, 57)`) and the v3.1 model entries have no caveat about prompt speed.
