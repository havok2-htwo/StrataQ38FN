# Kernel analysis: short-prompt prefill ("dequant" = expert copy) (2026-10-11)

## What "dequant" is

On this pack and setup, the "dequant" stage does no dequantization at all. It is a device-to-device memcpy of every routed resident expert's raw GGUF blocks into a contiguous MMQ group buffer.

Path for a chunk of T < 1024 tokens with all experts resident:
- `stream_all` is false (prefill.cpp:2198 needs `T >= stream_all_min()`, which is 1024, prefill.cpp:143-147). The layer therefore takes the host-grouped MMQ path.
- `use_mmq` is true for all 48 layers. MMQ covers IQ3_S, IQ4_NL and Q8_0 (moe_mmq.cu `supported`/`fits`, ~118-160; plan in prefill.cpp:922-948).
- The resident loop in prefill.cpp:3694-3712 calls `compute(j, cache->device_slot(slot), -1)` once per routed expert.
- `compute` (prefill.cpp:3627) starts with `pt.mark(kPfDequant)` at prefill.cpp:3629.
- In the non-streamed walk `group_gather` is false (prefill.cpp:3596 needs `stream_all`). Each expert therefore gets its own `mmq::gather_native(blob, blob+up_off, gub/2, blob+down_off, db, grp_gu+q*gub, grp_d+q*db)` (prefill.cpp:3642).
- `gather_native` is `copy16_kernel` (moe_mmq.cu:236-245). It copies gate|up, 1280x2560 IQ3_S = 1.41 MB, and down, 2560x640 IQ4_NL = 0.92 MB (Q8_0 = 1.74 MB), into slot q of a 16-expert group buffer.
- Only after the 16th expert does the mark switch to kPfGemmGU. Then come two memsets for MMQ_TAIL, the MMQ gate/up, swiglu (still charged to gemm gate/up), and the kPfGemmD quantize and MMQ down (prefill.cpp:3650-3672).

Why the copy exists: llama.cpp's `mul_mat_q` with ids addresses expert z as `x + z*stride_channel_x` (mmq.cuh:1053/1147/1231: one base pointer, stride counted in whole blocks). Resident experts sit at arbitrary `ExpertCache::device_slot()` addresses (expert_cache.cpp:658). Slots are rounded to 256 B (expert_cache.cpp:568), and 2,329,600 B is not a multiple of the 110 B IQ3_S block or the 18 B IQ4_NL block. The blob is already [gate|up|down] with gate and up contiguous (native_expert.cpp:73-74), so the copy only rebuilds a fixed-stride array.

Cost: per layer about 400-490 distinct experts × 2.33 MB, read plus write, so roughly 2 GB of DRAM traffic. That is about 2 ms per layer at the 3090's ~930 GB/s, which matches the measured 24-28 ms over 12 layers. MMQ then reads the same bytes a third time, because a 16-expert group is ~37 MB and does not fit the 6 MB L2. Per layer there are also ~500 copy launches plus ~30 groups × 6 launches.

## Why MMQ / STRATA_PF_FUSED do not help

MMQ is already on, and it is the cause rather than a cure. llama.cpp's MMQ needs the experts as one array with a fixed per-expert stride. For resident experts in the cache, that array has to be built by copying every routed expert's full blob each chunk (the 'dequant' stage). The only path that avoids the copy here is the FP16 path (STRATA_PREFILL_MMQ=0), which instead dequantizes each expert to FP16 with iq_dequant_gu_f16/iq_dequant_f16 (prefill.cpp:3677-3681). That is worse. STRATA_MMQ_RESIDENT_SORT_NE=1 is excluded for native packs (mmq_resident_sort.hpp:25 returns false when `native_layout` is set).

STRATA_PF_FUSED=1 changes nothing for three reasons:
1. The fused experts run only inside the streamed walk. `fused_nat` and `fused_l` need `stream_all` (prefill.cpp:2881-2882), and `stream_all` needs a chunk of at least `stream_all_min()` = 1024 tokens plus a ring of more than 8 slots (prefill.cpp:2198, 143-147, 291). With --prefill 512 no chunk ever qualifies, so every layer stays on MMQ plus gather. The fused path would read `cache->device_slot()` directly, with no gather, and group on the GPU with no host sync.
2. Even at 1024 tokens or more, the 5 layers with Q8_0 downs are not covered on CUDA. `d_covered` accepts only Q2_0 and IQ4_NL; Q8_0 is HIP-only behind STRATA_PF_FUSED_KQ (moe_fused_iq.cu:1003-1009, 1038-1044). Because of those layers, `fused_ring()` is false (prefill.cpp:219-236), so the pack keeps MMQ's buffers and ring.
3. The fused kernels use 64- or 128-row tiles (`pick_ww`, moe_fused_iq.cu:1021-1035). With about 3 rows per expert, ~95% of each tile's MMA work is padding.

The decode and verify expert kernels already do what this case needs. `native_expert_grouped` (iq_kernels.cu:3446, used by verify.cpp:1433 and peer_experts.cpp:239) takes a per-group blob pointer table (`grp_ptr`, which can be the cache slots directly), `grp_start`, and per-entry token and destination rows. It decodes each weight row once per 4 entries (GRP_NC=4, iq_kernels.cu:1239). It covers IQ3_S gate/up and both IQ4_NL and Q8_0 down (STRATA_GU_FMTS/STRATA_D_FMTS, iq_kernels.cu:611-615), so it handles all 48 layers. The prefill path never calls it.

## Options

### A. Small-row experts on the verify/decode grouped MMVQ kernel (per-expert threshold)

- Mechanism: After host grouping, split the routed experts by row count. Experts with cnt <= R (default 8, env-tunable, 0 = off) go to strata::kernels::native_expert_grouped, reading the resident cache slot pointers directly with no gather. Experts with cnt > R, and any non-resident or streamed expert, stay on today's MMQ plus gather. At 154 tokens (λ≈3), about 99.6% of experts are <= 8 rows, so the gather and the 16-expert MMQ groups almost disappear. The kernel decodes each IQ3_S/IQ4_NL/Q8_0 row once per 4 entries, so weight traffic is about one read of each routed expert (~2.33 MB) instead of copy + write + MMQ re-read. The whole layer takes ~4 launches instead of ~650.
- Changes: prefill.cpp, in the host-grouping block around 3120-3165:
- When assigning `m.off`, put the small resident experts first so their rows form one contiguous block [0, rows_small). `native_expert_grouped` uses grp_start[g+1] as group g's end, so its groups must be adjacent.
- Build per small expert: grp_ptr = cache->device_slot(host_res[l*NE+e]), grp_start = off, and a device n_groups. Upload these with the bounds through the mapped grp_host buffer or pinned memcpy.
- `order` and `bounds_host` then cover only the big experts, whose rows start at rows_small.

Before the compute loop:
- Run quantize_q8_1_rows(m.mixed, T, N, xq_tok, cs). This is per token, T*2880 B; add a small buffer in moe_set_bytes.
- Call native_expert_grouped with ent_tok = m.src_dev, ent_dst = m.ids_identity, out = m.Dm, cap_groups = n_small, cap_entries = rows_small.

iq_kernels.cu/.hpp: add an overload that takes separate gate/up/h/hq scratch pointers, so it can reuse m.GU (exactly 2×T*K*640 floats), m.H and m.Hq instead of one contiguous 3×cap×n_ff×4 + hq scratch.

Also:
- Add a timing phase, e.g. 'experts small'.
- Add env STRATA_PF_SMALL_ROWS (and 0 for today's bytes).
- moe_combine is unchanged: Dm rows still hold unweighted outputs addressed through slot_dev.
- Expected per card: 154 tokens, 12 layers: MoE expert work goes from dequant 24-28 + gate/up 16 + down 9 ≈ 49-53 ms to roughly 15-22 ms. That assumes ~400-490 experts × 2.33 MB ≈ 0.9-1.1 GB per layer at a realistic 600-750 GB/s for MMVQ-style IQ3_S kernels. Stage GPU time would go from 78-87 to ~48-58 ms, and the 4-stage admission from ~330 to ~200-230 ms. At 512 tokens (λ≈10), only about a third of experts are <= 8 rows, so the gain is partial: maybe -10..-20 ms per stage. Re-measure with R=16 or 32. More passes are cheap while a block's 16 weight rows (17.6 KB) stay in L1/L2.
- Exactness: Not bitwise equal to today's MMQ prompt path. The int8 dot products are identical, but the float accumulation order differs, so differences are at rounding level. Per token, the result is exactly what the verify/decode expert kernel computes for the same q8_1 activation, and it does not depend on grouping or grid size (`native_grouped_parity` asserts independence from gridDim.y). STRATA_PF_SMALL_ROWS=0 restores today's bytes. Tests:
- Kernel tests already exist: native_grouped_parity, native_expert_parity, iq_multi_parity.
- New: a prefill test that memcmps Dm rows of the small path against native_expert_grouped run with verify-style inputs on a real UD-IQ4_XS layer.
- End to end: teacher-forced KL of the prompt path vs the STRATA_PREFILL_MMQ=0 FP16 path and vs today's MMQ, the same gate used for STRATA_PF_FUSED (phaseA s18-tf style), plus the release gate prompts.
- Effort: 2-3 (host plumbing 1, kernel overload 0.5, parity, KL and timing 1-1.5) days, risk: Low to medium:
- It reuses kernels already proven in verify on this exact pack and covers all 48 layers, including the Q8_0 downs.
- Risk is in the bookkeeping: row reordering so small groups are contiguous, and the boundary between small and big experts.
- Scratch aliasing with GU/H/Hq must not overlap MMQ's use in the same layer. The small path finishes before the big groups start on the same stream, so this is safe.
- First, check kernel throughput at 3-8 entries per group with the existing bench: native_expert_bench <shard1.gguf> <layer> 450 3 0 0 (needs STRATA_BUILD_TESTS).

### B. MMQ in place: a per-expert pointer table in the vendored mul_mat_q (no gather)

- Mechanism: Patch third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh so mmq_args takes an optional `const char* const* x_ptrs`. In the three tile paths (lines ~1053, 1147, 1231), use x_ptrs[zt] + it*I*stride_row_x instead of x + zt*stride_channel_x. Product gets a gate/up pointer table (blob base) and a down pointer table (blob + down_off), filled from cache->device_slot. The gather and the two MMQ_TAIL memsets go away.

Two variants:
- Keep the same 16-expert groups and order. The arithmetic is then bitwise today's.
- Run one MMQ launch per layer over all experts. Fewer launches and better SM fill, but stream-k can split K differently.
- Changes: - mmq.cuh: mmq_args field, 3 offset_x sites, launch_mul_mat_q pass-through. All mmq-instance-*.cu recompile.
- moe_mmq.cu: Product.ptrs and a run() variant.
- prefill.cpp compute(): fill the pointer tables instead of calling gather_native.
- ExpertCache::open_sized: MMQ reads up to 4 blocks (72 B for IQ4_NL, 136 B for Q8_0) past a 640-wide down row. In place, that is the next slot's IQ3_S bytes, which can decode as Inf/NaN f16 scales, and NaN × 0 gives NaN. Every native slot needs >= 4 KiB of zeroed padding after the blob (today 2,329,600 B is already 256-aligned, so there is zero padding). That costs +0.18% of the arena, ~25 MB per card.
- Expected per card: Removes all of the 24-28 ms dequant stage and the per-group memsets. Gate/up plus down stays about 20-25 ms, maybe less with one launch per layer. MoE ≈ 20-25 ms per stage at 154 tokens. At 512 tokens it is better than A, because tensor-core MMQ wins at 10-25 rows per expert: dequant 26-55 ms removed.
- Exactness: Variant with unchanged 16-expert groups: same weight bytes, same J config (same max_rows), same tile count, so the Dm output should be bytewise identical to today. Verify with a memcmp of Dm, old gather vs pointer table, on real layers, plus the existing MMQ tests (prefill_fused_iq_test's MMQ reference). The per-layer single-launch variant changes rounding only (stream-k partition); gate it with KL.
- Effort: 2-3 (patch, padding change and test). Ongoing cost of carrying a patch in vendored llama.cpp on every pin bump. days, risk: Medium:
- Patch to vendored llama.cpp code.
- The read-past/NaN hazard needs the slot padding fix in ExpertCache (shared by peer_experts and remote_experts, which also call open_sized).
- int offset_x stays small, because the per-expert pointer removes the large channel term.
- Still about 3x slower than A per byte at 3 rows (J=8 tiles, ~60% padding), but compute is not the bottleneck.

### C. Fused native int8 experts below 1024 tokens (config now, code later)

- Mechanism: The existing STRATA_PF_FUSED=1 native kernels (moe_fused_iq) read cache slots directly through fused::Batch, with no gather. They also group on the GPU (fused::group), so they skip the per-layer host sync. They run only in the streamed walk. Forcing that walk with STRATA_PREFILL_STREAM_MIN and a small STRATA_PREFILL_RING puts 43 of 48 layers on them now; the 5 Q8_0-down layers stay MMQ with group_gather (one launch per 16 experts). A code follow-up would be:
(1) allow fused_nat without stream_all for the all-resident case;
(2) a 16-row tile (ww) for small average row counts;
(3) Q8_0 down on CUDA (d_covered).
- Changes: Config only:
- STRATA_PF_FUSED=1 STRATA_PREFILL_STREAM_MIN=1 STRATA_PREFILL_RING=16.

Code follow-up:
- prefill.cpp:2881-2882 and the fused_layout/fused_ring sizing.
- moe_fused_iq.cu: pick_ww with a 16-row tile, plus CUDA Q8_0 down kernels.
- Expected per card: Unmeasured and uncertain. The dequant stage disappears for 43/48 layers and host grouping goes too. But the fused kernels pay a full 64-row tile per expert. The 5070 numbers in moe_fused_iq.cu (IQ3_S, 2048 tokens, also about one tile per expert: 5.6 ms per layer) suggest ~3-5 ms per layer on a 3090, which could be no better than today's ~4.2 ms. With a 16-row tile it is plausibly ~2 ms per layer.
- Exactness: Output changes at rounding level, as documented for chunks below ~1000 tokens and for fused vs MMQ. Covered by prefill_fused_iq_test (FP64 reference and MMQ); end to end needs the KL gate.
- Effort: 0 for the config test (one timing run). 4-6 for the tile and coverage work. days, risk: Config:
- The ring and fused buffers borrow cache slots (16 × max_blob ≈ 50 MB per stage), so a few experts may stream again.
- Needs a non-null ExpertSource on the stage and no --peer-device (core::peer_portable).
- Stream-min changes how every chunk walks experts.

Code:
- Larger kernel work, and Q8_0 is uncovered on CUDA today.

## Analyst recommendation

Build option A first.

1. Route every resident expert with up to R rows (default R=8, the same small-batch limit llama.cpp uses for its MMVQ kernel) through `native_expert_grouped`. This is the kernel verify already runs for this exact pack, and it reads the cache slots directly. The rest of the layer stays on MMQ.
   - Lay the small experts' rows out first, so the kernel's grp_start[g+1] end convention holds.
   - Alias its scratch onto GU, H and Hq through a separate-pointer overload.
   - Add a T×2880 B per-token q8_1 buffer.
   - Keep STRATA_PF_SMALL_ROWS=0 as today's exact bytes.

   For a 154-token admission this removes the 24-28 ms D2D copy, the triple read of each expert and ~650 launches per layer. The expected MoE cost is ~15-22 ms per stage instead of ~50, so a stage takes ~50 ms instead of ~80 and the 4-stage admission ~200-230 ms instead of ~330. Before writing the host code, measure the kernel at 3-8 entries per group with native_expert_bench (G≈450, T=3).

2. Then, for 512-token chunks, either raise R if the bench shows the grouped kernel still ahead at 10-16 rows, or add option B for the big experts. B keeps today's bytes when the 16-expert grouping is kept. It needs ≥4 KiB of zeroed slot padding so MMQ's read past the down rows cannot hit NaN-decoding bytes.

3. Run the option C config experiment now, since it is free. Treat the extended fused-kernel work (16-row tile, CUDA Q8_0 down) as lower priority. It is the most code, and its 64-row tiles waste most of their work at ~3 rows per expert.

Once the per-stage MoE cost is down, revisit overlapping the 4 serial stages with micro-chunks. That is what g3-overlap-spike is about. Today it barely helps, because each micro-chunk still touches ~80% of the experts.

## Config-only switches

- Try STRATA_PF_FUSED=1 STRATA_PREFILL_STREAM_MIN=1 STRATA_PREFILL_RING=16. This is the only switch combination that can remove the dequant stage today. Lowering the stream minimum forces the streamed walk below 1024 tokens (prefill.cpp:143-147, 2198), so fused_nat turns on for the 43 IQ3_S/IQ4_NL layers. They then read cache slots directly with GPU-side grouping: no gather, no host grouping sync. STRATA_PREFILL_RING=16 keeps the ring small (16 × max_blob ≈ 50 MB per stage) instead of the ~0.5 GB byte-budget ring. The 5 Q8_0-down layers stay on MMQ with group_gather. The gain is uncertain because the fused kernels use 64-row tiles. Output changes at rounding level, so check KL. Requires no --peer-device and a stage that has an ExpertSource.
- STRATA_PF_FUSED=1 alone does nothing for chunks below 1024 tokens with --prefill 512. That matches the logs.
- STRATA_MMQ_RESIDENT_SORT_NE=1 does not apply. Its eligibility check returns false for native layouts (mmq_resident_sort.hpp:25).
- STRATA_PREFILL_GROUP_GATHER: on by default, but it only affects the streamed walk. With the STREAM_MIN experiment it turns the Q8_0 layers' per-expert copy launches into one launch per 16 experts. The same bytes are still copied.
- STRATA_PREFILL_MMQ=0 is worse. It swaps the copy for FP16 dequantization plus a cuBLAS GEMM per expert (prefill.cpp:3677-3690).
- STRATA_EXPERT_V2=1 (optionally with STRATA_TSUM=1) is the S26 grouped-expert kernel tuned for exactly IQ3_S/IQ4_NL at 2560/640 and documented as bitwise equal to the default (iq_kernels.cu:2005-2008). Today it only affects decode and verify, and its kernels are AMD-tuned, so measure before keeping it. Once option A routes prefill through native_expert_grouped, the same switch applies to prefill too.
- STRATA_PREFILL_HELP=1, STRATA_PREFILL_EQUAL=1 and a different --prefill chunk size do not help a 154-token prompt. Help needs streamed chunks of 1024 tokens or more, and the cost of a short prompt is set by how many distinct experts it touches, not by chunk geometry.

## Review (optimistic)

**Best option:** The diagnosis holds up against the code. With T < 1024, stream_all is false (prefill.cpp:2198, 143-147). group_gather is off, so each expert runs gather_native, which is copy16_kernel (moe_mmq.cu:236-245), under pt.mark(kPfDequant) (prefill.cpp:3629-3643). The fused path needs stream_all (2881-2882), CUDA d_covered excludes Q8_0 (moe_fused_iq.cu:1003-1009), and the resident sort excludes native packs (mmq_resident_sort.hpp:25). native_expert_grouped exists, takes per-group blob pointers, and covers IQ3_S (21), IQ4_NL (20) and Q8_0 (8) (iq_kernels.cu:611-615, 1239-1515, 3446). Option A is feasible: ent_tok = m.src_dev, ent_dst = m.ids_identity, out = m.Dm, the small rows placed first, and a per-token quantize_q8_1_rows buffer, because Xq is in MMQ's block_q8_1_mmq layout and cannot be reused.

Recommended order:
(1) Run native_expert_bench <shard1> <iq3s-layer>,<q8_0-layer> 450 3 0 0 and 450 8 0 0 on a 3090. If it does not reach about 1.05 GB per layer in ≤ ~1.3 ms (>800 GB/s), A loses to B.
(2) Build option B with the 16-expert grouping unchanged (a per-expert x pointer at the 3 offset_x sites, mmq.cuh:1053/1147/1231) plus a ≥256 B zeroed tail per slot (or an explicit guard on the last slot). This removes the whole 24-28 ms dequant stage and the ~490 copy launches per layer. The output stays bitwise identical (memcmp of Dm), and the 512-token chunks gain 26-55 ms too. Then try one MMQ launch per layer with a KL gate.
(3) Add A for experts with ≤4 rows only if the bench shows it clearly beating MMQ per byte.
(4) Skip the STREAM_MIN=1 fused config, or run it once only to confirm it is a regression.

**Realistic gain:** At 154 tokens, either A or B takes the MoE expert work from about 49-53 ms to about 22-30 ms per 12-layer stage. B removes exactly the copy and keeps MMQ at about 25 ms. A probably lands at 22-35 ms unless the bench shows more than 700 GB/s. Stage GPU time goes from 78-87 ms to about 52-62 ms, and the 4-stage serial admission from about 330 ms to about 220-250 ms (-25 to -33%). The analysis's 200-230 ms is the optimistic end.

At 512 tokens, B removes the 26-55 ms dequant phase (a stage goes from about 146-165 ms to about 105-125 ms). A alone gives only about -5 to -15 ms there.

All of these numbers come from timing runs that record an event per expert, so they need re-checking with untimed wall clock. Some of the 'dequant' time may be host launch and event overhead rather than DRAM. Both options remove that overhead too, so the gain could be a little larger, but the baseline is uncertain.

**Wrong claims:**
- Exactness of option A: "The int8 dot products are identical, but the float accumulation order differs." — The activation int8 values are not guaranteed to match. MMQ's activation quantizer (third_party/llama.cpp/ggml/src/ggml-cuda/quantize.cu:515-521, quantize_mmq_q8_1) computes q = roundf(x * (127/amax)) and d = 1/d_inv. The verify/decode q8_1 that native_expert_grouped reads (include/strata/kernels/q8_1_finite.hpp:23-26 and quantize.cu:91-92) computes q = roundf(x / (amax/127)) and d = amax/127. Near .5 these two can round to different int8 values, and the scale bits differ. On the down side, H also goes through a different SwiGLU (mmq::swiglu vs swiglu_q8_1_entries_kernel, which uses __expf) and a different quantizer. The weight int8 values do match. So the result is still a rounding-level difference that needs the KL gate, but more than accumulation order changes.
- Option A's MoE estimate of 15-22 ms per stage, from ~600-750 GB/s for MMVQ-style IQ3_S kernels. — Nothing in the repo supports that rate on sm_86. The only grouped-kernel numbers come from the same kernel family on a V100 (bench/results/2026-10-07-v100-decode-kernels/README.md): IQ2_S/Q2_0, 28 groups of 1.38 MB, T=3 takes 150-180 us. That is about 215-260 GB/s, or 25-30% of HBM peak (about 40% at T=1). Today's MMQ already reads the weights at about 500 GB/s: gate/up 16 ms plus down 9 ms for roughly 12 layers × 450 experts × 2.33 MB ≈ 12.6 GB. For A to beat B, the dp4a/grid-lookup IQ3_S kernel has to beat tensor-core MMQ per byte, and that has not been shown. A more realistic range is 22-35 ms MoE per stage until native_expert_bench measures it.
- Option B is "still about 3x slower than A per byte at 3 rows (J=8 tiles, ~60% padding), but compute is not the bottleneck", and A is therefore recommended first. — This contradicts itself. If both paths are memory-bound, padding in the tensor-core tile costs nothing per byte. The measured MMQ stages (gemm gate/up 16 ms, gemm down 9 ms at 154 tokens) already run near 500 GB/s on the copied weights. Removing only the copy (B) cuts exactly the measured 24-28 ms. B's output stays bitwise identical if the 16-expert grouping is kept, and B also helps the 512-token chunks, where A covers only about a third of the experts.
- The copy cost of about 2 GB per layer at about 930 GB/s "matches the measured 24-28 ms", so dequant is purely DRAM-bound copying. — That only works out at about 100% of the 3090's peak bandwidth if ~450-490 distinct experts are touched, and copy kernels usually reach 80-85% of peak. So either far fewer experts are touched (not measured) or other time is mixed in. PfTimer::mark (prefill.cpp:1895-1906) records a cudaEvent on every compute() call, about 490 per layer, and the GPU timeline is cudaEventElapsedTime between marks, so idle bubbles count too. With timing on, the dequant phase also absorbs host launch overhead: about 490 copy launches plus 490 event records per layer is roughly 2 ms of host time. In the 512-token logs, dequant (24-55 ms) and host grouping (4-48 ms) move against each other, which points to time being charged to the wrong phase. The baseline (~330 ms per admission) and the per-stage gains need a re-measurement with wall clock or nsys, without per-expert marks.
- Option C: fused kernels at about 3 rows per expert would take "~3-5 ms per layer on a 3090, which could be no better than today". — moe_fused_iq.cu:1021-1035 gives IQ3_S at 5.6 ms per layer on the 5070 for 2048 tokens, which is already about one 64-row tile per expert. At 154 tokens the tile count is the same (one per expert), so the cost should be about the same, roughly 5-6 ms per layer on a 3090. That is worse than today's ~4.2 ms per layer for dequant plus MMQ, so the config experiment is more likely a regression than neutral. Also, fused_ring() is false for this pack, so MMQ's buffers stay and the 5 Q8_0 layers keep the copy.

**Missed:**
- The prompt path already has a direct precedent for a pointer table: the fused streamed walk fills fused::Batch.blob[] with m.cache->device_slot(...) (prefill.cpp ~2925-2935). Option A or B can copy that bookkeeping instead of designing new bookkeeping.
- The MMQ group buffers grp_gu/grp_d (16 × gu_max and 16 × d_max + MMQ_TAIL, about 23 MB plus about 15-28 MB) are carved from borrowed buffers, which are 'live expert-cache slots until a request lends them' (prefill.cpp:1205-1212). Removing the gather (B, or A for most experts) frees that VRAM, so more experts stay resident during admissions. That is a side gain the analysis does not count.
- A variant of B that needs no vendored patch: pad every native slot so a layer's slots have a uniform stride that is a multiple of each block size (gate/up IQ3_S 110 B and down IQ4_NL 18 B, i.e. lcm(990, 256) = 126,720 B: 2,329,600 -> 2,407,680 B, +3.4%; Q8_0 layers +6.4%), and keep each layer's experts in contiguous slots (layer_slot_range exists). One unpatched MMQ launch per layer can then address the cache in place through x = layer base and stride_channel_x. The VRAM cost (~0.5 GB per card) is probably too high with all experts resident, but the trade-off should be stated.
- B's padding does not need 4 KiB. MMQ_ITER_K = 256 over the 640-wide down rows over-reads 4 blocks (72 B for IQ4_NL, 136 B for Q8_0), and the gate/up rows are 2560 = 10 × 256, so they do not over-read. Today 2,329,600 and 3,148,800 B are both exact 256-B multiples, so there is no padding at all. The last slot of the arena would be read out of bounds, not just into a neighbour's bytes.
- GRP_NC = 4 means an expert with 5-8 rows takes 2 passes and re-decodes its IQ3_S rows (an L1/L2 hit, but extra ALU work). The grouped kernel turns ALU-bound quickly above 4 rows, so raising R to 16-32 is unlikely to pay. The Poisson ≤8 coverage (99.6% at λ = 3) also assumes uniform routing; real routers have hot experts that get far more rows.
- Option A needs no separate-pointer overload to be correct. A dedicated scratch from native_expert_scratch_bytes(T*K, 640) is about 3 × 5120 × 640 × 4 + hq ≈ 43 MB at chunk 512. The overload only saves VRAM and adds aliasing risk.
- Before any code is written, the 330 ms baseline should be re-measured without STRATA_PREFILL_TIMING. Its per-expert cudaEventRecord inflates exactly the phase being optimised.