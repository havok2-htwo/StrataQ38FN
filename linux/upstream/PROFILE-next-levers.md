# Profile of the default and ranked next levers (2026-10-12)

GPU profile (prefill phases per stage, decode windows, nsys) of T16-142fca, two read-only lever analyses and an adversarial merge. **The merged ranking (last section) is the plan.** Raw data: `results/profile2/`.

## Profile

Profile of T16-142fca (engine `strata-g3-142fc`, unchanged), done 2026-10-12. Nine engine runs, each in its own `strata-pr-*` scope; all ended cleanly and the GPUs are free again (276/15/15/15 MiB).

Decode is limited by GPU3: it runs the last 12 layers plus the head and sampling, and is 98 % busy. Prefill is limited by the stage chain: a 150-token prompt (one chunk) goes through the 4 cards one after another, and a 1.2K prompt also waits on the MTP draft layer on GPU3.

## 1. Prefill (slot 15; idle vs 15 slots decoding are the same)

Read times without timing instrumentation, 6 reps: 143 tok **147–149 ms**, 1193 tok **457–458 ms**. `STRATA_PREFILL_TIMING` adds about 25 ms (143 tok) and 45 ms (1193 tok). No decode kernels run on GPU0–2 during a read, so reads still run with decode stopped.

**Per stage, per phase (ms, engine timing, median of 6; stage mapping checked against nsys)**

| unit | tok | start | GPU end | hand-off | GPU | MMQ gate/up | MMQ down | dequant | combine | GDN total¹ | hc read | qsa proj | qsa attn | router+shared | host grouping |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 150 S0..S3 | 143 | 0 / 45 / 86 / 128 | +41–44 each | 45 / 87 / 128 / 174 | 41–44 | 13.8–14.7 | 7.6–8.7 | 4.6–5.6 | 0.7 | 7.1 | 2.4 | 1.8 | 0.1 | 1.3 | 0.6 |
| 1.2K S0 (all) | 1193 | 0 | 220 | 260 | 220 | 70.8 | 42.0 | 20.6 | 6.1 | 37.7 | 14.4 | 9.3 | 3.5 | 5.9 | 2.3 |
| S1c0 / S2c0 / S3c0 | 512 | 88 / 172 / 255 | 171 / 254 / 338 | 172 / 255 / 352 | 82–83 | 27.3–27.7 | 15.7–16.1 | 6.4–7.0 | 2.2 | 14.8 | 5.8 | 3.6 | 0.7 | 2.2 | 0.7 |
| S1c1 / S2c1 / S3c1 | 512 | 175 / 260 / 353 | 258 / 343 / 436 | 260 / 354 / 451 | 82–84 | 27.1–27.3 | 15.5–15.8 | 6.3–6.9 | 2.2 | 14.9 | 5.8 | 3.6 | 1.8 | 2.3 | 0.7 |
| S1c2 / S2c2 / S3c2 | 169 | 260 / 353 / 450 | 305 / 398 / 495 | 354 / 451 / 500 | 44–45 | 14.9–15.1 | 8.4–8.7 | 4.5–4.9 | 0.8 | 7.3 | 2.5 | 1.9 | 1.0 | 1.3 | 0.6 |

¹ GDN total = gdn + conv/gates + recurrence + out proj. `gather` is ≤0.5 ms and `wait copy` is 0. "host grouping" is now under 1 ms (it was 5–68 ms before `--aux-cpus`). On S3 the "after each chunk" host time is 15 ms per 512 chunk; that is the draft layer.

**1.2K read timeline (nsys, idle, 484 ms)**

| GPU | c0 | c1 | c2 | busy inside a chunk |
|---|---|---|---|---|
| 0 | 0.2–81.6 | 83.3–166.5 | 168.1–207.4 | 96–97 % |
| 1 | 86.1–165.3 | 170.9–246.9 | 250.5–289.8 | 95–98 % |
| 2 | 169.2–247.5 | 251.9–329.8 | **347.1**–388.9 | 95–98 % |
| 3 | 251.4–330.1 | **348.7**–427.3 | **443.6**–483.7 | 95–98 % |

- **GPU3 extra work:** after each chunk GPU3 runs the MTP draft layer as a graph: 330.3–346.0, 427.4–443.0 and 483.8–488.8 ms.
- **Critical path (484 ms):**
  - Chunk compute: 436 ms (90 %).
  - Draft layer: 31 ms (6.5 %).
  - Hand-off steps between cards: about 16 ms (3.5 %).
- **Why GPU2 idles:** GPU2 sits idle from 329.8 to 347.1 ms because it can only pass a chunk on once GPU3 has finished the previous one, including its draft layer.
- **Utilisation:** each card is busy about 198 ms of the 484 (41 %). For a 150-token read each card is busy about 36 ms of 150 (24 %).
- **Hand-off cost:**
  - The data is 21 MB per 512-token chunk, copied D2H then H2D through host memory.
  - Each copy takes about 1.6 ms (~13 GB/s).
  - The host adds about 1.3 ms per step.
- **Kernel split for one 512 chunk on one stage (77.6 ms):**
  - `mul_mat_q_tab` 44.4 (57 %), cuBLAS dense (`Kernel2` + ampere gemm) about 17, `dequant_q8_0` 2.7, `gdn_rec_kh` 2.8, HC (`gr_*`) 4.3, attention 1.2–2.4, gaps 1.5–2.4.
  - For 143 tokens: MMQ 19.3 of 33.5, dense 7.2, dequant 2.7 (fixed per chunk).

## 2. Batch decode (no admissions, temperature 0.7)

**Rows/s and per-row cost**

| | 16 slots (4 rows/window) | 32 slots (8 rows/window) |
|---|---|---|
| rows/s plain | **616.6** | **670.1** |
| stage-window period | 6.48 ms | 11.95 ms |
| group step (one token per slot) | 25.9 ms | 47.8 ms |
| GPU busy, cards 0/1/2/3 | 88.5 / 80.8 / 80.9 / **97.7 %** | 84.7 / 76.5 / 76.9 / **97.9 %** |
| window graph span, cards 0/1/2/3 | 5.33 / 4.85 / 4.86 / 5.77 ms | 9.28 / 8.32 / 8.38 / 10.53 ms |
| second (commit) graph: GDN step/commit, KV append | 0.44 ms | 0.87 ms |
| idle per period, cards 0/1/2/3 | 0.75 / 1.25 / 1.25 / **0.15** ms | 1.83 / 2.82 / 2.77 / 0.25 ms |
| host: `cudaGraphLaunch` per window | 95–105 µs | 155–190 µs |
| host: sampling `cudaStreamSynchronize` per group step | 148 µs | 287 µs |
| previous stage end → next stage launch | 0.46 ms | 0.9 ms |
| GPU time per row on the GPU3 bottleneck | 1.60 ms | 1.47 ms |

- **Cost per extra row:** each extra row in a window costs about **+1.34 ms on GPU3** and about +0.98 ms on GPU1/2. The fixed cost per window is only about 1.0–1.4 ms.
- **Why time grows with rows:** expert reads, GDN state, KV and HC all grow with the number of rows, so decode time grows almost linearly with rows.
- **Host load:** the host is a single thread spending 10 % of its time in CUDA calls. It is not the limit.
- **Card balance:** GPU1/2 have about 1.25 ms of spare time per period. A fully balanced split would give at most about +13 %. Moving one layer off GPU3 would make the period about 0.44 ms shorter, roughly +7 % in decode. Split `12,24,37` was tested in #30 through HTTP only and showed no change; it has not been re-checked at engine level.

**Time per category, ms per stage-window (`STRATA_VERIFY_PROFILE`, costs about 5 % throughput: 585 / 652 rows/s)**

| category | 16 slots S1 / S2 / S3 / S4 | 32 slots S1 / S2 / S3 / S4 |
|---|---|---|
| routed experts (VRAM) | 1.59 / 1.43 / 1.47 / 1.56 | 2.63 / 2.31 / 2.38 / 2.54 |
| QSA projections + attention | 0.89 / 0.88 / 0.88 / 0.91 | 1.53 / 1.50 / 1.49 / 1.56 |
| HC read at layer start (down/up) | 0.51 / 0.51 / 0.48 / 0.52 | 0.77 / 0.77 / 0.77 / 0.80 |
| HC read mid-layer + router | 0.75 / 0.73 / 0.72 / 0.77 | 1.12 / 1.09 / 1.09 / 1.12 |
| GDN (projections, z, conv, gates, recurrence) | 0.95 | 1.63 |
| out-proj | 0.30 | 0.42 |
| shared expert + quant | 0.29 | 0.40 |
| head (S4 only) | 0.77 | 1.82 |
| stage 1 window-start wait (PLE flag) | 0.30 | 0.57 |
| total stamped | 5.68 / 5.17 / 5.18 / 6.19 | 9.21 / 8.22 / 8.23 / 10.43 |

The node-level nsys run gives the same picture per window at 16 slots, with tracing overhead: MoE 0.8 ms, dense GEMV 0.75, HC 0.52, GDN 0.37, attention 0.34, PLE 0.15 (GPU0 only), head and sampling 0.44 (GPU3 only), hand-off copies 0.05. Details are in `nsys_node_dec16.txt` and `nsys_node_dec32.txt`.

I did not add these runs to TESTLOG.md, because this task only allowed writing to `results/profile2/`.

Files are in /media/raid0/StrataQ38FN/linux/results/profile2/:
- Notes: notes.txt
- Prefill tables: pf_stages.txt, pf_reads_plain.txt, pf_reads_timing.txt, nsys_prefill.txt
- Decode tables: vprof_table.txt, dec_rows.txt, nsys_graph_dec16.txt, nsys_graph_dec32.txt, nsys_node_dec16.txt, nsys_node_dec32.txt
- Raw engine logs: {pf_plain, pf_timing, dec16_plain, dec16_prof, dec32_plain, dec32_prof, nsys_pf, nsys_dec16, nsys_dec32, nsys_dec16g, nsys_dec32g}/events.log
- nsys traces: nsys_pf/nsys_pf_{idle,dec15}_{150,1200}_r1.nsys-rep, nsys_dec16{,g}/nsys_dec_b16.nsys-rep, nsys_dec32{,g}/nsys_dec_b32.nsys-rep. I deleted the sqlite exports to keep the folder small (92 MB); regenerate with `nsys export --type=sqlite`.
- Scripts: run.sh, prof_driver.py, pf_reads.py, pf_stages.py, vprof.py, dec_rows.py, nsys_an.py, nsys_graph.py
## Prefill levers


The biggest exact lever is in MoE MMQ: sort the full 16-expert groups by row count so most groups get a small tile width (J). It looks bit-exact by construction, but that still has to be proven with a bitwise test.

Changes that move chunk boundaries are excluded, because they change the MMQ stream-k split and the GDN/attention boundaries, so they are not bit-identical. #50, #51 and #54 also showed they lose.

**Throughput model.** From the measured 493 and 376 tok/s, R is about 0.185 s for short and 0.508 s for 1.2K prompts. That is the read plus about 50 ms of fixed admission cost (drain ~22, first window ~27, checkpoint ~5.5). The model gives:
- 1.2K: +0.276 tok/s per ms saved (~0.073 %/ms).
- 150 tokens: +0.475 tok/s per ms saved (~0.096 %/ms).

**Critical path.** For a 1.2K read it is GPU0 c0 → GPU1 c0 → GPU2 c0 → GPU3 (c0 + c1 + c2 + drafts) plus the hops. That is five 512-token stage-chunks, one 169-token stage-chunk and five hops. A 150-token read is four 143-token stage-chunks, three hops and one draft pass.

**New evidence.** I exported the existing 1.2K and 150 idle nsys traces to sqlite in the scratchpad (nothing new was written to `results/`):
- **MMQ, 512 stage-chunk:** 237 full-group launches are tiling (grid 160/320/640, 97.6 % efficiency, no stream-k). About 11 partial last groups use stream-k (grid 82, fixups). Each block takes 43–58 KB of shared memory and 115–254 registers, so only one block fits per SM.
- **Tile width (J) in use:** mostly 64–128. One hot expert sets J for its whole 16-expert group, and 46 groups need two column tiles.
- **Cost vs tile width (IQ4_XS gate/up, grid 160):** J16 64 µs, J24 69–75, J32 75–81, J64 98, J96 121, J128 141. For down: J24 36–39 µs, J128 86.
- **Idle gaps:** gaps over 5 µs total only 0.17 ms per stage-chunk. Host grouping and the per-layer sync are not a lever.
- **MTP draft pass:** graphs of 6 rows, 86 graphs per 512 cells, 14.7 ms busy:
  - `native_mmvq_multi`: 7.1 ms (514 × 13.8 µs).
  - `gr_down_staged` + `gr_up_multi`: 4.4 ms.
  - Small kernels: 3.3 ms.

## Ranked levers

**1. Sort the full MMQ groups by row count and keep the partial tail group unchanged**
- **Where:** `src/prefill/prefill.cpp:3290-3293` (the existing "Aurora" hook), `src/prefill/mmq_resident_sort.hpp:37-60`. Context: `prefill.cpp:937` (`MMQ_GROUP=16`), the group launch at `prefill.cpp:3800-3846`, and the tiling vs stream-k decision at `moe_mmq_tab.cuh:396,421-431`.
- **Change:** sort only `order[0 : n - n%16]` by count. Leave the last `n%16` experts in id order and last, so the stream-k launch stays identical.
- **Why it should be exact:**
  - Full groups stay at 16 experts, so they keep tiling with no K split.
  - The per-element sum in the mma `vec_dot` path does not depend on J.
  - The combine still sums by slot in k order.
  - Plain Aurora (`STRATA_MMQ_RESIDENT_SORT_NE=1`) is probably not exact on sm_86, because the hottest experts would land in the stream-k tail group.
- **Saving:**
  - 512 stage-chunk: about 8–12 ms (gate/up 27.9 → ~20–21, down 15.7 → ~11–12).
  - 143/169 stage-chunk: about 2–3.5 ms.
  - Per read: **1.2K −40 to −60 ms; 150 −8 to −14 ms.**
- **Throughput:** 1.2K +11 to +17 tok/s (+3–4.5 %); 150 +4 to +7 tok/s (+1–1.3 %).
- **Effort:** 2–4 h.
  - First a zero-code probe: run plain Aurora for timing and compare `STRATA_PREFILL_DUMP_R`.
  - Then the tail-preserving variant, checked with `prefill_mmq_tab_test`, `batch_test16` and a logits/R dump.
- **Risk:** medium, only on exactness (whether the J-invariance holds). If the bits differ, it becomes a "new deterministic reference" lever.

**2. Keep FP16 copies of the dense Q8_0 projections instead of dequantizing every chunk**
- **Where:** `src/prefill/gemm.cu:819-872` (`Gemm::native`: `dequant_f16` into scratch, then `f16`; row slicing at 858-867) and `src/prefill/prefill.cpp:1923-1928` (`native_proj`).
- **Today:** 75 `dequant_q8_0` launches cost a fixed 2.7 ms per stage-chunk.
- **Change:** a persistent buffer per weight, filled lazily, keeping the exact same row slicing so cuBLAS sees the same shapes. Bit-identical.
- **VRAM:** about +1.44 GB per card (764 MiB Q8 → FP16). Runtime free is 4.1 / 5.5 / 5.1 / 3.1 GB; GPU3 is the tightest. Offer a budget env (cache the largest matrices first) to be safe.
- **Saving:** **1.2K −16 ms; 150 −11 ms.**
- **Throughput:** +4.4 tok/s on 1.2K (+1.2 %), +5.2 on 150 (+1.1 %).
- **Effort:** 2–4 h. **Risk:** low.

**3. Run the MTP draft prompt pass on GPU3 in the background**
- **Where:**
  - `generate.cpp:8361-8385`: `on_chunk` calls `mtp.prefill(..., sync=true)`.
  - `prefill.cpp:4203-4214`: the last stage syncs and runs `on_chunk` before `run_impl` returns.
  - `prefill.cpp:4090-4094`: the successor-rule wait, so GPU2 waits on GPU3's draft.
  - `mtp.cpp:1181-1246`: a `sync=false` path already exists.
- **Change:** copy the chunk's R rows (21 MB, about 0.1 ms) into a drafter staging buffer, because `Rin_` reads `m.R` and the next hand-in overwrites it. Launch the graphs on the drafter stream at high priority and return. Synchronize the drafter stream before checkpoint saves, the slot copy and the end of the read. Same kernels and inputs, so exact.
- **Saving:**
  - 1.2K: the two mid-read drafts (d0 + d1 = 31 ms) are partly hidden; MMQ already takes one block per SM, so expect about −12 to −25 ms.
  - 150: about 0 to −3 ms. The only draft (2.6 ms) is the tail and can only hide behind admission work.
- **Throughput:** 1.2K +3 to +7 tok/s; 150 +0 to +1.4.
- **Effort:** 4–8 h. **Risk:** medium (stream ordering, checkpoint and slot copy of the drafter KV, error paths).

**4. Faster hand-off between cards**
- **Where:** D2H, sync and `std::async` at `prefill.cpp:4081-4105`; H2D at `prefill.cpp:2166-2175`; hand buffers at `prefill.cpp:1061`.
- **Today:** one 512-token hop costs D2H 1.6 + H2D 1.6 + host 0.7–1.3 ms, about 3.9–4.5 ms.
- **Change:**
  - Copy R in 4 pieces, so the H2D of piece i on the next card overlaps the D2H of piece i+1.
  - Gate the next card's H2D with cross-device events instead of a host sync.
  - Use a persistent stage worker instead of `std::async`.
  - Same bytes, so exact.
- **Saving:**
  - 1.2K: −6 to −10 ms (5 hops × ~1.5–2 ms).
  - 150: −1 to −1.5 ms (5.9 MB hops, ~1 ms each today).
- **Throughput:** +1.7 to +2.8 tok/s on 1.2K, +0.5 to +0.7 on 150.
- **Effort:** 4–6 h. **Risk:** low to medium.

**5. Wider drafter prompt graphs (6 → 8–16 rows)**
- **Where:** `mtp.cpp:1045-1061` (capture), `mtp.cpp:678-790` (`record_front`: the fc_hidden loop already runs in 8-row pieces; rope is one launch per token).
- **Exact only if** `native_mmvq_multi`, `gr_*_multi<T>` and `kv_append_q8_steps` are per-row invariant. That needs a parity test.
- **Saving:** 15–30 % of the draft time. 1.2K −5 to −9 ms (less if #3 is done); 150 about −1 ms.
- **Effort:** 3–5 h. **Risk:** medium.

**6. Make `gr_write_norm_rs` use memory bandwidth better**
- **Where:** `src/prefill/kernels.cu:142` and 1874. It runs at about 520 GB/s, 2.6 ms per 512 stage-chunk.
- **Exact only if** the reduction tree stays the same.
- **Saving:** 1.2K about −5 ms; 150 about −1 ms.
- **Effort:** 4–6 h. **Risk:** medium.

**7. Stretch goal: run the draft pass for c0/c1 on idle GPU1**
- GPU1 is idle from about 290 ms; it needs a copy of the drafter's dense weights (111 MiB), and the KV is shipped back to GPU3.
- Exact on the same sm_86 kernels. It removes the full 31 ms, replacing #3.
- **Effort:** 12–20 h. **Risk:** high.

## Combined estimate (levers 1–4)
- **1.2K:** about −75 to −105 ms, so the read goes from ~457–484 to ~380–390 ms. R falls from 0.508 to ~0.42 s, and throughput rises from 376 to about 400 tok/s (+6–7 %).
- **150:** about −21 to −28 ms, so the read goes from ~148 to ~122–127 ms. R falls from 0.185 to ~0.16 s, and throughput rises from 493 to about 503–505 tok/s (+2–2.5 %).
- **What remains:** GPU3 is then the only bottleneck. The rest of the critical path is filling the first chunk through three cards (~200 ms), and only changes to the chunk layout, which are not exact, could cut that.

## Not levers here (measured)
- **Host grouping and per-layer sync:** gaps over 5 µs total 0.17 ms per stage-chunk, and the shared expert is already deferred.
- **Running MMQ groups on two streams:** one block per SM and 97.6 % tiling efficiency, so maybe 1–3 %.
- **PLE:** 0.3 ms per chunk on GPU0, already double-buffered.
- **Successor rule:** GPU2's idle time only matters through the draft; fixed by #3.
- **GDN and QSA prefill kernels:** `gdn_rec_kh` 2.8 ms, attention 0.7–2.4 ms per 512 stage-chunk; an exact rewrite would gain at most 1–2 ms.
- **Chunk order or size** (169 first, equal splits, 128/1024): not exact.

## Faster but not bit-identical at temperature 0.7
These are worth knowing about. Under greedy the target tokens stay identical, but at 0.7 the accept/reject step (`sampler.cu:1009-1090`) uses the draft and its probabilities, so the tokens can differ.
- **`STRATA_SPLIT_MTP_BATCH=1`:** env only; batches the draft pass through MMQ. 1.2K about −25 to −28 ms, 150 about −2 ms.
- **`--mtp-window` below max context:** skips draft KV for early cells.
- **Plain Aurora sort:** see #1.
- **Choosing a different cuBLAS algorithm for the shapes forced onto ALGO2:** the 3× 478 µs `s1688gemm` calls per stage-chunk.

## Next to the read (lowers R, not the read)
The admission drain (`generate.cpp:10037-10046`, 21–24 ms per admission) waits for all four stages before GPU0 starts. Letting GPU0 start c0 while GPUs 1–3 finish draining could save about 10–20 ms per admission for both prompt lengths (+3–5 tok/s on 1.2K, +5–9 tok/s on 150). Effort 6–10 h, risk medium to high.

Files are in /tmp/claude-1000/-home-administrator-G3-VLLM/ddcb6733-9ffb-4382-97c5-d4797ced160b/scratchpad/nsx:
- idle1200.sqlite
- idle150.sqlite
- q2.py … q5.py
## Decode levers

**Batch-decode levers for T16-142fca, ranked by gain per effort (read-only code analysis plus the existing nsys traces, re-exported)**

Baseline: D = 616.6 rows/s at 16 slots (6.48 ms per stage-window) and 670.1 rows/s at 32 slots (11.95 ms).

How I turn a D gain into a 16-user gain: in the lockstep benchmark, decode is about 80 % of wall time for short prompts (493 tok/s) and about 61 % for 1.2K prompts (376 tok/s). So a 16-user gain is roughly 0.8 × ΔD (short) and 0.6 × ΔD (long).

**What the per-kernel numbers show (sqlite exports of the node-level traces):**
- **Last-stage period, 16 slots:** GPU3's 6.48 ms is the window graph 5.77, the commit graph 0.44, sampling 0.17 (4 launches of `sampler_split_part` and `sampler_split_merge`, one per row) and about 0.11 ms waiting for `cudaGraphLaunch`. The host reacts in only about 15 µs once a stage finishes, so host overhead is not the limit.
- **Stage balance:** GPU3 runs the Q6_K head: 0.69 ms at 4 rows, but 1.76 ms at 8 rows, where it falls back to the slow multi-column kernel.
- **GDN:** every slot runs `gdn_step_norm_multi` twice per GDN layer, once in the window and again in the commit graph. That is 36 + 36 launches of 7.7 µs on only 48 blocks.
- **Attention:** `attn_chunk_kernel` takes 41 µs per row per QSA layer and runs once per row: 0.50 ms per window at 4 rows, 1.0 ms at 8.
- **HC read (`gr_*`):** 1.0 ms per window at 4 rows, 1.6 at 8, against a memory floor of about 0.35 ms. `gr_down_staged` launches only 41 blocks on 82 SMs.
- **Already at memory bandwidth, no lever:** the routed experts (gate/up IQ3_S at about 735–860 GB/s, down IQ4_NL at about 770 GB/s). With top-10 of 512 experts, the rows in a window share almost no experts.
- **Pipeline limit after rebalancing:** with 4 groups on 4 stages, the period can't be shorter than (sum of the four stages' busy time + launch gaps + the GPU3→GPU0 turnaround of about 0.45 ms) ÷ 4. Today that sum is about 24.5 ms against a 25.9 ms cycle. Once the stages are balanced (lever 1), only changes that cut all four stages, or that cut this turnaround, still pay off.

| # | Lever | Where | D16 / D32 gain | 16 users short / long | Effort (agent-h) | Risk |
|---|---|---|---|---|---|---|
| 1 | Layer split `12,25,37` or `12,24,37`: moves one layer of about 0.40 ms (0.73 at 32) off GPU3 | `configs/T16-142fca.json` `layer_split` | +6 % (≈655) / +6 % (≈710) | +5 % / +4 % | 0.5–1 | Low: exact. Check the "6656 of 6656 profiled pairs" log line (room is 18.67 GiB, need about 14.9). #30 was HTTP-only and from before B-tab and `--aux-cpus`, so it doesn't settle this. |
| 2 | Self-commit for batch windows with one row per slot, dropping the commit graph | `verify.cpp:786` (`self_commit` excludes `batch_rec_`), `1068-1081`, `1093-1103` (pass `one_` instead of `nullptr`), `1194-1200` (tail snapshot), `976-991` (PLE history snapshot), `capture_commit_batch 2456-2540`, `batch_launch 2926-2927`, `commit_slot_prefixes 2802`, `bkey verify.hpp:321` | −0.31 / −0.62 ms on every stage: +5 % / +5.5 % | +4 % / +3 % | 5–8 | Low–medium. Keep the old graphs for duplicate-row (`--batch-mtp`) layouts. Gate: `batch_test16`/`32`, interleave and parking tests must stay token-identical. |
| 3 | Multi-slot QSA attention: one launch over the window's slots, passing up to 8 `QsaAttnPools` (72 B each) and indexing by `blockIdx.z` | `verify.cpp:1288-1301`; `qsa_decode_attn.cu:118-131` (q/ids/step/scratch already offset by z), launcher `619-705`; same for merge, scores/top-k (`qsa_select.cu`) and `kv_resolve` (`layer.cpp:904`) | −0.33 / −0.70 ms on every stage: +5–6 % / +6 % | +4.5 % / +3.5 % | 6–10 | Low (each block computes exactly as before). Not via graph branches: `verify.cpp:688-690` (#905) records graphs stalling under NVML queries on Linux 3090s. |
| 4a | HC read, no code change: try `STRATA_GR_V3=1` (164-block down kernel) and `STRATA_HC_Q8=1` (the GGUF's own Q8_0 bytes, half the traffic, 176 blocks) | `fused_gr.cu:1686-1745`, `1168-1200`, `1674-1677`; `native_dense.cpp:40-47`, `323-347` | −0.3..−0.5 ms on every stage: +5–8 % | +4–6 % / +3–5 % | 1 per arm, plus 2–3 for a quality check | Medium: both change output bits, so run a KL / eval check (#17 set). Batch still equals solo. |
| 4b | HC read, exact: `gr_down_staged` with 4 warps per block (81 blocks, one warp per row, same order), plus exact-T 7/8 instantiations for 8 rows | `fused_gr.cu:35` (`DOWN_BLOCKS`), `672-753`, `906-915`, `930-936` | −0.15..−0.25 / −0.3..−0.4 ms: +3–4 % | +2.5 % / +2 % | 3–5 | Low (bitwise) |
| 5a | One sampling launch for all sampled rows, with per-row seed and counter = position. Today it is one launch per row plus a host stream sync. | `verify.cpp:2842-2855`, `batch_poll 3008-3011`; `sampler.cu:1239-1290` (the multi-row path uses `p.counter + t`); device-side params already used by `coupled_draft_*` at `797-830`, `1317-1346` | −0.13 / −0.28 ms on GPU3 only: +2 % / +2.4 % | +1.5 % / +1 % | 3–4 | Low–medium: each row must still draw Philox(seed, position) |
| 5b | Queue the next waiting group's stage-3 window right behind the sampling, before the host waits. This hides the 0.11 / 0.21 ms launch gap. | pump `generate.cpp:9889-9938` and `9941-9986`; Verifier single-window state `verify.cpp:2920-2941`, `2943-3016`; needs a per-group copy of the sampled tokens | −0.11 / −0.21 ms on GPU3: +1.7 % | +1.3 % / +1 % | 5–8 | Medium (state machine, drain/overlap paths) |
| 6 | Multi-slot GDN step and conv: grid (h_v, S) with a per-slot state pointer table. The window step now reads the 3.1 MB state at only about 400 GB/s. | `verify.cpp:1068-1103`; `verify_kernels.cu:874-975` | −0.09 / −0.19 ms on every stage, after #2: +1.5 % | +1 % | 4–6 (do with #2) | Low |
| 7 | Stage-0 PLE, run once per row today: 0.29 / 0.58 ms on GPU0 plus a synchronous host gather before launch. Only pays once GPU0 is the slowest stage (after #1 and #5). | `verify.cpp:972-991` (a batched path exists at `960-967`), `stage_batch 2640-2657` | −0.15 / −0.3 ms on GPU0 | ≈+1–2 % (when GPU0 limits) | 4–6 | Low–medium |
| 8 | 32 slots only. Head at 8 rows: the interleaved kernel only supports 2–4 columns, so 8 rows fall back to the generic multi-column kernel (1.76 vs 0.69 ms). Same pattern in `gdn_ab_multi` at 8 rows (26.8 vs 6.4 µs). | `verify.cpp:1682-1687`; `native_mmvq.cu:2613-2617`, `2482`; `verify_kernels.cu:900-907` | Two 4-column launches: −0.38 ms (D32 +3 %), 1–2 h. A true 8-column kernel: −0.8 ms (+7 %), 6–8 h. `gdn_ab` case 8: −0.18 ms (+1.5 %), 0.5 h. | none at 16 users | see gain column | Low (bitwise) |
| 9 | Q8_0 expert-down layers (46 and 47 on GPU3) run the per-entry `native_down_kernel` (101 µs) | `iq_kernels.cu:663-906` (`kSplit`), `2687-2694` | −0.04 ms on GPU3: <1 % | ~0 | 3–4 | Low |
| 10 | Bitwise opt-in switches, no code change: `STRATA_LFUSE=1` (folds the 1-block shared-expert gate launch and the KV append), `STRATA_TSUM=1` | `verify.cpp:793-797`, `1363-1365`; `native_mmvq.cu:1156` | 0–2 % | 0–1.5 % | 0.5 each | Low |

**Not levers:**
- **Routed experts:** already at memory bandwidth.
- **Dense Q8_0 GEMVs:** about 0.75 TB/s at 4 rows. At 8 rows they are compute-bound, but the exact-layout rule at `native_mmvq.cu:1154` makes this 8–16 h for 32 slots only.
- **Hand-off copies:** 0.05–0.08 ms per window.
- **Host polling and reaction:** about 15 µs.
- **Groups and rows per window:**
  - 2 groups of 8: #20, half the cards sit idle.
  - 8 groups of 2: pays the fixed per-window cost of about 1.0–1.4 ms twice, roughly 510 rows/s.
  - More groups than stages: only with 20 or more slots.
  - So 4 groups × 4 rows is the best shape for 16 slots.
- **MTP draft rows in the batch:** an extra row costs +1.34 ms on GPU3, against 1.62 ms per produced token today. At an acceptance rate of about 0.65, each extra token would cost about 2 ms, so D would go down.

**Projected effect of all levers combined:**
- **D at 16 slots:** 616 → about 655 (#1) → 690 (#2) → 730 (#3) → 760–790 (#4) → about 800 with #5–#7, roughly +30 %. This is the same range as the earlier Lever C analysis.
- **D at 32 slots:** 670 → about 880–900 with #8.
- **16 users (lockstep):** short 493 → about 590–605, long 376 → about 430–440.

**Suggested order:**
1. #1 (config only) and the no-code arms #4a and #10, measured at engine level with `dec_rows.py`.
2. #2 together with #6.
3. #3.
4. #5a and #5b, then #7.

Nothing in the repo, `results/profile2` or TESTLOG.md was changed, and no GPU was used. I exported the four existing nsys reports to sqlite and wrote three small analysis scripts, all in `/tmp/claude-1000/-home-administrator-G3-VLLM/ddcb6733-9ffb-4382-97c5-d4797ced160b/scratchpad/nsys/`: `dec16.sqlite`, `dec32.sqlite`, `g16.sqlite`, `g32.sqlite`, `kt.py`, `react.py`, `react0.py`. The sqlite files total about 160 MB and can be deleted.
## Merged ranking (critique)


I re-checked both lever lists against the profile, TESTLOG #30/#48–#54, the Strata-142 code and the existing nsys sqlite exports. Nothing was run on a GPU and nothing in the repo or in `results/` was changed.

**Formulas used for the 16-user percentages:**
- **Decode:** throughput % = f × ΔD %. f is 0.82 (short) / 0.62 (long) in the lockstep benchmark and 0.76 / 0.54 under mixed traffic. The mixed values are fitted to #53, which implies an effective D of about 537 at an average reply of 320 tokens.
- **Read time:** throughput change per ms saved = thr²/L. That is 0.096 %/ms (short) and 0.073 %/ms (long) in lockstep, and 0.128 / 0.091 %/ms mixed.

## Corrections to the two lever lists

1. **"Not bit-identical at temperature 0.7" is wrong for every change that only touches the drafts.**
   - `STRATA_SPEC_PROB` is not set in T16-142fca.json.
   - The verify window draws each row from Philox(seed, pos0+t) and keeps a draft only when it equals that draw (`src/core/verify.cpp:2144-2153`). The min(p/q) kernel at `sampler.cu:1009ff` is the `SPEC_PROB` path only.
   - So drafts change acceptance (speed) but never the tokens. Upstream says the same in `include/strata/program/draft_kv_plan.hpp`.
   - Also, `ab_quick` runs greedy (#49).
   - Consequence: `STRATA_SPLIT_MTP_BATCH=1` is exact in tokens. It needs no code, and it makes prefill lever #3 (draft in the background) and #5 (wider draft graphs) unnecessary.
2. **Prefill lever #3 (background draft on GPU3) is overestimated.**
   - From 251.4 ms to 488.8 ms GPU3 is busy 233.7 of 237.4 ms: c0 78.7 + d0 15.7 + c1 78.6 + d1 15.6 + c2 40.1 + d2 5.0.
   - A second stream on the same card cannot remove that work. MMQ runs one block per SM with up to 254 registers, so there is little room to overlap.
   - Realistic saving is −3 to −8 ms, not −12 to −25. **No-go.**
3. **Decode lever #1 (layer split) is capped by the group's round trip through all four cards, not only by GPU3.**
   - A stage counts as done only when its stream has finished both the window graph and the commit graph (`verify.cpp:2926-2927` and `3000-3003`).
   - One group's round trip is therefore Σ(window + commit) + sampling + 4 hand-offs ≈ 24.6 ms, so a stage-window can't be shorter than about 6.15 ms.
   - That makes the realistic gain +5.4 %, not the +6 % from GPU3 alone.
   - #30 does not disprove it. Its log shows all experts resident (`6656 of 6656 profiled pairs`, results/engine-strata-T16-split12-24-37-short.log). It ran before `--aux-cpus`, when all 19 host threads shared CPU 0 and hand-offs were slow. Still, it is a warning, so measure at engine level first.
4. **Levers that only cut GPU3 lose most of their value once the cards are balanced** (decode #5a, #5b, #8, #9). After that, the 4-card round trip sets the speed, so a GPU3-only cut counts about ¼. The combined "+30 % D" is too high. Levers 1+2+3+4b at their estimates give about **+22 %**. After the #48 experience (the plan assumed ≤1.45 ms per extra row, measured 2.4–4), expect 50–70 % of each estimate to show up.
5. **The MMQ sort estimate holds for 512-token chunks but is too high for short reads.**
   - GPU3 c0 histogram: gate/up is about 27 ms per stage-chunk, mostly in tile widths J64–J128 at 98–183 µs per group, so about −8 to −11 ms per 512 chunk is believable.
   - 143-token chunk: tile widths are already J16–J64, and the cost only goes from 64.6 to 90 µs. That is about −1.5 to −2 ms per stage-chunk, so **−5 to −9 ms per 150-token read**, not −8 to −14.
6. **The MMQ sort is exact by construction (still needs the bitwise test).**
   - In the Ampere MMQ config, I=128, K_vram=`MMQ_ITER_K` and the same SRAM layout hold for every tile width for IQ4_XS, IQ4_NL and Q8_0. The settings come from `mmq-config-ampere.cuh` (third_party/llama.cpp/ggml/src/ggml-cuda/) and are used by `moe_mmq_tab.cuh:396-431`.
   - A full 16-expert group always gives 160·ntx tiles on 82 SMs. That is 97.6 % efficiency, at or above the 90 % threshold, so it runs one tile per block with no K split, whatever the width.
   - The per-group bounds are rebuilt from `off[order[j]]` (`prefill.cpp:3371-3376`), so they stay consistent with the B-tab direct path.
   - Only the stream-k tail group must be left unchanged.

## Merged ranking (next five)

| # | Lever | Go? | Effort | Lockstep short / long | Mixed short / long | Exact | Cheapest validation |
|---|---|---|---|---|---|---|---|
| 1 | Layer split `12,24,37` and `12,25,37` (config only). `12,25,37` also shortens the 1.2K read by about 8–10 ms; with `12,24,37` GPU2 becomes about as slow as GPU3 in prefill, so only −0 to −5 ms. | **GO** | 0.5–1 h | +4.4 % / +3.5–4 % | +4 % / +3–3.5 % | yes (same kernels per layer) | `dec_rows.py` at 16 slots against 616.6 rows/s for both arms plus a 12,24,36 control; check "N of N profiled pairs" and free VRAM; `batch_test16` against 12,24,36; then 2× `ab_quick` |
| 2 | `STRATA_SPLIT_MTP_BATCH=1` (env only): draft-layer prompt K/V through MMQ on GPU3. 1.2K −25 to −28 ms on the critical path, 150 −2 ms. | **GO** | 1 h | +0.2 % / +1.9 % | +0.3 % / +2.4 % | tokens yes (only the drafter K/V changes) | `pf_reads.py` at 1.2K (base 457 ms); log must show "draft layer prompt K/V batched on CUDA3"; `batch_test16` token-identical; 1-user tok/s (#46: 115.7) to confirm acceptance did not drop; GPU3 VRAM |
| 3 | MMQ group sort that keeps the tail: sort only `order[0 : n−n%16]` at the Aurora hook (`prefill.cpp:3290-3293`). 1.2K −40 to −55 ms, 150 −5 to −9 ms. | **GO** (after the probe) | probe 1–2 h, code 2–4 h | +0.7 % / +3.3 % (2.9–4.0) | +0.9 % / +4.1 % (3.6–5.0) | expected yes; must be proven | Zero-code probe: `STRATA_MMQ_RESIDENT_SORT_NE=1` plus `pf_reads.py`. This gives the timing; plain Aurora puts the hot experts in the stream-k tail, so its bits may differ. Compare with `STRATA_PREFILL_DUMP_R_ALL` (DUMP_R only dumps every 64th row). Then the variant, gated by `prefill_mmq_tab_test` 874/874, `batch_test16` and an all-rows R dump. |
| 4 | Batch self-commit, dropping the commit graph (decode #2). Removes about 0.31 ms on every card and from the round trip. The commit is already unconditional in batch mode ("every row is kept", `verify.cpp:2927`), so the risk is in the snapshot/PLE ordering, not in what gets committed. Do decode #6 (multi-slot GDN) together with it. | **GO** | 5–8 h | +4.3 % / +3.3 % | +4.0 % / +2.9 % | yes | `STRATA_VERIFY_PROFILE` plus `dec_rows.py` (D +5 %); gates: `batch_test16`/`32`, interleave, parking, `batch_overlap_test` including RESUME |
| 5 | Multi-slot QSA attention (decode #3): one launch across the window's slots for attention, select and `kv_resolve`. About −0.33 ms on every card. | **GO**, but last | 6–10 h (could run over) | +4.6 % / +3.5 % (range 2–6) | +4.3 % / +3.0 % | yes (per-block arithmetic unchanged) | Engine-level `STRATA_VERIFY_PROFILE` "QSA" category (0.89 ms per window today), then `dec_rows.py`; `batch_test16`/`32` |

**If 1–5 all reach their estimates:**
- Decode goes from about 616 to about 720 rows/s.
- Lockstep: short 493 → ~565 (+15 %), long 376 → ~439 (+17 %).
- Mixed: short 410 → ~467 (+14 %), long 290 → ~338 (+16 %).
- Realistically, expect about +8 to +12 %.

## Small but certain, worth keeping as #6

FP16 copies of the dense Q8_0 weights (prefill #2, `gemm.cu:819-872`). The dequant costs a fixed 2.7 ms per stage-chunk (75 launches), so this saves −16 ms per 1.2K read and −11 ms per 150-token read.
- Gain: lockstep +1.0 % / +1.2 %, mixed +1.4 % / +1.5 %. Exact, low risk, 2–4 h.
- Cost: about 1.44 GB more VRAM per card. Check the budget together with lever 1, which adds about 1.1 GiB on the card that gains a layer. That card has 5.1 GB free today on GPU2 or 5.5 GB on GPU1; GPU3 gets room back.

## No-go or demoted

- **Prefill #3 (background draft):** see correction 2.
- **Prefill #5 (wider draft graphs) and #7 (draft on GPU1):** made unnecessary by lever 2. #7 would only gain the remaining few ms, at 12–20 h.
- **Prefill #4 (faster hand-off):** about 15.6 ms of hops are on the critical path, so it saves −6 to −10 ms (+0.5–0.7 %) for 4–6 h. Later.
- **Decode #4a (`STRATA_GR_V3` / `STRATA_HC_Q8`):** both change output bits. Worth a 1 h engine-level measurement as information only; adopt only after a KL/eval check against the #17 set.
- **Decode #5a, #5b, #8, #9:** GPU3-only, so after lever 1 they are worth about ¼ of the quoted values (≤ 0.5–1 %). #8 matters only at 32 slots.
- **Admission drain overlap:** risk is medium to high, and #41/#53 showed reads that are stretched or overlapped mostly trade throughput for latency.
- **A cheaper, partial version of lever 4 (my own idea, not checked against the code):** record an event after the window graph and hand off on that event, leaving the commit graph queued behind it on the same stream. This takes the commit out of the round trip and is exact by stream order. It is worth about +2 % only on top of lever 1, and lever 4 makes it unnecessary.

## Evidence files

- /media/raid0/StrataQ38FN/Strata-142/src/core/verify.cpp (786, 2144-2153, 2926-2927, 3000-3003)
- /media/raid0/StrataQ38FN/Strata-142/src/program/generate.cpp (8356-8385)
- /media/raid0/StrataQ38FN/Strata-142/include/strata/program/draft_kv_plan.hpp
- /media/raid0/StrataQ38FN/Strata-142/src/prefill/prefill.cpp (3285-3293, 3371-3376, 3820-3850)
- /media/raid0/StrataQ38FN/Strata-142/src/prefill/mmq_resident_sort.hpp
- /media/raid0/StrataQ38FN/Strata-142/src/prefill/moe_mmq_tab.cuh (396-431)
- /media/raid0/StrataQ38FN/Strata/third_party/llama.cpp/ggml/src/ggml-cuda/mmq-config-ampere.cuh (lines 104-119, 295-340)
- /media/raid0/StrataQ38FN/linux/results/engine-strata-T16-split12-24-37-short.log
- Tile-width histograms: /tmp/claude-1000/-home-administrator-G3-VLLM/ddcb6733-9ffb-4382-97c5-d4797ced160b/scratchpad/nsx/q3.py run on idle1200.sqlite and idle150.sqlite (GPU3)