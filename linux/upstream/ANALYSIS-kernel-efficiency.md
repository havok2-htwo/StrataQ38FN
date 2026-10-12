# Batch-decode kernel efficiency vs. the memory roofline (2026-10-12)

Read-only analysis (no GPU used, no engine started). Engine code: `/media/raid0/StrataQ38FN/Strata-143` (branch `g3-143`).
Traces: the existing `results/profile2/nsys_dec16` / `nsys_dec32` node-level reports (T16-142fca, before levers 3/4/5),
re-exported to sqlite. Current per-category numbers: `results/levers/qsa-slots/vprof16.txt` (T16-143l5).

## TL;DR

- One stage window (4 rows, 12 layers) must move **~2.5 GB** (S4 with the head **~2.9 GB**). Roofline at 936 GB/s:
  **2.66–3.07 ms**; at 85 % (796 GB/s): **3.13–3.62 ms**. Measured: **~5.5 ms** on the slowest stage (725 rows/s),
  4.7–5.0 ms on the 12-layer middle stage. Decode runs at **~53–56 % of peak bandwidth**, **~64 % of the 85 % roofline**.
- **The routed experts and the large dense Q8_0 GEMVs are already at the roofline** (745–860 GB/s). The ~1.5 ms gap per
  window is in four places:
  - the hyper-connection (HC) reads at 317–367 GB/s: **0.62 ms** over the floor;
  - QSA attention, latency-bound: **~0.3 ms**;
  - router, top-10, plan and quantize kernels: **~0.25 ms**;
  - ~250 tiny kernels per window (1.3–2 µs fixed each), mostly GDN conv/ab and per-slot launches: **~0.2 ms**.
- **New finding, production only:** with `--kv-resident 32768` every slot's QSA state is "streamed" (`kv_mode = 1`), so
  `STRATA_QSA_SLOTS` (lever 5) is off in production and all QSA layers use the per-row loop. This matches the −5.7 % of
  `T16-143-prod262` against the 64K config (#73). One-row-per-slot windows can run the slots kernels with streaming
  bit-identically.
- **The HC weights in the pack are not the GGUF's.** The pack holds a BF16 round-to-nearest copy of the GGUF's Q8_0 HC
  matrices (`conversions.json`: `exact: false`, `max_abs_err 0.0019`). `STRATA_HC_Q8=1` reads the GGUF's own bytes at half
  the traffic.
- The "UD-IQ4_XS" pack is **IQ3_S gate/up + IQ4_NL down** in 43 of 48 layers (Q8_0 down in layers 2, 4, 30, 46, 47;
  IQ4_XS gate/up only in layer 2).

| # | Lever | Exact | ms / window | Decode | Agent-h |
|---|---|---|---|---|---|
| 1 | QSA slots kernels also with KV streaming (one-row-per-slot windows): production only | yes | −0.30..−0.38 (prod) | **+5.5–7 % (prod)** | 2–4 |
| 2 | HC read: (a) `STRATA_GR_FAST=1`, zero code; (b) `gr_down_staged` with 3 tiles of weights in flight; (c) register norm on CUDA | yes | −0.43..−0.53 | **+8.5–10.5 %** | 0.5 + 2–4 |
| 3 | Bitwise opt-in arms, zero code: `STRATA_MMVF_ROWS=1` (router / indexer), `STRATA_ATTN_LANECELL=1` | yes (upstream parity) | −0.08..−0.15 | +1.5–3 % | 1–2 (GPU) |
| 4 | Exact fusions removing ~70–90 launches per window: swiglu→down prologue, route+plan, GDN conv over slots, qkv+z+a/b in one launch, GDN step split over slots | yes | −0.2..−0.3 | +4–5.5 % | 12–20 |
| 5 | `STRATA_HC_Q8=1`: the GGUF's Q8_0 HC bytes; output bits change, needs a KL/eval gate | no (GGUF-faithful) | a further −0.15..−0.25 | +3–5 % | 1 + eval |

Exact levers 1–4 together: about −1.0..−1.2 ms per window, +20–28 % decode on paper; after the #48 experience expect
**+12–18 %**.

## 1. Method

- **Model geometry** from the GGUF header; formats as the engine reads them (pack `index.txt`, `conversions.json`): HC
  down/up/inject, router `ffn_gate_inp`, `ssm_alpha/beta`, indexer q/k and the shared-expert gate-inp are BF16; dense
  projections, shared expert and PLE key are GGUF Q8_0.
- **Bytes per window:** dense weights once per window (multi-row kernels read a weight once for all rows). Routed experts
  are grouped by expert (`iq_kernels.cu:1226-1233`, `1262-1289`): D distinct experts × 2.33 MB per layer, D ≈ 39 (37–40)
  for 4 × top-10 of 512. GDN state 3.15 MB FP32 per row and layer, read and written (self-commit). QSA: f32 pooled indexer
  keys (128 B per cell) + up to 2048 selected cells × ~1040 B int8 K/V. Head Q6_K 521 MB on S4. PLE 41 MB on S1.
- **Kernel times** from the node-level nsys traces cut per window graph (GPU1, layers 12–23, 9 GDN + 3 QSA; 141 windows at
  16 slots, 81 at 32), adjusted for levers 3/4/5 (QSA 0.90 → 0.52 ms; commit graph gone, GDN step 0.23 → 0.32 ms).
  Node-level tracing adds ~0.1–0.2 µs per kernel (4.95 vs 4.85 ms span).
- **Achieved bandwidth** = bytes moved / mean kernel duration; **floor** = bytes / 796 GB/s (85 % of 936).

## 2. Roofline per stage window (split 12,24,37, context 2K per slot)

| stage | layers | dense incl. HC (MB) | of which HC BF16 | experts D=39 | GDN state R+W | extra | total | floor @936 | floor @85 % | measured | % of peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 0–11 (9 GDN, 3 QSA) | 1138 | 318 | 1167 | 235 | PLE 41 | 2610 | 2.79 | 3.28 | 5.52 | 51 % |
| S2 | 12–23 | 1138 | 318 | 1090 | 235 | – | 2492 | 2.66 | 3.13 | 4.99 | 53 % |
| S3 | 24–36 (10/3) | 1234 | 344 | 1213 | 261 | – | 2737 | 2.92 | 3.44 | 5.46 | 53 % |
| S4 | 37–47 (8/3) | 1041 | 291 | 1063 | 209 | head 535 | 2877 | 3.07 | 3.62 | 5.59 | 55 % |

Context dependence (KV + indexer reads per window, 4 rows × 3 QSA layers): 2K 29 MB (+0.04 ms), 8K 38 MB, 32K 76 MB
(+0.10 ms), 128K 227 MB (+0.29 ms). Per layer: GDN dense 61.9 MB, QSA dense 56.2 MB; HC 26.5 MB BF16 (14.2 MB as Q8_0) =
**28 % of the dense bytes**; router 2.6 MB; shared expert 5.2 MB.

Upper bound: every kernel at 85 % → slowest stage ~3.6 ms instead of 5.5 (+50 %); with ~150 unavoidable launches at ~2 µs a
realistic floor is ~3.9–4.0 ms (+35–40 %).

## 3. Per kernel category (GPU1, 16 slots, 4 rows per window, 12 layers)

| kernel (grid × block) | calls/win | µs/call | µs/win | MB/call | GB/s | floor µs | excess µs/win | why |
|---|---|---|---|---|---|---|---|---|
| `native_gu_multi_kernel` (80×40 × 256) | 12 | 73.9 | 886 | 54.9 | 744 | 69.0 | 58 | at roofline (850–900 GB/s incremental) |
| `native_down_multi_kernel` | 12 | 46.3 | 556 | 35.9 | 776 | 45.2 | 14 | at roofline |
| **`gr_down_staged_kernel` (41 × 256)** | 24 | 20.9 | 502 | 6.63 | **317** | 8.3 | **302** | 41 blocks on 82 SMs, weights only 1 tile ahead |
| **`gr_up_multi_kernel` (160 × 256)** | 24 | 18.9 | 453 | 6.92 | **367** | 8.7 | **244** | one dependent load chain per row and warp |
| `gr_norm_split_kernel` (4×4 × 256) | 24 | 3.6 | 85 | 0.41 | 115 | 0.5 | 73 | 16 blocks, latency |
| `native_mmvq_multi` qkv / out / z / q | 9/12/9/3 | 37/23/24/44 | 957 | 28/17/17/33 | 707–761 | – | 75 | fine (795 GB/s incremental + 2 µs) |
| `gdn_step_norm_multi_kernel` (48 × 512), per slot | 36 | ~8.9 | ~320 | 6.3 | ~710 | 7.9 | ~35 | 48 blocks per launch |
| `bf16_f32_mmvf_multi` router / idx-q | 15 | 9.4 | 141 | 2.62 | **278** | 3.3 | 92 | every block re-reads all 4 activation rows (20 MB L2/call) |
| `attn_chunk_kernel` per row (now one slots launch) | 12 | 41.4 | 496 | ctx | – | – | latency | QSA now 0.52 ms/window (§4.6) |
| `gdn_conv_l2_multi_kernel`, per slot | 36 | 1.7 | 63 | 0.21 | 120 | 0.3 | 53 | launch-bound |
| `gdn_ab_multi_kernel` (48 × 64) | 9 | 6.4 | 58 | 0.49 | **76** | 0.6 | 52 | 3072 threads for 96 dots × 4 rows |
| `route_multi` + `resident_plan_kernel` | 12+12 | 4.9+3.8 | 105 | – | – | – | ~80 | single-block latency on the critical path |
| `swiglu_q8_1_entries` + `combine_k10_vec4` | 12+12 | 3.4+2.1 | 67 | – | – | – | 56 | launch-bound |
| quantize kernels | 48 | 1.4–1.6 | 75 | – | – | – | ~70 | launch-bound |
| shared expert (side stream) | 7/layer | 2.4–7.2 | ~380 | 1.74 | 240–290 | – | – | hidden behind `native_gu_multi` |

Critical-path budget (µs per window, 12-layer stage): experts 1508 (floor 1369), dense 986 (892), **HC 1040 (421)**, GDN
441 (302), QSA ~360 (~50), router/top-10/plan 218 (40), quantize 75, hand-in 30 → **≈4660 vs ≈3080 floor, ≈1570 excess**.

Launches and gaps: ~456 kernels per window (~504 in production with streaming); graph idle only ~13 µs per window, gaps
~0.1 µs → **launch gaps are not a lever**; the fixed per-kernel duration is (~250 kernels < 4 µs ≈ 560 µs per window;
GEMV fit 2.0 µs intercept + 1.26 µs/MB). PDL is sm_90+; graph branches are excluded on Linux 3090s (`verify.cpp:688-690`, #905).

At 8 rows (32 slots): qkv 37 → 58 µs, HC down 21 → 33, HC up 19 → 31, `gdn_ab` 6.4 → 26.8, router 9.4 → 17.7; experts stay
at 850–900 GB/s. Marginal cost per extra row 4 → 8: 1.13 ms per stage window (byte floor ~0.42 ms).

## 4. Levers

Decode gain = x / (W − x), W = 5.50 ms (64K config) or ~5.85 ms (production 262K).

### 4.1 QSA slots kernels with KV streaming (production) — exact

`layer.cpp:634-651` (`kv_plan`) marks every main QSA state `mode = 1` as soon as `--kv-resident` < `--max-context`;
`verify.cpp:1403-1406` then forces the per-row loop (`1439-1451`). For one-row-per-slot windows (`keep_rows`,
`verify.cpp:918`; `one_row_per_slot`, `196-202`) run: slots scores → slots top-k → `qsa_kv_resolve` per row → slots attention
+ merge. Each row is a different slot with its own map, page pool and host copy (`QsaState`, `layer.hpp:228-240`), so
"resolve every row, then attend" is exact; the lever-5 caveat (`verify.cpp:189-192`) only concerns several rows of the same
slot. **−0.30..−0.38 ms per window in production (+5.5–7 %)**, 2–4 h. Verify: `qsa_slots_parity` with streamed pools,
`kv_stream_parity`, bt16/bt32/il/t07 with the prod config, a 2-slot 40–60K-context dump, overlap test. Risk: check that
`kv_stream.cu:89-180` only touches its own `st.map`/`st.host`.

### 4.2 Hyper-connection read — exact, the biggest kernel-level gap

- `gr_down_staged_kernel` (`fused_gr.cu:810-880`): 41 blocks × 8 warps; tile h+1 loaded at the start of tile h, so every
  tile waits a DRAM round trip; ~20 KB in flight per SM → 317 GB/s.
- `gr_up_multi_kernel` (`fused_gr.cu:323-375`): each warp walks 8 rows, each a dependent chain → 367 GB/s.
- **(a) `STRATA_GR_FAST=1`** (zero code): `gr_up_fast_kernel` (`fused_gr.cu:1196-1270`), upstream-bitwise (gr_parity),
  compiled for CUDA (`1815-1824` default only on Volta). gr_up 18.9 → ~9–11 µs: **−0.19..−0.24 ms (+3.5–4.5 %)**. Run
  `gr_parity` with the switch on sm_86 first.
- **(b) deeper prefetch in `gr_down_staged_kernel`:** ring of 3 tiles (~60 KB in flight per SM) or 81 blocks × 4 warps with
  2–3 tiles lookahead; same per-lane accumulation order → bitwise. 20.9 → ~10–12 µs: **−0.21..−0.26 ms (+4–5 %)**, 2–4 h.
- **(c)** allow `gr_norm_split_fast_kernel` (`fused_gr.cu:700-758`, gated to gfx12 by `norm_fast_on()`, `1008-1023`) on CUDA:
  −0.03 ms.

Total **−0.43..−0.53 ms per window, +8.5–10.5 %**, VRAM 0; also lowers the per-row cost (helps 32 slots and batch-MTP).

### 4.3 `STRATA_HC_Q8=1` — GGUF's own Q8_0 HC bytes (not bit-identical)

`native_dense.cpp:40-47, 323-347`, kernels `fused_gr.cu:1349-1810`. HC traffic 26.5 → 14.2 MB per layer (floor 0.42 → 0.21
ms per window). The BF16 pack copy is a lossy conversion of the GGUF (`exact: false`), so this is a reference change towards
the source, but output bits change → KL/eval gate (#17 set) and new reference dumps. VRAM +~170 MB per card while the BF16
copies stay. A further −0.15..−0.25 ms on top of 4.2.

### 4.4 Zero-code bitwise arms not yet tried here

| switch | where | what | expectation |
|---|---|---|---|
| `STRATA_MMVF_ROWS=1` | `native_bf16.cu:136-251`, gates 272/302 | 4 output rows per block; activations read once per 4 rows (today 20 MB L2 per router call) | router/idx-q 9.4 → ~4–5 µs: −0.06..−0.08 ms |
| `STRATA_ATTN_LANECELL=1` | `qsa_decode_attn.cu:167-196`, 644-646 | one cell per thread, no 12 `warp_sum`s per cell; parity checked in #64 | −0.02..−0.08 ms, grows with context |
| `STRATA_EXPERT_V2=1` | `iq_kernels.cu:3464-3475` | grouped decode experts for IQ3_S + IQ4_NL | ≤ −0.05 ms; `native_expert_parity` first |

Not worth it: `STRATA_Q8_PACKED=1` (second copy of every Q8_0 matrix, ~500 MB/card, against KV-first), `STRATA_GDN_SPLIT`
(needs ≥ 2 rows per call), `STRATA_TSUM`/`STRATA_LFUSE` (noise, #63), `STRATA_QFUSE` (disables solo self-commit).

### 4.5 Fewer launches through exact fusion

| fusion | where | removes | saving | effort |
|---|---|---|---|---|
| swiglu + q8_1 into the down kernel's prologue (`s_hq_buf`, same expression as `swiglu_q8_1_entries`) | `iq_kernels.cu:1362-1416`, 1546-1564 | 12 | −0.04 ms | 2–3 h |
| `route_multi` + `resident_plan` in one kernel | `native_router.cu:100`, `verify_kernels.cu:1065`, `verify.cpp:1519/1555` | 12 | −0.03..−0.04 | 2 h |
| GDN conv for the window's slots in one launch (`results/levers/self-commit/with-gdn-slots.patch`) | `verify.cpp:1164-1174` | 27 | −0.03..−0.04 | 1–2 h |
| horizontal qkv + z + a/b in one launch (generalised `PAIR` mode, `native_mmvq.cu:1163-1176`) | `verify.cpp:1151-1196` | 18 | −0.06..−0.08 | 3–5 h |
| QSA input projections k+v (+ idx-k, q, idx-q) | `verify.cpp:1277-1282, 1364-1377` | 6–9 | −0.02 | 2 h |
| GDN step over the slots with the split kernels (`verify_kernels.cu:338-500`, 768 blocks × 128) | `verify.cpp:1199-1209` | 27 | −0.04..−0.08 | 4–6 h |

Sum **−0.2..−0.3 ms per window (+4–5.5 %)**, 12–20 h, low risk; each with its parity test + bt16/bt32 dumps.

### 4.6 QSA attention value loop — exact

`attn_chunk_body` (`qsa_decode_attn.cu:253-285`): the data-dependent `continue` per cell serialises loads (one latency per
cell). Load `srow` and V for 8 cells into registers first, then the fmaf chain in the same order → bitwise. With LANECELL:
attention per QSA layer ~50–60 → ~20–25 µs, **−0.1 ms per window at ≥ 2K context** (+2 %), more at 32K+. 2–3 h.

### 4.7 Not levers

Routed experts (grouped, ~4–6 µs ramp/tail per call), dense Q8_0 (exact-layout rule `native_mmvq.cu:1150-1154`), L2 reuse
(already per window), graph gaps/host (~13 µs idle), a persistent megakernel (weeks, re-prove exactness), head (Q6_K at
755 GB/s; only lossy options).

## 5. GDDR6X memory overclock

Only bandwidth-bound time scales: ~58 % of the window (S2), ~62 % (S4), ~65–70 % after the §4 levers. +5 % data rate →
~+3 % decode; +10 % (21.5 Gbps) → ~+5.5–6.5 %. Risks: **no ECC** on GeForce GDDR6X (EDC only protects the link; errors past
it can silently flip resident expert weights read for hours); **memory junction temperature not exposed** by nvidia-smi/NVML
(only undocumented BAR0 reads, root + `iomem=relaxed`); 12 back-side modules run hot under sustained bandwidth; +5–15 W per
card under the 380 W cap. Setting needs root (`nvmlDeviceSetMemClkVfOffset`) or Coolbits + X. **Lower priority than §4.** If
tried: maintenance window with junction temperature readable, per-card steps, stop where rows/s stops rising (EDC replay) or
junction ≥ 95 °C, gate on byte-identical dumps + VRAM pattern test + 1 h soak; expected keep +3–5 %.

## 6. Batch-MTP for 2–8 users

Today (#71): 1 user 125 tok/s (solo, spec-4 MTP), 2 users 86, 4 users 75, 8 users 56 per user. Window time per stage:
T(1 row) ≈ 2.9–3.3 ms, T(2) ≈ 4.0–4.5, T(4) = 5.5, T(8) = 10.0; per-user speed ≈ 1 / (4 · T(rows)).

`--batch-mtp` exists but not with pipelined groups (`docs/BATCHING.md:31-39, 214-216`). Needed: fixed S = (1+d)·GS rows,
accept rule on the last stage, deferred commits, slot drafters on GPU3. Costs not counted before: the commit graph returns
for proposal rows (~0.3–0.45 ms per stage window, or keep both candidate GDN states, ~28 MB per slot per card); the drafter
on S4 (~0.45 ms per draft position); 8-row cap (`kVerifyMaxT = 8`) is not binding for 2–8 users.

| users | layout | today | d=1, c=0 | d=1, c=0.35 ms | d=2, c=0 | d=3, c=0 |
|---|---|---|---|---|---|---|
| 2 | 2 groups × 1 | 86 | 110–120 | 101–110 | 113–133 | 108–136 |
| 4 | 4 × 1 | 75 | 91–100 | 85–93 | 91–107 | – |
| 8 | 4 × 2 | 56 | 64–70 | 60–66 | 57–67 | does not fit |
| 16 | 4 × 4 | 45 | worse | | | |

Verdict: promising for 2–4 users (+15–45 % per user, ceiling = solo 125), modest at 8, a loss at 16 (must stay off).
~20–35 agent-h + 8–12 h GPU validation; later turns not byte-identical to today's reference (#59 (4)), needs new references.
Do it after §4.1/§4.2.

## 7. Suggested order

1. Zero-code arms in one GPU session: `STRATA_GR_FAST=1`, `STRATA_MMVF_ROWS=1`, `STRATA_ATTN_LANECELL=1` (parity tests
   first), dec_rows 16/32 slots + bt16 dumps. Expected +5–7 % together.
2. §4.1 QSA slots under streaming (production +5.5–7 %).
3. §4.2b/c HC prefetch and norm (+4–5 %).
4. §4.5 fusions (horizontal qkv+z+a/b first).
5. §4.6 attention V-loop.
6. Information only: `STRATA_HC_Q8=1` + KL/eval.
7. Batch-MTP for 2–8 users as a separate project; GDDR6X OC only under §5's conditions.
