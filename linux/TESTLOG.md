# Test log (chronological)

Every attempt is recorded here **before** the next one starts: date, what was tested, result, verdict, raw data. Before a
new test, check here whether it has already been run. Load profile unless stated otherwise: `bench_concurrency.py`, 2
requests per user, 512 output tokens, temperature 0.7, thinking off; "short" ≈ 120, "long" ≈ 1,230 prompt tokens. Numbers =
total throughput in tok/s (output) or decode per user, TTFT in s. All Strata runs: UD-IQ4_XS, 4× RTX 3090.

Verdicts: ✅ adopted · ➖ no measurable effect · ❌ worse / discarded · 🔬 insight only

## 2026-10-10

| # | Test | Result | Verdict | Raw data |
|---|---|---|---|---|
| 1 | Strata setup default, solo (no `parallel`), with/without `--resident-experts` | 121 tok/s single, ~88 total; without resident mode 60 GB RAM → oomd kill | ✅ `--resident-experts` mandatory | `strata-A-solo*` |
| 2 | `parallel 8` (upstream limit), split auto | ~177 short/long, 31 tok/s per user | 🔬 baseline | `strata-B-batch8` |
| 3 | Engine patch 1: 16/32 slots (8 rows per group) | 16 slots bit-identical, batch decode 603 tok/s; over HTTP only ~195 | ✅ patch (HTTP limit = prefill) | `batch_test_16/32.txt`, `strata-C/D-*` |
| 4 | Prefill chunks 256/512/768/1024 + decode share 1/2 | 512 best (~246–250 long); decode share has no effect (see #21) | ✅ `--prefill 512` | `strata-E*`, `strata-F*` |
| 5 | Slots 12/16/20/24/28 (short prompts) | plateau ~330–361, 12 slots best latency | 🔬 | `strata-H*-short` |
| 6 | Users 24–40 at 12 slots | flat 346–358, wait time rises | 🔬 | `strata-U-b12-users` |
| 7 | 3 GPUs (`--trim-stage-weights`, split 16/32) | 117 single / 266 batch | 🔬 4th card only useful under load | `strata-T3-*` |
| 8 | UD-Q4_K_XL | 97% of experts in VRAM, 100 single / 241 batch | 🔬 IQ4_XS remains the target | `strata-K-*` |
| 9 | MTP off (`--spec 4` without `--mtp`) | 81 instead of 117 tok/s | ✅ MTP on (+45%) | `strata-M-nomtp` |
| 10 | Prefill across length 2K–28K: chunks 512 / auto / auto:32768 | 512: 1,247→2,239 tok/s, best overall picture | ✅ 512 | `strata-P-*` |
| 11 | vLLM 0.29 Flash-Next W4A16 (EP, FP8 table via patch), prefix cache on/off | 65–75 single, ~420–450 plateau, max ~21 concurrent | 🔬 | `vllm-w4a16-*` |
| 12 | vLLM 0.31 Flash-Next, prefix cache on | 79 single, ~400 long / ~710 short, cap ~22 | 🔬 | `vllm031-flashnext-pc-*` |
| 13 | vLLM 0.31 Flash-Next with MTP at 64 and 32 seqs | OOM and no KV space, respectively | ❌ | `vllm031_mtp*.log` |
| 14 | vLLM 0.31 prefill across length | ~3,500 tok/s constant | 🔬 | `vllm031-P-*` |
| 15 | vLLM 0.31 27B FP8 preset | 64 single, 409 long / 838 short (64 users), ~100 GB RAM | 🔬 | `vllm031-27b-fp8-*` |

## 2026-10-11

| # | Test | Result | Verdict | Raw data |
|---|---|---|---|---|
| 16 | vLLM 0.31 Flash-Next MTP with 8 seqs, 1,024 batch tokens, 0.93 | **136 single**, 490 short (8 users), 65K KV | 🔬 MTP only works with low parallelism | `vllm031-flashnext-mtp2-s8-*` |
| 17 | Quality set (MMLU-Pro 240 / GSM8K 150) | Strata IQ4_XS 77.9/94.0 · 27B 72.1/95.3 · vLLM-W4A16 70.4/94.7 · RAZOR 62.5/92.7 | 🔬 | `eval/eval_results.jsonl` |
| 18 | RAZOR-65B (50% of experts pruned, NVFP4) in vLLM | 111 single, ~412 long; quality dropped sharply, German output flawed | ❌ model deleted | `vllm031-razor65-*` |
| 19 | Hub survey of 910 Flash-Next repos | no smaller vLLM-compatible quant without quality loss | 🔬 | `quant-survey/` |
| 20 | 16 slots in 2 groups; 12 slots in 2 groups | 266–272 / 264–270 short, 200 long | ❌ half the cards idle | `strata-Q2-*` |
| 21 | Decode share 4.0 | 366–375 / 259 | ➖ no effect in pipeline mode (`generate.cpp:9706` upstream) | `strata-Q3-*` |
| 22 | 2 instances of 2 GPUs each (round-robin) | 152–182 short, 100 long, 90 single | ❌ | `strata-Q1-dp2-*` |
| 23 | **Split `12,24,36` + `--trim-stage-weights`** | **405 short / 288 long** (+14/+16%), no buffer borrowing | ✅ | `strata-T16-trim-*` |
| 24 | same, 20 / 24 slots | 409/288, 416/293 | ➖ plateau | `strata-T20/T24-*` |
| 25 | same, chunks 1,024 / 2,048 | 406/281, 403/290 | ➖ | `strata-T16-pf*` |
| 26 | Multi-turn 8/16 agents, trim config 32K vs. agent config (262K, `--kv-resident`, 24 GB conv. cache) | follow-up turn TTFT 7.1/10.4 s → **4.8/5.0 s**, 16 agents 43 → 73 tok/s | ✅ agent config | `multiturn.jsonl` |
| 27 | Multi-turn against vLLM 0.31 (+20 GB RAM KV tier) | 8: 89 tok/s, 6.4 s; 16: 77 tok/s, **32 s** | 🔬 Strata ahead under agent load | `multiturn.jsonl` |
| 28 | Engine patch 2: direct slot copy GPU→GPU | copy 65.6 → 34.0 ms, 405/294, agent TTFT 4.5 s; bit-identical | ✅ (small) | `exact_direct_*`, `strata-T16-direct-*` |
| 29 | Profiling the batch window (`STRATA_VERIFY_PROFILE`) | GPU 5.1–6.1 ms of ~6.95 ms per step; card 4 +0.73 ms head | 🔬 overhead small | `prof_batch16*` |
| 30 | Split `12,24,37` / `13,25,37` | 399/289, 397/282 | ➖ | `strata-T16-split*` |
| 31 | `--ple-io ram` | **417 / 297**, 1.2K prompt 821 instead of ~1,100 ms | ✅ | `strata-T16-pleram-*` |
| 32 | Overlapped admission, prototype (`STRATA_PIPE_OVERLAP_SPIKE`) | 413/294 (no gain), **TTFT p50 2.03 → 0.57 s** | 🔬 latency lever only | `strata-S16-spike-*`, `patches/experimental/` |
| 33 | Prefill timing of short prompts (`STRATA_PREFILL_TIMING`) | 154 tok.: 78–87 ms per card, 31% "dequant" = expert copy for MMQ | 🔬 | `prefill_timing.txt`, `upstream/ANALYSIS-short-prompt-moe.md` |
| 34 | `STRATA_PF_FUSED=1` | no change (only kicks in from 1,024-token chunks) | ➖ | `prefill_timing_fused.txt` |
| 35 | Design "MMQ direct on padded cache slots" (idea B) — code review | rejected: cache is profile-ordered (layers interleaved) and changing MMQ's channel count changes stream-k rounding → not bit-identical; exact alternative "B-tab" = own copy of MMQ's outer kernel with a per-expert pointer table | 🔬 design in `upstream/DESIGN-mmq-direct.md` | `upstream/DESIGN-mmq-direct.md` |
| 36 | Upper bound: skip the resident-expert gather (`STRATA_HACK_SKIP_GATHER=1`, wrong outputs, branch `g3-hack-skip-gather`) | prompt read ~162 tok: 397–413 → **240–292 ms** (−35–40%); ~1,228 tok: 900–908 → **609–619 ms** (−32%) | 🔬 B-tab worth building (est. +12–15% multi-user throughput, −30–40% TTFT, also single GPU) | engine log 2026-10-11 ~12:00, `configs/HK-*.json` |

## Analyses (no GPU)

- `upstream/ANALYSIS-batching-levers.md`: 4 engine levers (overlapped admission, MTP in the batch, window overhead,
  quick wins) with effort estimates and cross-checks.
- `upstream/DESIGN-overlap.md`: complete design of overlapped admission (6 milestones).
- `upstream/ANALYSIS-short-prompt-moe.md`: short-prompt prefill, options A (decode kernel for ≤ 8 rows),
  B (MMQ directly on the cache slots), C (fused kernels).

## Open / next steps

- Variant B (MMQ directly on padded cache slots): design in progress, then implementation + exactness check + measurement.
- Safe overlapped admission (latency).
- Parked: dynamic switching pipeline ↔ expert parallel (see the memory/ideas note in the README section).

## Appendix: all measurement runs (generated automatically from the JSONL files)

### `results/sweep.jsonl`

| Label | Users: total tok/s (TTFT p50) |
|---|---|
| `vllm-w4a16-ep4-nomtp` | 1: 72.2 (0.99 s), 2: 121.6 (0.57 s), 4: 210.3 (1.46 s), 1: 65.0 (1.34 s), 2: 115.1 (0.58 s), 4: 197.2 (1.45 s), 8: 269.3 (2.67 s), 16: 375.8 (2.42 s), 32: 384.2 (7.2 s), 48: 418.4 (29.54 s), 60: 400.1 (50.93 s), 96: 414.2 (87.52 s) |
| `vllm-w4a16-ep4-nopc` | 1: 74.9 (0.37 s), 8: 255.9 (1.81 s), 16: 374.7 (2.26 s), 24: 419.8 (3.4 s), 32: 418.1 (6.62 s), 48: 452.2 (27.14 s), 64: 442.0 (48.21 s) |
| `strata-A-solo` | 1: 91.5 (1.5 s), 2: 87.1 (7.37 s), 4: 87.4 (18.94 s), 8: 87.6 (41.64 s), 16: 89.7 (85.95 s) |
| `strata-B-batch8` | 1: 85.9 (1.69 s), 2: 106.2 (2.51 s), 4: 147.9 (3.32 s), 8: 177.0 (5.03 s), 12: 167.8 (10.84 s), 16: 170.8 (11.74 s), 32: 174.4 (35.66 s), 64: 175.8 (65.32 s) |
| `strata-C-batch16` | 1: 84.8 (1.7 s), 8: 171.0 (5.94 s), 16: 198.4 (9.26 s), 24: 193.6 (19.38 s), 32: 192.8 (20.26 s), 64: 194.9 (46.04 s) |
| `strata-D-batch32` | 16: 179.9 (10.61 s), 32: 191.4 (8.41 s), 48: 191.5 (21.26 s), 64: 192.1 (27.28 s), 96: 191.9 (37.75 s) |
| `strata-E1-b16-pf512-share1` | 16: 246.3 (4.58 s), 32: 240.8 (15.1 s) |
| `strata-E2-b16-pf256-share1` | 16: 228.7 (5.42 s), 32: 226.5 (17.78 s) |
| `strata-E3-b16-prefillhelp` | 16: 193.6 (7.17 s), 32: 193.4 (20.26 s) |
| `strata-F1-b16-pf1024-share1` | 16: 226.4 (6.65 s), 32: 221.9 (12.89 s) |
| `strata-F2-b16-pf512-share2` | 16: 246.0 (4.58 s), 32: 250.6 (12.25 s) |
| `strata-F3-b16-pf768-share1` | 16: 245.3 (4.56 s), 32: 238.6 (15.62 s) |
| `strata-G16-long` | 1: 95.6 (1.15 s), 2: 117.6 (1.12 s), 4: 164.4 (2.28 s), 8: 209.4 (4.52 s), 16: 248.4 (4.57 s), 24: 237.7 (11.02 s), 32: 233.9 (11.07 s), 64: 246.0 (46.48 s) |
| `strata-G16-short` | 1: 103.7 (0.53 s), 8: 281.0 (2.0 s), 16: 357.5 (2.6 s), 24: 364.6 (6.55 s), 32: 364.5 (20.27 s), 64: 354.3 (29.53 s) |
| `strata-G32-short` | 16: 318.8 (2.73 s), 32: 342.4 (2.98 s), 48: 330.0 (7.63 s), 64: 340.9 (12.86 s), 96: 344.7 (14.22 s) |
| `strata-H12-short` | 12: 361.1 (1.6 s), 18: 335.9 (2.62 s) |
| `strata-H16-short` | 16: 347.9 (2.07 s), 24: 358.1 (6.31 s) |
| `strata-H20-short` | 20: 335.9 (2.97 s), 30: 326.6 (6.51 s) |
| `strata-H24-short` | 24: 331.2 (2.63 s), 36: 328.3 (7.43 s) |
| `strata-H28-short` | 28: 328.0 (3.16 s), 42: 328.3 (8.21 s) |
| `strata-A-solo-mtp` | 1: 87.2 (1.5 s) |
| `vllm031-27b-fp8-p1000` | 1: 59.1 (0.76 s), 4: 130.7 (2.62 s), 12: 241.0 (4.3 s), 16: 283.1 (4.34 s), 32: 370.6 (5.13 s), 64: 409.1 (9.2 s) |
| `vllm031-27b-fp8-p100` | 1: 63.5 (0.13 s), 4: 203.7 (0.46 s), 12: 382.0 (1.24 s), 16: 484.9 (1.25 s), 32: 705.2 (2.25 s), 64: 838.1 (3.35 s) |
| `vllm031-flashnext-pc-p1000` | 1: 75.0 (0.38 s), 8: 238.2 (1.85 s), 12: 365.9 (1.9 s), 16: 385.8 (1.95 s), 24: 416.0 (4.16 s), 32: 397.5 (22.27 s), 48: 395.0 (33.44 s), 64: 398.2 (49.53 s) |
| `vllm031-flashnext-pc-p100` | 1: 78.0 (0.15 s), 8: 413.5 (0.56 s), 12: 499.3 (0.61 s), 16: 621.8 (0.68 s), 24: 590.8 (0.88 s), 32: 706.4 (0.95 s), 48: 683.2 (16.21 s), 64: 710.0 (31.46 s) |
| `strata-K-q4kxl-solo` | 1: 70.0 (2.26 s) |
| `strata-K-q4kxl-b12-short` | 12: 241.0 (4.12 s), 24: 233.7 (11.02 s) |
| `strata-K-q4kxl-b12-long` | 12: 149.8 (9.32 s) |
| `strata-T3-solo` | 1: 95.8 (1.1 s) |
| `strata-T3-b12-short` | 12: 265.9 (2.48 s), 18: 253.0 (6.26 s) |
| `strata-M-nomtp` | 1: 64.6 (1.42 s) |
| `strata-U-b12-users` | 24: 347.3 (4.17 s), 28: 346.0 (5.06 s), 32: 357.3 (9.11 s), 36: 358.2 (15.81 s), 40: 351.2 (17.2 s) |
| `strata-P-pf512-pf1600` | 1: 17.1 (1.56 s) |
| `strata-P-pf512-pf6500` | 1: 7.3 (4.06 s) |
| `strata-P-pf512-pf13000` | 1: 4.2 (7.28 s) |
| `strata-P-pf512-pf24000` | 1: 2.4 (12.73 s) |
| `strata-P-auto-pf1600` | 1: 11.6 (2.44 s) |
| `strata-P-auto-pf6500` | 1: 5.0 (6.16 s) |
| `strata-P-auto-pf13000` | 1: 3.7 (8.35 s) |
| `strata-P-auto-pf24000` | 1: 2.5 (12.45 s) |
| `strata-P-auto32k-pf1600` | 1: 11.5 (2.46 s) |
| `strata-P-auto32k-pf6500` | 1: 4.7 (6.46 s) |
| `strata-P-auto32k-pf13000` | 1: 2.7 (11.36 s) |
| `strata-P-auto32k-pf24000` | 1: 1.5 (20.84 s) |
| `vllm031-P-pf1600` | 1: 33.6 (0.56 s) |
| `vllm031-P-pf6500` | 1: 12.2 (2.22 s) |
| `vllm031-P-pf13000` | 1: 6.7 (4.41 s) |
| `vllm031-P-pf24000` | 1: 3.8 (8.13 s) |
| `vllm031-flashnext-mtp2-s8-p1000` | 1: 115.8 (0.66 s), 4: 268.9 (0.81 s), 8: 224.9 (1.53 s), 12: 234.6 (14.15 s) |
| `vllm031-flashnext-mtp2-s8-p100` | 1: 132.7 (0.15 s), 4: 346.4 (0.3 s), 8: 490.3 (0.3 s), 12: 486.1 (0.91 s) |
| `strata-eval-iq4xs` | 1: 10.8 (1.33 s) |
| `vllm031-razor65-mtp2-s12-p1000` | 1: 102.7 (0.36 s), 4: 263.9 (0.58 s), 8: 345.7 (0.76 s), 12: 411.7 (1.11 s), 24: 416.2 (15.04 s) |
| `vllm031-razor65-mtp2-s12-p100` | 1: 110.2 (0.15 s), 4: 304.1 (0.22 s), 8: 445.5 (0.37 s), 12: 536.1 (0.29 s), 24: 532.4 (11.49 s) |
| `strata-Q2-b16-groups2-short` | 16: 266.3 (1.59 s), 24: 271.6 (5.13 s) |
| `strata-Q2-b16-groups2-long` | 16: 199.5 (3.45 s) |
| `strata-Q2-b12-groups2-short` | 12: 269.8 (0.93 s), 24: 263.7 (2.91 s) |
| `strata-Q3-b12-share4-short` | 12: 366.4 (1.59 s), 24: 374.9 (4.13 s) |
| `strata-Q3-b12-share4-long` | 12: 258.9 (4.17 s) |
| `strata-Q1-dp2-short` | 1: 74.4 (1.11 s), 12: 152.5 (7.23 s), 24: 174.2 (7.17 s), 32: 182.3 (8.83 s) |
| `strata-Q1-dp2-long` | 16: 99.5 (25.39 s) |
| `strata-T12-trim-short` | 1: 108.7 (0.4 s), 12: 344.8 (2.03 s), 24: 366.2 (3.39 s) |
| `strata-T12-trim-long` | 12: 275.6 (4.35 s) |
| `strata-T16-trim-short` | 1: 110.6 (0.4 s), 16: 404.7 (2.32 s), 32: 389.1 (4.83 s) |
| `strata-T16-trim-long` | 16: 288.1 (5.37 s) |
| `strata-T20-trim-short` | 20: 409.4 (2.05 s) |
| `strata-T20-trim-long` | 20: 288.5 (5.35 s) |
| `strata-T24-trim-short` | 24: 416.5 (2.85 s) |
| `strata-T24-trim-long` | 24: 292.8 (5.34 s) |
| `strata-T16-pf1024-long` | 16: 280.7 (4.47 s) |
| `strata-T16-pf1024-short` | 16: 406.3 (2.33 s) |
| `strata-T16-pf2048-long` | 16: 290.5 (5.3 s) |
| `strata-T16-pf2048-short` | 16: 403.4 (2.15 s) |
| `strata-T16-trim-split-mt-warm` | 1: 11.6 (1.19 s) |
| `strata-X16-agent-trim-262k-mt-warm` | 1: 9.9 (1.41 s) |
| `strata-T16-direct-short` | 16: 405.3 (1.97 s), 24: 391.8 (3.37 s) |
| `strata-T16-direct-long` | 16: 294.5 (4.31 s) |
| `strata-X16-direct-mt-warm` | 1: 10.2 (1.43 s) |
| `strata-T16-split12-24-37-short` | 16: 399.2 (2.25 s) |
| `strata-T16-split12-24-37-long` | 16: 289.3 (4.47 s) |
| `strata-T16-split13-25-37-short` | 16: 397.3 (2.22 s) |
| `strata-T16-split13-25-37-long` | 16: 281.6 (4.27 s) |
| `strata-T16-pleram-short` | 16: 416.8 (2.03 s) |
| `strata-T16-pleram-long` | 16: 296.5 (4.32 s) |
| `strata-S16-spike-short` | 16: 412.9 (0.57 s) |
| `strata-S16-spike-long` | 16: 293.5 (1.18 s) |
| `strata-P16-timing` | 1: 6.1 (1.23 s) |
| `strata-P16-fused` | 1: 6.0 (1.23 s) |

### `results/multiturn.jsonl`

| Label | Users | tok/s | Follow-up turn TTFT p50 |
|---|---|---|---|
| `strata-T16-trim-split-mt-u8` | 8 | 49.2 | 7.05 s |
| `strata-T16-trim-split-mt-u16` | 16 | 42.8 | 10.4 s |
| `strata-X16-agent-trim-262k-mt-u8` | 8 | 57.1 | 4.76 s |
| `strata-X16-agent-trim-262k-mt-u16` | 16 | 72.8 | 4.95 s |
| `vllm031-full-offload20-mt-u8` | 8 | 89.2 | 6.37 s |
| `vllm031-full-offload20-mt-u16` | 16 | 76.7 | 32.18 s |
| `strata-X16-direct-mt-u16` | 16 | 62.3 | 4.47 s |

### `eval/eval_results.jsonl`

| Label | MMLU-Pro | GSM8K |
|---|---|---|
| `vllm-full-w4a16-fp8ple` | 70.4% | 94.7% |
| `strata-ud-iq4xs` | 77.9% | 94.0% |
| `vllm-qwen3.8-27b-fp8` | 72.1% | 95.3% |
| `vllm-razor65-nvfp4` | 62.5% | 92.7% |
