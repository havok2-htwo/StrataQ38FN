# StrataQ38FN on Linux: 4× RTX 3090

Measurements from 2026-10-10/11 on the Linux host "Galactica": 4× RTX 3090 (24 GB each, PCIe effectively ~13 GB/s per card),
Threadripper 3990X, 128 GB RAM (budget for tests: 96 GB), driver 580, CUDA 13.0. Strata v0.1.41 (self-compiled
for sm_86), model **Unsloth UD-IQ4_XS** (target), for comparison UD-Q4_K_XL, vLLM 0.29/0.31 and Qwen3.8-27B-FP8.

## Summary

| | Strata UD-IQ4_XS | vLLM 0.31 Flash-Next W4A16 | vLLM 0.31 Qwen3.8-27B FP8 |
|---|---|---|---|
| Decode, 1 user | **~121 tok/s** (MTP) | 79 tok/s; with MTP 136 tok/s (then max. 8 users) | 64 tok/s |
| Total, 1.2K prompts | ~327 tok/s (16 slots, plateau) | ~400 tok/s | 409 tok/s (64 users) |
| Total, short prompts | ~444 tok/s (16 slots) | ~710 tok/s | 838 tok/s (64 users) |
| Agent load, 16 users × 5 turns (11K→28K context): TTFT follow-up turn | **~4.5–5 s** | ~32 s | – |
| Prefill | 1,250 (2K) → 2,240 tok/s (28K) | ~3,500 tok/s | up to ~3,100 tok/s |
| Quality: MMLU-Pro (240) / GSM8K (150), thinking off | **77.9%** / 94.0% | 70.4% / 94.7% | 72.1% / 95.3% |
| VRAM | ~88 GB (all experts resident) | ~86 GB | ~63 GB (preset 0.68) |
| RAM (actual) | **~3–5 GB** (agent config: + KV/conv. cache) | ~64 GB (FP8 n-gram table) | ~100 GB (80 GB KV offload) |

Strata is the best choice for **single users and agentic load**: fastest single stream, highest measured
quality, short wait times with many conversations and almost no RAM. vLLM delivers more total throughput with many concurrent
*new* prompts, but with this W4A16 checkpoint its quality is measurably worse.

## Setup (Linux)

```bash
git clone git@github.com:havok2-htwo/StrataQ38FN.git /media/raid0/StrataQ38FN
cd /media/raid0/StrataQ38FN
git clone https://github.com/Niko1221/Strata.git Strata
git -C Strata am ../linux/patches/*.patch ../linux/patches/mmq-tab/*.patch            # >8 batch slots + direct slot copy (see below); rebuild the engine afterwards
cd Strata
./setup.sh --yes --family unsloth --model UD-IQ4_XS --gpus all \
  --models-dir /media/raid0/Models/Strata-GGUF --no-start --no-browser --vision no
```

The setup compiles the engine (no prebuilt for 0.1.41, ~10 min with CUDA 13), downloads 94 GB and builds the pack and
MTP draft. After that, `linux/start-strata.sh` starts the recommended configuration on `http://127.0.0.1:8090/v1`
(8080 is taken by Open WebUI here). For LAN: `HOST=0.0.0.0 API_KEY=<secret> ./start-strata.sh`.

Two pitfalls:
- **Without `--resident-experts`, Strata loads all experts (~60 GB) into RAM**, even though all 24,576 experts fit into the four
  cards (split 12/24/36) – that blows the budget and was killed by `systemd-oomd`. With the flag,
  0 GB of experts sit in RAM.
- Always start heavy processes in their own systemd scope (`run_strata_variant.sh`, `start-strata.sh`): otherwise a
  global OOM kills the whole editor scope (happened once).

## Recommended configuration

`configs/T16-direct-pleram.json` (throughput, default of `start-strata.sh`) or `configs/X16-agent-direct.json` (agent load) = setup config plus:

| Setting | Value | Why |
|---|---|---|
| `--resident-experts` | on | all experts in VRAM, no RAM copy |
| `"layer_split"` + `--trim-stage-weights` | **`12,24,36`** + on | each card loads only the dense weights of its layers and keeps its own prompt buffers: no expert borrowing per request, **+14% short / +16% long** |
| `"parallel"` | **16** (12–24) | plateau from 16; 24 slots +2%, all experts stay resident |
| `--ple-io` | **ram** | n-gram table (~50 GB) in RAM instead of direct I/O from the SSD: 1.2K prompt 821 instead of ~1,100 ms, +1–3% throughput (`mlock` fails without a higher `ulimit -l`, the pages stay in the page cache anyway) |
| `--prefill` | **512** | short prompts 1.6× faster than `auto` (1,247 instead of 795 tok/s at 2K), better up to 15K |
| MTP (`--mtp`, `--spec 4`) | on | +45% decode (117 instead of 81 tok/s), 61% draft acceptance at temperature 0.7 |
| Engine | `engine/strata-g3b` | with both patches (>8 slots, direct slot copy) |
| Additionally for agent load | `--max-context 262144 --kv-resident 32768 --conversation-cache-mib 24576 --conversation-cache-slots 24 --prompt-cache 24 --prompt-cache-every 8192`, env `STRATA_KV_PREFETCH=1` | KV in RAM, conversations stay parked: follow-up turns read only the new part |

**Correction:** `STRATA_BATCH_DECODE_SHARE` (previously recommended here at 2.0) has **no effect** with layer split and pipeline groups.
The code reads the prompt there in one piece (`generate.cpp:9706` upstream). The gain back then came solely from
`--prefill 512`.

## Strata measurement results

Load profile (`bench_concurrency.py`): N users each send 2 streamed requests in sequence, unique prefix (no
cache hit), 512 output tokens, temperature 0.7, thinking off. "Long" = ~1,230 prompt tokens, "short" = ~120.

**Batching variants (total tok/s):**

| Configuration | 1 user | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| Setup default, solo (long) | 92 | 88 | 90 | – | – |
| `parallel 8` (long) | 86 | 177 | 171 | 174 | 176 |
| 16 slots, chunks 512, share 2.0 (long) | 96 | 209 | 248 | 234 | 246 |
| 16 slots, chunks 512, share 2.0 (short) | 104 | 281 | 358 | 365 | 354 |
| 32 slots, chunks 512, share 2.0 (short) | – | – | 319 | 342 | 341 |

**Fine-grained slot steps (short prompts, users = slots / 1.5× slots):**

| Slots | Rows per window | tok/s | tok/s (1.5×) | per user | Experts in VRAM |
|---|---|---|---|---|---|
| **12** | 3 | **361** | 336 | **34** | 100% |
| 16 | 4 | 348 | 358 | 24 | ~100% |
| 20 | 5 | 336 | 327 | 19 | 99% |
| 24 | 6 | 331 | 328 | 15 | 98% |
| 28 | 7 | 328 | 328 | 13 | 96% |

More users than slots only wait longer (12 slots, 24–40 users: 346–358 tok/s, TTFT p95 37 → 72 s).

**Prefill across prompt length (1 request):**

| Prompt | Chunks 512 | `auto` | `auto:32768` | vLLM 0.31 |
|---|---|---|---|---|
| ~2K | **1,247** | 795 | 790 | 3,471 |
| ~8K | **1,913** | 1,260 | 1,202 | 3,498 |
| ~15K | **2,120** | 1,848 | 1,359 | 3,500 |
| ~28K | 2,239 | **2,289** | 1,368 | 3,505 |

Prefill is Strata's bottleneck under load: a prompt that fits into one chunk is read in one piece and stalls
all batch slots during that time.

**Other variants:**

| Variant | 1 user | Batch (12 slots, short) | Note |
|---|---|---|---|
| UD-IQ4_XS, 4 GPUs | 117–121 | 361 | reference |
| UD-IQ4_XS, **3 GPUs** (`--trim-stage-weights`, split 16/32) | 117 | 266 | fourth card only matters under load |
| **UD-Q4_K_XL**, 4 GPUs | 100 | 241 (long: 150) | 97% of experts in VRAM, 1.9 GB in RAM |
| UD-IQ4_XS without MTP | 81 | – | |

PCIe (all cards ~13 GB/s via risers) plays practically no role for IQ4_XS and Q4_K_XL: experts sit in VRAM,
and during decode only a small hand-off buffer moves from card to card per window.

## More than 8 batch slots (engine patch)

Strata limits batch slots to 8 (a verify window has at most 8 rows). With layer split, however, the slots run
through the cards in groups, and each group window carries only `slots / groups` rows. The patch in
`patches/0001-*.patch` therefore allows up to 8 rows **per group** (up to 32 slots with 4 GPUs) and sizes the
hand-off buffer between the cards per slot. Verified with Strata's own exactness test (`tools/batch_test16.py`,
`batch_test32.py`): **16 and 32 slots produce token-for-token the same outputs as solo**; batch decode with 16 slots
603 tok/s versus 345 tok/s with 8. Built as `Strata/engine/strata-g3`. Candidate for an upstream PR (`upstream/PR.md`).

## vLLM for comparison

Qwen3.8-Flash-Next runs in vLLM only as a W4A16 checkpoint (`VnimanieAI/Qwen3.8-Flash-Next-W4A16`, 67 GB experts +
102 GB n-gram table in bf16) and on 4× 24 GB needs:
- `--enable-expert-parallel` (640 / TP4 is not divisible by the group size 128; as a side effect Marlin MoE instead of Triton),
- the n-gram table as **FP8** in pinned RAM (~51 GB instead of 102 GB): `vllm/patch031/sitecustomize.py` (for 0.31; older
  version for 0.29 in `vllm/patch/`), scale from `vllm/ple_scan.py` (absmax 0.0894 → 1.995e-4). The patch pins exactly
  (Torch rounds pinned memory up to powers of two) and runs the Triton lookup over uint8 views, because
  Triton cannot load e4m3 on Ampere. Start: `vllm/run_vllm031_flashnext.sh`.

| vLLM Flash-Next | 1 user | 12 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 0.29, prefix caching on (long) | 65 | – | 376 | 384 | – |
| 0.29, prefix caching off (long) | 75 | – | 375 | 418 | 442 |
| 0.31, prefix caching on (long) | 75 | 366 | 386 | 398 | 398 |
| 0.31, prefix caching on (short) | 78 | 499 | 622 | 706 | 710 |

With MTP (`SPEC=2`, `--max-num-seqs 8`, `--max-num-batched-tokens 1024`, utilization 0.93) it does fit after all: **136 tok/s
single**, 490 tok/s with 8 users (short), mean acceptance length 2.2. But then only 65K KV tokens and max. 8 users.

Without MTP, vLLM holds at most ~22 requests concurrently (the shared KV/Mamba block pool is full); MTP does not fit into
VRAM with 64 sequences (OOM at 0.90 utilization, at 0.86 no room left for the KV cache).

## Batching optimization (night of 10./11.10.)

| Attempt | short | long | Result |
|---|---|---|---|
| Reference: 16 slots, split auto, chunks 512 | 348–358 | 248 | |
| 16 slots in 2 instead of 4 groups (8 rows/window) | 266–272 | 200 | ❌ half the cards idle |
| Decode share 4.0 | 366–375 | 259 | ➖ noise (no effect in pipeline mode) |
| 2 instances of 2 GPUs each (round-robin) | 152–182 | 100 | ❌ half the experts in RAM, slow prefill |
| **Split 12/24/36 + trim, 16 slots** | **405** | **288** | ✅ no more buffer borrowing |
| … 20 / 24 slots | 409 / 416 | 288 / 293 | ➖ plateau |
| … chunks 1,024 / 2,048 | 406 / 403 | 281 / 290 | ➖ |
| … + direct slot copy (patch 2) | 405 | 294 | ✅ small (+2%) |

Code analysis (4 analysts + 4 cross-checkers over `generate.cpp`, `verify.cpp`, `server.py`): the rest of the gap to vLLM
comes from blocking prompt ingestion (admissions run one at a time and drain the pipeline) and from the fixed-cost share per
batch window (~6.6 ms instead of the ~2 ms bandwidth limit). Estimated engine rework: overlapping admission 4–10 days, MTP in
batch with layer split 8–16 days, window overhead 5–14 days. Realistic target afterwards ~450–550 / ~300–350 tok/s,
vLLM level probably not. Upstream texts on this: `upstream/ISSUE.md`, `upstream/PR.md`.

## Agent load (multi-turn)

`bench_multiturn.py`: N users, 5 turns each, turn 1 with an ~8K document, each further turn appends the answer + ~3K new
tokens (context 11K → 28K); the users run interleaved like main agents and sub-agents.

| | 8 users: tok/s | 8: TTFT follow-up turn | 16 users: tok/s | 16: TTFT follow-up turn |
|---|---|---|---|---|
| Strata trim config (32K, without conv. cache) | 49 | 7.1 s | 43 | 10.4 s |
| **Strata agent config** (262K, KV in RAM, 24 GB conv. cache) | 57 | **4.8 s** | 73 | **5.0 s** |
| Strata agent config + direct slot copy | – | – | 62* | **4.5 s** |
| vLLM 0.31 (prefix cache + 20 GB RAM KV tier) | **89** | 6.4 s | 77 | 32 s |

\* without a pre-warmed conversation cache (the 73 run benefited in its first turn from the preceding 8-user run).

## Quality (own set)

`eval/quick_eval.py`: 240 MMLU-Pro questions (all 14 subject areas) + 150 GSM8K, greedy, thinking off, identical for all.

| Model / setup | MMLU-Pro | GSM8K |
|---|---|---|
| **Flash-Next UD-IQ4_XS (Strata)** | **77.9%** | 94.0% |
| Qwen3.8-27B FP8 (vLLM) | 72.1% | 95.3% |
| Flash-Next W4A16 + FP8 n-gram table (vLLM) | 70.4% | 94.7% |
| Flash-Next RAZOR-65B (50% of experts pruned, NVFP4, vLLM) | 62.5% | 92.7% |

On MMLU-Pro, only Strata gets 23 questions right, only vLLM-W4A16 gets 5 (significant). The cause is probably the
third-party W4A16 checkpoint or vLLM's Ampere path, not MTP (lossless with greedy). RAZOR was dropped afterwards.

## Direct slot copy (engine patch 2)

`copy_to_slot`/`copy_from_slot` now copy the slot state directly on the GPU (`cudaMemcpyAsync`, one sync per
card) instead of via a host image with a sync per layer: 65.6 → 34.0 ms per admission (pure copy ~8 ms), bit-identical to the
solo run (`batch_test16.py` 16/16, `batch_interleave_test.py` incl. follow-up turn from the slot), two reviews without findings
except for one error path (fixed). Can be disabled with `STRATA_SLOT_COPY_DIRECT=0`.

## Profiling and kernel analysis (11.10.)

**Decode window** (`STRATA_VERIFY_PROFILE=1`, 16 slots in 4 groups): GPU time per window 5.1–6.1 ms (card 4
slowest: +0.73 ms output head), measured cycle ~6.95 ms → ~85% GPU work, ~1 ms host overhead. Moving one layer from
card 4 to card 3 (split `12,24,37` / `13,25,37`) brought **nothing** (399/289 and 397/282 tok/s): over HTTP the limit is
ingesting new prompts, not pipeline balance.

**Overlapping admission, prototype** (`patches/experimental/`, `STRATA_PIPE_OVERLAP_SPIKE=1`, for measurement only): the prompt
is read on a worker thread while the slot groups keep decoding.

| | without | with prototype |
|---|---|---|
| Total throughput short / long | 417 / 297 | 413 / 294 |
| **TTFT p50 short / long** | 2.03 s / 4.32 s | **0.57 s / 1.18 s** |
| Decode during an ingestion | 0 | 180–206 rows/s (instead of 620) |

Prefill and decode share the cards almost completely (Ampere cannot preempt running kernels): **no
throughput gain, but −70% wait time to the first token.** Safe implementation (PLE thread safety, graph capture,
error paths): `upstream/DESIGN-overlap.md`, 6 milestones.

**Prefill of short prompts** (`STRATA_PREFILL_TIMING=1`): a 154-token prompt costs 78–87 ms per card, the 4 cards
run one after another → ~330–410 ms. ~31% of that is the "dequant" phase, which is actually a **copy**: llama.cpp's
MMQ needs all experts of a group in one array with a fixed stride, the resident experts are scattered across the
cache, so every touched expert (2.33 MB) is copied per chunk and layer (~2 GB of memory traffic per layer).
`STRATA_PF_FUSED=1` only kicks in from 1,024-token chunks and does not cover Q8_0 downs on CUDA. Possible fixes (~2–3 days each):
decode kernel `native_expert_grouped` for experts with ≤ 8 rows, or MMQ directly on the cache slots (pointer table).
Expectation: short prompts −25–33% ingestion time, throughput +5–10%, also helps on a single GPU.

## Pointer-table MMQ ("B-tab", engine patch series 3)

For prompt chunks below 1,024 tokens Strata copied every routed, VRAM-resident expert (2.33 MB) into a contiguous group
buffer before llama.cpp's MMQ could use it (~2 GB DRAM traffic per layer). The patch series in `patches/mmq-tab/` adds a
Strata copy of MMQ's outer stream-k kernel that takes a per-expert pointer table (16 kernel parameters), so MMQ reads the
experts in place; only the address changes, the launch geometry and arithmetic stay identical. Over-read bytes behind each
slot are covered by a 256 B guard that is zeroed per group. Build with `-DSTRATA_MMQ_TAB=ON` (off by default: costs
~36 MiB VRAM per GPU under eager module loading and ~70% of upstream MMQ compile time per architecture), enable at run
time with `STRATA_MMQ_DIRECT=1`.

- Exactness: unit test 874/874 bitwise, compute-sanitizer clean; engine A/B old vs new: tokens and fp32 logits
  byte-identical (4-GPU split, 1 GPU, streamed experts); batch == solo.
- Effect (16 users, HTTP): short 417 → **444 tok/s**, 1.2K prompts 296 → **327 tok/s**, 16 agents multi-turn
  62 → **72 tok/s** with follow-up TTFT 5.05 → 4.57 s; single-user TTFT for a 1.2K prompt 0.88 → 0.73 s.
- Applies to every VRAM-resident expert, so it also helps a single-GPU setup. Not built or tested here: HIP, SYCL,
  Windows (MSVC).

## Files

| File | Contents |
|---|---|
| `TESTLOG.md` | **chronological log of all tests with verdict**: check here before every new test |
| `start-strata.sh` | start the recommended configuration |
| `run_strata_variant.sh` | start a config variant, run the benchmark, stop |
| `bench_concurrency.py` | load benchmark (OpenAI-compatible, streaming, TTFT/decode/total) |
| `series_final.sh` | last measurement series (Q4_K_XL, 3 GPUs, MTP off, users 24–40, prefill) |
| `configs/` | all measured Strata configs |
| `results/sweep.jsonl`, `results/*.txt` | all measurement results |
| `patches/` | Strata engine patches: more than 8 slots, direct slot copy; `patches/experimental/`: overlap prototype (measurement only) |
| `bench_multiturn.py` | agent/multi-turn benchmark |
| `eval/` | quality set (MMLU-Pro + GSM8K) with answers and results |
| `upstream/` | issue/PR text for Niko1221/Strata (not yet posted), design of overlapping admission, detailed code analyses (`ANALYSIS-*.md`) |
| `quant-survey/` | evaluation of all 910 Flash-Next repos on Hugging Face |
| `vllm/` | vLLM start scripts and PLE FP8 patches |
