# StrataQ38FN on Linux: 4× RTX 3090

Measurements from 2026-10-10 to 2026-10-12 on the Linux host "Galactica": 4× RTX 3090 (24 GB each, PCIe effectively
~13 GB/s per card), Threadripper 3990X, 128 GB RAM (budget for tests: 96 GB), driver 580, CUDA 13.0. Current engine:
Strata **v0.1.42 + our patch series** (`patches/v0.1.42/`, 28 patches, self-compiled for sm_86), model **Unsloth
UD-IQ4_XS** (target); for comparison UD-Q4_K_XL, vLLM 0.29/0.31 and Qwen3.8-27B-FP8. Summary, Setup and Recommended
configuration describe the current state; the sections after them are the chronological record (0.1.41 → 0.1.42).

## Summary

Current default `configs/T16-142fcb.json` (CHANGELOG 0.12.0, TESTLOG #57). Protocol for the Strata numbers:
`ab_quick.sh` / `bench_concurrency.py`, **greedy**, 16 users, each sends 2 streamed requests in sequence, unique prefix
(no cache hit), prompt **~170 tokens ("short") / ~1.24K tokens ("1.2K")** (measured 162–178 / 1,232–1,243),
**512 output tokens fixed**, thinking off. The vLLM runs used the bench default temperature 0.7.

| | Strata UD-IQ4_XS (T16-142fcb) | vLLM 0.31 Flash-Next W4A16 | vLLM 0.31 Qwen3.8-27B FP8 |
|---|---|---|---|
| Decode, 1 user (p50) | **131 tok/s** short / 122 tok/s 1.2K prompt (MTP, greedy) | 79 tok/s; with MTP 136 tok/s (then max. 8 users) | 64 tok/s |
| Total, 16 users, short prompts | 522 tok/s (TTFT p50 1.0 s) | **622 tok/s** (0.7 s) | 485 tok/s |
| Total, 16 users, 1.2K prompts | **405 tok/s** (TTFT p50 2.0 s) | 386 tok/s (2.0 s) | 283 tok/s |
| Plateau (users) | 16 users (16 slots; more users only queue, #6, #47) | 710 / 398 tok/s (64 users, TTFT p50 31 / 50 s) | 838 / 409 tok/s (64 users) |
| Agent load, 16 users × 5 turns (11K→28K context): TTFT follow-up turn | **~4.6 s** (0.1.41 engine, #38; not re-measured on 0.1.42) | ~32 s | – |
| Prefill, 1 request | 1,250 (2K) → 2,240 tok/s (28K) (0.1.41, before B-tab) | ~3,500 tok/s | up to ~3,100 tok/s |
| Quality: MMLU-Pro (240) / GSM8K (150), thinking off | **77.9%** / 94.0% | 70.4% / 94.7% | 72.1% / 95.3% |
| VRAM | ~88 GB (all experts resident) | ~86 GB | ~63 GB (preset 0.68) |
| RAM (actual) | **~3–5 GB** RSS + ~50 GB page cache for the n-gram table (`--ple-io ram`) | ~64 GB pinned (FP8 n-gram table) | ~100 GB (80 GB KV offload) |

At equal load (16 users) Strata is ahead with 1.2K prompts (405 vs 386 tok/s) and behind with short prompts (522 vs
622 tok/s). vLLM's maximum (710 / 398 tok/s) needs 64 concurrent users, with a median wait of 31–50 s for the first
token. Strata is the best choice for **single users and agentic load**: fastest single stream, highest measured quality,
short wait times with many conversations and almost no private RAM. vLLM delivers more total throughput with many
concurrent *short* prompts, but with this W4A16 checkpoint its quality is measurably worse.

**Progress of the 16-user default** (total tok/s; note the protocol change between 0.9.0 and 0.10.0):

| Release | Config | Short | 1.2K | Protocol | TESTLOG |
|---|---|---|---|---|---|
| start | setup default, solo (0.1.41) | – | ~88 | T=0.7 | #1 |
| 0.4.0 | split 12,24,36 + trim, 16 slots, direct slot copy (0.1.41) | 405 | 294 | T=0.7, 1 run | #28 |
| 0.5.0 | `T16-direct-pleram` (0.1.41) | 417 | 297 | T=0.7, 1 run | #31 |
| 0.6.0 | `T16-mmqtab` (0.1.41 + B-tab) | 444 | 327 | T=0.7, 1 run | #38 |
| 0.9.0 | `T16-142` (0.1.42) | 467 | 342 | T=0.7, 3 runs | #46 |
| 0.10.0 | `T16-142fc` (fixed-cost switches) | 493 | 357 | greedy, 2 runs; same config as 0.9.0 greedy: 470 / 349 | #49 |
| 0.11.0 | `T16-142fca` (`--aux-cpus`) | 493 | 376 | greedy | #52 |
| 0.12.0 | `T16-142fcb` (split 12,24,37, MTP batch) | **522** | **405** | greedy | #57 |

The 0.9.0 → 0.10.0 step reads +5.6 % / +4.4 % across protocols; measured on one protocol (#49) the fixed-cost
switches give **+5.0 % / +2.2 %**.

**Headroom.** There is no fixed 600–620 tok/s decode ceiling: that was the engine-level decode rate before 0.12.0
(603 tok/s in #3 on 0.1.41, 616.6 rows/s for `T16-142fca` in #55). The current default decodes **~654 rows/s** at 16
slots (engine level, no HTTP), and **686** with batch self-commit (lever 4, exact at engine level, HTTP A/B pending).
Through the server, throughput follows `512 / (512/D + R)` with D = decode rate and R = all-slot stall per admission
(prompt read + ~51 ms fixed), which gives ~0.8 × D for short prompts. Delivered vs. open (sources: TESTLOG #57,
`upstream/PROFILE-next-levers.md`, `upstream/REVIEW-2026-10-12.md`):
- delivered in 0.12.0: levers 1–2 (split 12,24,37, batched draft-layer K/V) **+6.5 % short / +7.3 % 1.2K**;
- open, measured at engine level only: lever 3 (MMQ group sort) −65 ms per 1.2K read / −9.6 ms per ~150-token read,
  exact, model projection +5 % 1.2K / +1 % short; lever 4 (batch self-commit) D +4.9 %, projection +3.9 % / +3.0 %.
  Both together project to ~548 / ~441 tok/s; neither has been confirmed through HTTP yet.

## Setup (Linux)

```bash
git clone git@github.com:havok2-htwo/StrataQ38FN.git /media/raid0/StrataQ38FN
cd /media/raid0/StrataQ38FN
git clone https://github.com/Niko1221/Strata.git Strata

# 1. Upstream setup: Python venv, 94 GB model download, pack, MTP draft (it also installs an unpatched
#    engine, which the configs below do not use). Here it ran on 0.1.41; the 0.1.42 engine uses its pack and draft unchanged.
git -C Strata checkout v0.1.42
(cd Strata && ./setup.sh --yes --family unsloth --model UD-IQ4_XS --gpus all \
  --models-dir /media/raid0/Models/Strata-GGUF --no-start --no-browser --vision no)

# 2. Patched 0.1.42 engine in its own worktree (branch g3-142), pointer-table MMQ compiled in
git -C Strata worktree add ../Strata-142 -b g3-142 v0.1.42
git -C Strata-142 am ../linux/patches/v0.1.42/*.patch          # 28 patches, see "Patches" below
cmake -S Strata-142 -B Strata-142/build-142 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_MMQ_TAB=ON -DSTRATA_GGML_DIR=$PWD/Strata/third_party/llama.cpp   # GGML_DIR optional (empty: pinned commit)
cmake --build Strata-142/build-142 --target strata -j
cp Strata-142/build-142/strata Strata/engine/strata-g3-142fc

# 3. Start the default (T16-142fcb) with the server from Strata-142
linux/start-strata.sh
```

The configs run the engine `Strata/engine/strata-g3-142fc` with working directory `Strata` (pack, expert profile and MTP
draft from step 1). `start-strata.sh` runs `Strata-142/serve/server.py` (the server must match the 0.1.42 engine) with
the venv from step 1 in its own systemd scope and serves `http://127.0.0.1:8090/v1` (8080 is taken by Open WebUI
here). For LAN: `HOST=0.0.0.0 API_KEY=<secret> ./start-strata.sh`; another config: `CONFIG=configs/<name>.json`. Without
`-DSTRATA_MMQ_TAB=ON` the binary has no pointer-table MMQ: `STRATA_MMQ_DIRECT=1` then only logs a warning. The series
is tested on v0.1.42 only; upstream has since tagged v0.1.43 (+72 commits), not rebased yet.

**Patches** (`patches/v0.1.42/`, rebased from the 0.1.41 series in `patches/`, `patches/mmq-tab/`, `patches/overlap/`):
0001 + 0018 more than 8 batch slots with a layer split; 0002 direct GPU→GPU slot copy; 0003–0011 + 0021
pointer-table MMQ; 0012–0017, 0019–0020 overlapped admission (opt-in, off in the default); 0022 admission phase trace;
0023–0028 fixed per-admission cost switches (opt-in, on in the default).

Two pitfalls:
- **Without `--resident-experts`, Strata loads all experts (~60 GB) into RAM**, even though all 24,576 experts fit into the four
  cards (split 12,24,37) – that blows the budget and was killed by `systemd-oomd`. With the flag,
  0 GB of experts sit in RAM.
- Always start heavy processes in their own systemd scope (`run_strata_variant.sh`, `start-strata.sh`): otherwise a
  global OOM kills the whole editor scope (happened once).

## Recommended configuration

`configs/T16-142fcb.json` (default of `start-strata.sh`) = setup config plus:

| Setting | Value | Why |
|---|---|---|
| Engine (`exe`) | `strata-g3-142fc` | 0.1.42 + `patches/v0.1.42/` 0001–0028, built with `-DSTRATA_MMQ_TAB=ON` |
| `--resident-experts` | on | all experts in VRAM, no RAM copy |
| `"layer_split"` + `--trim-stage-weights` | **`12,24,37`** + on | cards hold layers 0–11 / 12–23 / 24–36 / 37–47 (+ head, sampling, MTP draft layer on the last card); each card loads only the dense weights of its layers, no expert borrowing. 12,24,37 instead of 12,24,36 takes one layer off the last card, which was 98 % busy in batch decode (#55, #56) |
| `"parallel"` | **16** | 16 batch slots in 4 pipeline groups of 4 rows (needs patches 0001 + 0018); 24 slots only help with more than 16 concurrent users (#47) |
| `--prefill` | **512** | prompt chunks of 512 tokens; smaller (#50) and larger (#25, #54) chunks are slower |
| `--mtp`, `--spec 4`, `--spec-min-p 0.5` | on | MTP drafting for a single request (+45 % decode, #9); batch slots decode one token per window without drafts |
| `--max-context 32768`, `--kv int8` | | context per slot, 8-bit KV cache |
| `--ple-io` | **ram** | n-gram table (~50 GB) through the page cache instead of direct I/O from the SSD: 1.2K prompt 821 instead of ~1,100 ms (#31) |
| `--aux-cpus` | **65-127** | the prefill stage threads leave the host thread's CPU 0 (SMT siblings of the idle pool cores): 1.2K read 557 → 484 ms, +5.4 % 1.2K throughput (#52) |
| `STRATA_MMQ_DIRECT=1` | env | pointer-table MMQ reads VRAM-resident experts in place instead of copying them per chunk (#37, #38) |
| `STRATA_SLOT_COPY_DIRECT=1` | env | slot state copied GPU→GPU (opt-in on 0.1.42, #45) |
| `STRATA_PIPE_BYIELD_NODRAIN=1` | env | a late BYIELD no longer drains the pipeline (#48) |
| `STRATA_ADMIT_MERGE_FIRST=1` | env | last prompt tokens and the first generated token in one window (#48) |
| `STRATA_CKPT_POOL=1`, `STRATA_CKPT_THREADS=1` | env | turn-checkpoint buffers recycled and saved on 4 per-card threads, 17 → 7 ms (#48) |
| `STRATA_SLOT_CKPT_MOVE=1` | env | the turn checkpoint is moved into the slot instead of copied (#48) |
| `STRATA_SLOT_COPY_BATCHSYNC=1` | env | one sync per card for the slot copy (#48) |
| `STRATA_SPLIT_MTP_BATCH=1` | env | upstream switch: the draft layer reads its prompt K/V batched (1.2K read 484 → 449 ms, #56); target tokens unchanged |

The six fixed-cost switches (`BYIELD_NODRAIN` … `SLOT_COPY_BATCHSYNC`) together cut the per-admission stall by ~25 ms
(#48; +5.0 % short / +2.2 % 1.2K through the server, #49).

**Agent load (long multi-turn conversations):** `configs/X16-agent-mmqtab.json` (0.1.41 engine `strata-g3-mmqtab`,
split 12,24,36) = `--max-context 262144 --kv-resident 32768 --conversation-cache-mib 24576 --conversation-cache-slots 24
--prompt-cache 24 --prompt-cache-every 8192`, env `STRATA_KV_PREFETCH=1`: KV in RAM, conversations stay parked,
follow-up turns read only the new part (#26, #38). Not re-measured on 0.1.42; an agent config on the current engine is
still open. The overlap option refuses `--kv-resident` configs.

**Correction:** `STRATA_BATCH_DECODE_SHARE` (previously recommended here at 2.0) has **no effect** with layer split and pipeline groups.
The code reads the prompt there in one piece (`generate.cpp:9706` upstream). The gain back then came solely from
`--prefill 512`.

## Strata measurement results (0.1.41, 2026-10-10/11)

Early measurements with the 0.1.41 engine (patches 1–2 at most), before pointer-table MMQ, the 0.1.42 rebase and the
12,24,37 split; current numbers are in the
Summary. Load profile (`bench_concurrency.py`): N users each send 2 streamed requests in sequence, unique prefix (no
cache hit), 512 output tokens, temperature 0.7, thinking off. "Long" = ~1,235 prompt tokens, "short" = ~165 (measured
162–171; earlier versions of this report said ~120, which was wrong).

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
603 tok/s versus 345 tok/s with 8. Built as `Strata/engine/strata-g3` (0.1.41); on 0.1.42 it is
`patches/v0.1.42/0001` + `0018`. Candidate for an upstream PR (`upstream/PR.md`, still the 0.1.41 draft).

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
vLLM level probably not. Upstream texts on this: `upstream/ISSUE.md`, `upstream/PR.md`. (State of 11.10.; the
overlap was built as an opt-in latency option, and the 0.12.0 default reaches 522 / 405 tok/s, see Summary.)

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

All Strata rows here are 0.1.41 engines; with pointer-table MMQ (`X16-agent-mmqtab.json`, #38) 16 agents reach 72 tok/s
with a follow-up TTFT of 4.57 s. The agent path has not been measured on 0.1.42.

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
except for one error path (fixed). Can be disabled with `STRATA_SLOT_COPY_DIRECT=0`. On 0.1.42 it is opt-in
(`STRATA_SLOT_COPY_DIRECT=1`, set in the current configs; `patches/v0.1.42/0002`).

## Profiling and kernel analysis (11.10.)

**Decode window** (`STRATA_VERIFY_PROFILE=1`, 16 slots in 4 groups): GPU time per window 5.1–6.1 ms (card 4
slowest: +0.73 ms output head), measured cycle ~6.95 ms → ~85% GPU work, ~1 ms host overhead. Moving one layer from
card 4 to card 3 (split `12,24,37` / `13,25,37`) brought **nothing** (399/289 and 397/282 tok/s, one run each): over HTTP
the limit was ingesting new prompts, not pipeline balance. **Superseded:** on 0.1.42 with pointer-table MMQ and
`--aux-cpus`, the same split 12,24,37 gave +5 % and is the default since 0.12.0 (TESTLOG #56/#57, section "Split
12,24,37" below); why #30 showed no gain was not re-checked at engine level.

**Overlapping admission, prototype** (`patches/experimental/`, `STRATA_PIPE_OVERLAP_SPIKE=1`, for measurement only): the prompt
is read on a worker thread while the slot groups keep decoding.

| | without | with prototype |
|---|---|---|
| Total throughput short / long | 417 / 297 | 413 / 294 |
| **TTFT p50 short / long** | 2.03 s / 4.32 s | **0.57 s / 1.18 s** |
| Decode during an ingestion | 0 | 180–206 rows/s (instead of ~620 then; the 0.12.0 default decodes ~654) |

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
experts in place (0.1.41 series; on 0.1.42: `patches/v0.1.42/0003–0011` + `0021`); only the address changes, the launch
geometry and arithmetic stay identical. Over-read bytes behind each
slot are covered by a 256 B guard that is zeroed per group. Build with `-DSTRATA_MMQ_TAB=ON` (off by default: costs
~36 MiB VRAM per GPU under eager module loading and ~70% of upstream MMQ compile time per architecture), enable at run
time with `STRATA_MMQ_DIRECT=1`.

- Exactness: unit test 874/874 bitwise, compute-sanitizer clean; engine A/B old vs new: tokens and fp32 logits
  byte-identical (4-GPU split, 1 GPU, streamed experts); batch == solo.
- Effect (16 users, HTTP): short 417 → **444 tok/s**, 1.2K prompts 296 → **327 tok/s**, 16 agents multi-turn
  62 → **72 tok/s** with follow-up TTFT 5.05 → 4.57 s; single-user TTFT for a 1.2K prompt 0.88 → 0.73 s.
- Applies to every VRAM-resident expert, so it also helps a single-GPU setup. Not built or tested here: HIP, SYCL,
  Windows (MSVC).

## Overlapped admission (engine patch series 4, opt-in)

Without it, every new request stops all 16 slots while its prompt is read (0.28 s for ~150 tokens, 0.67 s for 1.2K,
measured per admission). With `STRATA_PIPE_OVERLAP=1` (branch `g3-overlap`, `patches/overlap/0001-0005`, engine
`engine/strata-g3-overlap`, config `configs/T16-overlap.json`) the prompt is read on a worker thread while the slot
groups keep decoding. Before the read, every group layout the read could need is captured, so no CUDA graph is captured
or evicted during it; a gate checks live that every expert is resident, nothing is lent and no K/V streams, otherwise
it falls back to the old drained read with one log line. `--kv-resident` configs (the X16 agent configs) are refused.

- Exactness: env on = env off = reference engine, token- and resume-identical (batch_test16/32, interleave,
  `tools/batch_overlap_test.py` incl. BSTOP/STOP/BYIELD mid-read and slot reuse, 64-request soak, both splits,
  forced pad-up/wait arms); fault injection (a)-(e): exactly one ERR and a bounded exit.
- Effect through the server (3 repetitions, means; TESTLOG #41):

| 16 users | throughput | TTFT p50 | TTFT p95 | request time p50 |
|---|---|---|---|---|
| short prompts, off → on | 441 → 437 tok/s | **1.74 → 0.47 s** | 5.3 → 6.4 s | 18.5 → 17.0 s |
| 1.2K prompts, off → on | 327 → 328 tok/s | **4.13 → 1.00 s** | 11.0 → 13.7 s | 25.0 → 21.0 s |

  With 24 users (more users than slots) throughput rises by 2 % (short) and 7 % (1.2K), but the median TTFT gets worse.
- Why no throughput gain at 16 users: a read beside decoding takes longer (1.2K: 840 instead of 609 ms) and the decode
  rate drops meanwhile (418 ms of decode lost instead of 609), but the admitted slot itself idles for the longer read.
- Use it for interactive use with up to 16 users; the default (then `T16-mmqtab`, now `T16-142fcb`) stays without it until
  the p95 tail is fixed. On 0.1.42 the series is `patches/v0.1.42/0012–0017` + `0019–0020`.

## Engine 0.1.42 (rebased patch series, default in 0.9.0)

All patch series are rebased onto upstream Strata 0.1.42 (`patches/v0.1.42/`, worktree `Strata-142`, branch `g3-142`,
engine `strata-g3-142`, server `Strata-142/serve/server.py`, config `configs/T16-142.json`; superseded as default by
`T16-142fc` in 0.10.0, see below).
Exact (batch = solo, tokens equal to the 0.1.41 engines). The direct slot copy is opt-in there
(`STRATA_SLOT_COPY_DIRECT=1`, set in the config). Upstream's own changes give, at 16 users (3 repetitions, temperature 0.7, TESTLOG #46; greedy the same
`T16-142` gives 470 / 349 tok/s, #49):

| 16 users | 0.1.41 + B-tab | 0.1.42 | 0.1.42 + overlap |
|---|---|---|---|
| short prompts | 441 tok/s, TTFT p50 1.74 s | **467 tok/s**, 0.96 s | 462 tok/s, 0.43 s |
| 1.2K prompts | 327 tok/s, 4.13 s | **342 tok/s**, 2.60 s | 347 tok/s, 0.90 s |

More slots (#47): 24 slots raise throughput only ~2 %, but with more than 16 concurrent users they cut queueing
(TTFT p95 21 → 7 s at 24 users) at a lower per-user decode speed (21 instead of 31 tok/s): `configs/T24-142.json`.
`STRATA_ROUTE_TAIL_SKIP` (+12–13 % on the RTX 5090) does nothing here: it only skips experts missing from VRAM.

## Fixed per-admission cost (engine patch series 5, default since 0.10.0)

Every admission stalled all slots for ~112 ms besides the prompt read (drain, header windows, turn checkpoint, first
window, copies, refill). Six opt-in switches (`patches/v0.1.42/`, design `upstream/DESIGN-admission-fixed-cost.md`)
cut ~25 ms of it, exactly: the last prompt tokens and the first token in one window (`STRATA_ADMIT_MERGE_FIRST`),
checkpoint buffers recycled and saved on 4 per-card threads (`STRATA_CKPT_POOL`, `STRATA_CKPT_THREADS`, 17 → 7 ms),
the turn checkpoint moved into the slot instead of copied (`STRATA_SLOT_CKPT_MOVE`), one sync per card for the slot
copy (`STRATA_SLOT_COPY_BATCHSYNC`), and no drain for a late BYIELD (`STRATA_PIPE_BYIELD_NODRAIN`). Config
`configs/T16-142fc.json` (engine `strata-g3-142fc`): 16 users (greedy) **493 tok/s short, 357 tok/s
1.2K** (0.1.42 without them, same protocol: 470 / 349, i.e. +5.0 % / +2.2 %), TTFT p95 −21 % (TESTLOG #48–#49). Merging the first window saved less than
planned (7 instead of 22 ms): every extra row in a window costs 2.4–4 ms.

## CPU placement of the prefill threads (`--aux-cpus`, in the default since 0.11.0)

The engine pins its host thread to CPU 0, and every thread it creates inherits that CPU unless `--aux-cpus` is set -
including the three prefill stage threads of the layer split. Once 3–4 cards read a long prompt at once, their host-side
expert grouping queues on that one core (5 → 60–68 ms per card per 512-token chunk). `--aux-cpus 65-127` (the SMT
siblings of the idle pool cores) moves them off: a 1.2K read takes 484 instead of 557 ms, 16 users with 1.2K prompts
**376 instead of 357 tok/s**, short prompts unchanged, output identical (TESTLOG #52). Config `configs/T16-142fca.json`
= `T16-142fc` + this flag (default in 0.11.0; the flag stays in `T16-142fcb`). `auto` only finds CPU 64 here and gains
less; `65-79` is equivalent within noise (491.2 / 375.6 tok/s).

## Split 12,24,37 and batched draft-layer prompt K/V (default since 0.12.0)

The GPU profile (`upstream/PROFILE-next-levers.md`, TESTLOG #55) shows GPU3 at 98 % in batch decode: it holds the last
12 layers plus the head and the sampling, while GPU1/2 idle ~1.25 ms per 6.5 ms window period. Moving one layer to GPU2
(`layer_split 12,24,37`) and letting the draft layer read its prompt K/V batched (`STRATA_SPLIT_MTP_BATCH=1`, upstream
switch; tokens unchanged) gives, back to back (#57): 16 users **522 tok/s short (+6.5 %), 405 tok/s 1.2K (+7.3 %)**,
single user unchanged (~131 tok/s decode), output identical. Config `configs/T16-142fcb.json`, used by `start-strata.sh`.
`12,25,37` was equivalent within noise (517.2 / 396.0 vs 519.2 / 398.0 tok/s, #56). The same split showed no gain on
0.1.41 (#30, before pointer-table MMQ and `--aux-cpus`); that earlier result is superseded.

## MMQ group sort and self-committing batch windows (default since 0.13.0)

From the profile's ranked list (`upstream/PROFILE-next-levers.md`): `STRATA_MMQ_GROUP_SORT=1` sorts the full
16-expert MMQ groups of a prompt chunk by row count, keeping the partial group (the stream-k tail) as it is, so most
groups run with a narrower tile and the output stays bitwise identical (1.2K read 435 → 361 ms). `STRATA_BATCH_SELF_COMMIT=1`
lets a batch window with one row per slot advance its GDN state itself instead of running a second commit graph per
stage (decode 654 → 686 rows/s at 16 slots). Through the server (2 ABBA pairs, #60): 16 users **549 tok/s short (+7.3 %),
439 tok/s 1.2K (+10.6 %)**, every reply identical (reply hashes). Config `configs/T16-142gsc.json` (engine
`strata-g3-142lv`), used by `start-strata.sh`; agent load: `configs/X16-agent-142gsc.json` (split 12,25,37, because
with the agent's K/V the 12,24,37 split made CUDA2 borrow expert slots, #59).

## LAN dashboard

The repository's dashboard (`dashboard/`, the same one as on Windows) runs here too, with or without a Strata server:

```bash
linux/start-dashboard.sh                  # http://192.168.20.126:8082/ (binds 0.0.0.0, port 8082)
PORT=8083 linux/start-dashboard.sh        # another port; the script refuses a port that is in use
systemctl --user stop strata-dashboard.scope
```

- Own systemd user scope (`strata-dashboard.scope`, `MemoryMax=1G`), Python stdlib + psutil (system `python3`,
  else Strata's venv), ~3 % of one core. Log: `dashboard/dashboard-linux.out`; totals and power settings in
  `dashboard/state-linux.json` / `settings-linux.json` (not tracked; set the idle wall/GPU power on the page).
- Always live: all 4 GPUs (load, VRAM, power, temperature, SM clock, PCIe link gen/width now and max, PCIe RX/TX in
  MB/s from NVML), CPU, RAM, disks (whole disks and the md RAID volume, members not counted twice).
  NVML through Strata's `serve/telemetry.py` (`STRATA_SERVE`, default `Strata-142/serve`) or an own ctypes reader;
  read-only queries only, safe while benchmarks run.
- With a Strata server: live request (phase, answering progress tokens/max, tok/s, prompt-reading progress), every
  `--batch` slot (reading / decoding / idle, tokens, tok/s), speeds, caches, recent requests, decode timing from the
  engine log (the running server's config `"log"`, else `CONFIG`'s). Sources: `SOURCES` (default
  `http://127.0.0.1:8090 http://127.0.0.1:8080`) plus every running `server.py --engine` found in the process list;
  port 8080 is Open WebUI here and is recognised as "not a Strata server". `API_KEY_FILE` for a server with a key
  (the key stays on this machine). Only `GET /metrics` is sent, never a request that runs the model.
- Engine offline: the engine panels show "offline · last seen" and the last known values; the rest stays live.

## Engine 0.1.43 and multi-slot QSA attention (default since 0.14.0)

The series is rebased onto upstream 0.1.43 (`patches/v0.1.43/`, worktree `Strata-143`; tokens byte-identical to the
0.1.42 build, #61). Upstream's new NaN guard costs ~2 % on our pipelined batch windows (#62) and is switched off in our
configs (`STRATA_NAN_GUARD=0`). Lever 5, `STRATA_QSA_SLOTS=1`, runs the QSA block scores, top-k and attention of a batch
window in one launch over all its slots instead of one launch per slot (bitwise identical; decode +6.3 % at 16 slots).
Through the server (2 ABBA pairs, #64): 16 users **578 tok/s short, 456 tok/s 1.2K** (+5.3 % / +4.1 %), every reply
identical. Config `configs/T16-143l5.json` (engine `strata-g3-143l5`, server `Strata-143/serve/server.py`), used by
`start-strata.sh`; agent load `configs/X16-agent-143l5.json`.

## KV first: 64K per slot and the conversation cache (default since 0.15.0)

A KV audit (TESTLOG #65) found a re-prefill cascade with more conversations than slots: in the old default the
conversation cache was off, so every follow-up turn was read again from the start (24 agents × 4 turns: 91.5 tok/s,
follow-up TTFT p50 11.3 s); in the old agent config 52.6 GiB of pinned streamed K/V left the conversation cache no
room inside the 90 GiB scope. The default now gives every slot **64K context in VRAM** (2× before, free in speed) and
parks finished conversations in a **48 GiB RAM conversation cache**, plus the pipelined stage hand-off (`STRATA_HANDOFF_PIPE=8`,
exact, no VRAM): multi-turn **176.8 tok/s, follow-up TTFT 5.6 s (1.9×)**. Parking cost ~6 % in one-shot traffic until
stage 1 of the KV tiering (TESTLOG #68): `STRATA_PARK_ASYNC=1` captures the parked state on one thread per GPU while
the slots decode, so the default `configs/T16-143-64khoca.json` (engine `strata-g3-143kt`, used by `start-strata.sh`)
reaches 573 / 451 tok/s one-shot (no cache: 575 / 454) and 182.7 tok/s multi-turn. 80K per slot no longer fits
without expert loans (−8 %); `--kv-resident` streaming costs −30 % decode.

## License notices

The engine patches contain Strata source (MIT) and, in the pointer-table MMQ series, code mirrored from
llama.cpp/ggml (MIT); both notices are in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The rebased series
(`patches/v0.1.42/`) also carries the ggml notice in the mirrored file itself.

## Files

| File | Contents |
|---|---|
| `TESTLOG.md` | **chronological log of all tests with verdict**: check here before every new test |
| `start-strata.sh` | start the recommended configuration |
| `start-dashboard.sh` | start the LAN dashboard (port 8082, own scope; see LAN dashboard) |
| `run_strata_variant.sh` | start a config variant, run the benchmark, stop |
| `bench_concurrency.py` | load benchmark (OpenAI-compatible, streaming, TTFT/decode/total) |
| `ab_quick.sh` | quick A/B protocol since #49: one server start per arm, 16 users, greedy, short + 1.2K |
| `series_final.sh` | last measurement series (Q4_K_XL, 3 GPUs, MTP off, users 24–40, prefill) |
| `configs/` | all measured Strata configs |
| `results/sweep.jsonl`, `results/*.txt` | all measurement results |
| `patches/` | Strata engine patches. **Current: `patches/v0.1.42/` (28 patches on v0.1.42, see Setup)**; the 0.1.41 series: `0001`/`0002` (more than 8 slots, direct slot copy), `mmq-tab/`, `overlap/`; `experimental/`: overlap prototype (measurement only) |
| `bench_multiturn.py` | agent/multi-turn benchmark |
| `eval/` | quality set (MMLU-Pro + GSM8K) with answers and results |
| `upstream/` | issue/PR drafts for Niko1221/Strata (not yet posted; written for 0.1.41, to be redone on the current base), designs (`DESIGN-*.md`), code analyses (`ANALYSIS-*.md`), GPU profile and lever ranking (`PROFILE-next-levers.md`), methodology/patch review (`REVIEW-2026-10-12.md`) |
| `quant-survey/` | evaluation of all 910 Flash-Next repos on Hugging Face |
| `vllm/` | vLLM start scripts and PLE FP8 patches |
