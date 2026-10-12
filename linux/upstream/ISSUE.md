> **Status:** drafts for 0.1.41 (fb58e0d), to be redone on the current base (0.1.42 + `../patches/v0.1.42/`, default `T16-142fcb`). Numbers, patch paths and engine names here are superseded; see `REVIEW-2026-10-12.md`. The interleave part-4 failure under Finding 1 still reproduces on 0.1.42.

# Batching on a 4-GPU layer split (4x RTX 3090): measurements, two patches, and three findings

Hi, thanks for Strata. We ran it as a multi-user server on 4x RTX 3090 (24 GB each, PCIe ~13 GB/s per card, no P2P), a Threadripper 3990X and 128 GB RAM, with Qwen3.8-Flash-Next UD-IQ4_XS. Engine 0.1.41 (fb58e0d), compiled for sm_86 with CUDA 13.0. All 24,576 experts fit VRAM with the split 12/24/36. Below are the measurements, two small patches that are ready to upstream, and three findings that may be worth a doc fix or a follow-up.

## Setup and numbers

Load generator: N clients, each sending 2 streamed requests back to back. Every request has a unique prefix (no cache hit), 512 output tokens, temperature 0.7, thinking off. "Short" prompts are ~120 tokens, "long" ones ~1,230. The table shows total output tok/s at the plateau.

| config | 1 client | short | long |
|---|---|---|---|
| solo (no `parallel`) | 117-121 | ~88 | ~88 |
| `parallel` 8, auto split, `--prefill auto` | 120 | ~177 | ~175 |
| 16 slots (patch 1), auto split, `--prefill 512` | 121 | ~350-360 | ~248 |
| **16 slots, `layer_split 12,24,36` + `--trim-stage-weights`, `--prefill 512`** | **121** | **~405** | **~288** |
| same + direct slot copy (patch 2) | 121 | ~405 | ~294 |

Pure batch decode (`tools/batch_test16.py`, no admissions) reaches 603-640 tok/s with 16 slots in 4 groups of 4. The gap between that and the ~405 seen through the server is almost entirely admissions (see finding 3).

## Patch 1: more than 8 slots with a layer split

`--batch` is capped at `kVerifyMaxT` (8) rows. With `--batch-groups`, however, a group's window carries only `batch / groups` rows. The patch allows up to 8 rows **per group**, i.e. up to 8 x stages slots, and keeps the 8-row cap for one group or without a split. To make that work it does two things:
- sizes the verifiers by `min(batch, kVerifyMaxT)`;
- sizes the inter-stage hand-off buffer per slot, because groups write at `hbase = group * rows`; with more than 8 slots this overflowed into an illegal memory access.

**Verification:** `tools/batch_test16.py` and `batch_test32.py` (`batch_test.py` with 16/32 questions) produce output identical to solo for all 16 and all 32 slots, with `--layer-split 12,24,36 --batch-groups 4 --pcie-frac 0 --adapt-every 1000000`.

**Throughput:** On this box 12-16 slots is the sweet spot. Above that, the slot sessions start to evict experts, unless you use trim.

## Patch 2: device-to-device slot copy

`copy_to_slot` / `copy_from_slot` currently go through a host image: checkpoint save/restore plus `conversation_kv_save`/`restore` per QSA layer, with a device sync per layer. However, a stage's main session and its slot sessions sit on the same device and are carved alike. The patch adds `conversation_checkpoint_copy` and `conversation_kv_copy`. Both use `cudaMemcpyAsync` on the legacy stream and write the same bytes to the same places in the same order. The residency refresh for streamed K/V is shared with `conversation_kv_restore`. Each device syncs once before and once after, and on an error path the queued copies are drained.

**Verification:** `batch_test16.py` gives 16/16 identical. `batch_interleave_test.py` gives A, B, C and "A's next turn from its slot" identical. `STRATA_SLOT_COPY_DIRECT=0` keeps the old path.

**Effect:** admission copy drops from 65.6 to 34.0 ms on average; the pure copy is ~8 ms, the rest is waiting for in-flight windows. That gives +2% on long prompts and -10% follow-up TTFT in a multi-turn test. The gain is small but consistent.

## Finding 1: decode share and BYIELD are inert with `--batch-groups` > 1

`read_part` returns `sp.run(...)` whenever `piped` (generate.cpp:9706 at fb58e0d). So the `STRATA_BATCH_DECODE_SHARE` loop (9688ff) and the BYIELD scan never run with a pipelined split. Since 0.1.41 that is the default for a split with `--batch`. BATCHING.md line 79 ("the slots decode between the prompt's chunks") does not hold there. We measured decode share 0.5/1.0/2.0/4.0 and the differences were noise.

Related: in `tools/batch_interleave_test.py` the "D gave way" step fails on a split with `ERR BGEN: no such free slot (--batch 3)`, with and without patch 2.

## Finding 2: prompt-buffer loans cost ~60-70 ms per lending card per admission

With 16 slot sessions, CUDA0 and CUDA3 had to lend expert slots to the prompt path (166-170 slots, 0.39 GiB) and refill them after every request. With `layer_split 12,24,36` and `--trim-stage-weights` (your documented recipe in BATCHING.md), every card keeps its own prompt buffers and residency stays at 100%. That alone gave +14% on short prompts and +16% on long ones, which is the biggest single win we found. Setup could recommend or apply this automatically whenever `parallel` is used with a split.

## Finding 3: admissions dominate multi-user throughput on a split

Admissions are serialized and drain the pipeline (any stdin line except BSTOP calls `pipe_drain`). A short prompt that fits in one chunk then runs through the 4 cards one after another while all slots wait. Measured throughput fits `L / (L/R + T_adm)` well, with L = 512 output tokens, R = 603 rows/s and T_adm ≈ 0.57 s for short and ≈ 1.2 s for long prompts. We looked at what it would take to overlap admissions with decode: the batched prompt read runs on a worker thread while `pump` keeps the groups going. Our estimate is 4-10 days of work. Known blockers are that `PleTable::gather_batch` and the stage-0 `ple.scratch` are not thread-safe against the batch windows, and adaptive swaps must be held during the read. We can share the detailed notes and file:line references if useful.

## Finding 4: short-prompt prefill spends ~31% in an expert copy for MMQ

With `STRATA_PREFILL_TIMING=1`, a 154-token prompt takes 78-87 ms of GPU time per card (12 layers), and the 4 cards run one after another. About 31% of that time is the phase timed as "dequant". It is actually `gather_native`/`copy16_kernel`: each routed, resident expert blob (2.33 MB) is copied into a contiguous 16-expert group, because `mul_mat_q` addresses expert z as `x + z*stride`. That is ~2 GB of DRAM traffic per layer for prompts below 1024 tokens. `STRATA_PF_FUSED=1` does not engage at `--prefill 512` (it needs the streamed walk), and on CUDA it does not cover Q8_0 downs. Since each layer's slots are a contiguous range of equally sized blobs, one option is to pad those slots to a uniform stride that is a multiple of the block sizes (+3-6% VRAM). MMQ could then read the cache directly. The other option is to route experts with at most 8 rows through `native_expert_grouped`. Notes with file:line references: `ANALYSIS-short-prompt-moe.md` in this folder (also `ANALYSIS-batching-levers.md` and `DESIGN-overlap.md`).

**Follow-up (patch series 3, "pointer-table MMQ"):** we implemented a third option that needs no padding. A Strata copy of MMQ's outer stream-k kernel takes a per-expert pointer table (16 kernel parameters), so MMQ reads the resident experts in place. Only the address changes; launch geometry and arithmetic are identical. A 256 B guard behind each sized slot (zeroed per group) covers MMQ's over-read. It is behind `-DSTRATA_MMQ_TAB=ON` (off by default) and `STRATA_MMQ_DIRECT=1`. Exactness: the unit test is 874/874 bitwise, and engine A/B gives byte-identical tokens and fp32 logits (4-GPU split, 1 GPU, streamed experts). Effect: prompt read −18..−24%; with 16 users, 417 → 444 tok/s on short prompts and 296 → 327 tok/s on 1.2K prompts; 16 agents 62 → 72 tok/s. Not built here: HIP, SYCL, MSVC. Design notes: `DESIGN-mmq-direct.md`.

## Finding 5: overlapping admissions does not raise throughput on Ampere, but cuts TTFT

We prototyped an overlapped admission: the prompt read runs on a worker thread while `pump` keeps the groups going. During a read, the groups decode at ~180-206 rows/s instead of ~620, and the read itself takes 10-20% longer. Total throughput stayed at 413/294 tok/s against 417/297. The median TTFT, however, fell from 2.0 s to 0.57 s for short prompts and from 4.3 s to 1.2 s for 1.2K prompts. `--ple-io ram` gave another +1-3% (a 1.2K read takes 821 ms instead of ~1.1 s). Per-window profiling (`STRATA_VERIFY_PROFILE`) shows ~85% GPU work and ~1 ms host overhead per group step.

## Multi-turn / agent workload

This profile is where Strata shines. 16 clients x 5 turns, with context growing from 11K to 28K tokens. With `--max-context 262144 --kv-resident 32768 --conversation-cache-mib 24576 --prompt-cache 24 --prompt-cache-every 8192`, the median follow-up TTFT is ~4.5-5 s. vLLM 0.31 on the same box, with prefix caching plus a 20 GB CPU KV tier, needs ~32 s at similar total throughput.

Happy to open PRs for the patches: separately, together, or rebased onto a branch of your choice. Thanks again!
