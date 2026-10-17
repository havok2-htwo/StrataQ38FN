# Conversation-state capacity of Strata production, per tier (2026-10-17)

How many tokens of conversation state Strata production can hold in each tier: GPU slots, the RAM conversation cache
and the SSD tier. All numbers below are read-only measurements: GET on the live server, engine logs, source and `df`.
No services or GPUs were touched.

Abbreviations: E = `/media/raid0/StrataQ38FN/Strata/strata-unsloth-ud-iq4_xs.log` (the production engine log),
C = `/media/raid0/StrataQ38FN/Strata/strata-cmpv-1441g.log`, S = source tree `/media/raid0/StrataQ38FN/Strata-1441`
(the size code in `conversation_state.cpp`, `conversation_cache.hpp` and the tier block is the same in Strata-1441-ll).

## 0. What production actually runs (this differs from the task text)

- `curl -s http://127.0.0.1:8090/metrics` -> `engine`: `"kv": "int8", "kv_resident": 28672, "batch_slots": 16,
  "context": 262144, "conversation_cache_mib": 131072, "conversation_cache_slots": 448,
  "conversation_cache_min_free_mib": 16384, "conversation_cache_min_tokens": 1024`.
  The same values are in `linux/configs/T16-1441g-lanes4-prod262r-vision-cpu.json` (production.json points to it)
  and in the engine command line at E:1551160.
- **The RAM cache is 128 GiB with 448 slots, not 96 GiB with 320** as the task text and the memory note say.
  The 96 GiB figure below is given only for comparison.
- SSD tier env (same config): `STRATA_CONV_TIER_DIR=/media/raid0/strata-kvspill`, `STRATA_CONV_TIER_MIN_TOKENS=8192`,
  `STRATA_CONV_TIER_WRITE_GB_DAY=300`, `STRATA_CONV_TIER_WARM=1`. `STRATA_CONV_TIER_CAP_GB`, `_MAX_FILES`, `_TTL_H` and
  `_MIN_FREE_GB` are unset, so they use their defaults: 600 GB, 2000 files, 168 h and 64 GiB (S `src/program/generate.cpp:8896-8903`).
- The engine was restarted at 10:31:13 (E:1551160). The status call showed `uptime_s: 95`, so the cache was empty at
  query time (`parked: 0`). That restart was not made by this analysis.
- Memory: the cgroup limit is `MemoryMax=224G` (`linux/systemd/strata.service`). The engine sees it at E:1551161:
  `memory guard: limit 224.0 GiB (source: cgroup) ... floor 16384 MiB`.

## 1. Bytes per token, and the fixed part of every conversation

Geometry (S `include/strata/plan/plan.hpp:23-58`, `include/strata/kernels/qsa.hpp:111-112`):
- 48 layers. 12 of them are QSA attention layers (3, 7, ..., 47), with 2 KV heads of dimension 256 and an indexer
  key of 128 values per 4-token block.
- The other 36 layers are GDN, with state 128 x 48 v-heads x 128 and a conv history of 3 x 10240.
- The MTP draft layer is one more attention layer.

### Per-token K/V (grows with length)

| part | bytes/token | source |
|---|---:|---|
| int8 K+V per QSA layer: 2 x 2 x 256 x 1 B + fp16 scale per 64 values (32 B) | 1,056 | S generate.cpp:977 "int8's 1,056"; plan.hpp:91-101 |
| indexer pooled key, fp32: 128 x 4 B per 4 tokens | 128 | plan.hpp:33-38, conversation_snapshot.cpp (pooled part) |
| x 12 QSA layers | 14,208 | |
| MTP draft layer K/V (int8, no indexer) | 1,056 | conversation_state.cpp:262-288 (`qsa + (draft?1:0)` layers) |
| token ids (int32) | 4 | conversation_state.cpp:270 |
| **total, image with draft layer (derived)** | **15,268** | |
| **total, `draftless` image of a pipelined batch slot (derived)** | **14,212** | conversation_cache.hpp:198-202 (no draft-layer K/V) |
| measured, `capture=slot` images (k=2 class) | 14,282 | least-squares slope of `snapshot_bytes` on tokens, 25,790 park lines, 66-78,190 tokens |
| measured, `capture=async` images (k=2 class) | 15,258 | 4,200 park lines, 23-84,588 tokens |
| measured, lines without a capture tag (k=2 class) | 15,291 | 836 park lines, 148-108,925 tokens |

The fits use the 30,826 park lines of E + C whose k (see below) rounds to 2; together their slope is 14,525 B/token,
a mix of the two image types. The draftless rate fits the slot images, the full rate the async/untagged ones.
**Most parked images are draftless slot images**: 29,485 of the 39,336 park lines in E + C are `capture=slot`
(7,785 `async`, 2,067 untagged), and 95 of the 106 in today's C.

So a parked conversation costs **14.2-15.3 KB per token of K/V, which is 70.3K-75.5K tokens per GiB**
(2^30 / 15,268 = 70,326; 2^30 / 14,212 = 75,552). The capacity rows below give this range.

### Fixed state per conversation (does not depend on length)

- GDN recurrence + conv history: 36 x (128·48·128·4 + 3·10240·4) = **117,669,888 B**. Source:
  `plan.hpp:68-86`, whose comment says "36 GDN layers of fp32 state is **117.7 MB**" (plan.hpp:75). The same formula
  is in `conversation_state.cpp:107-120`.
- Plus the PLE history (9 x 10240 x 4 = 368,640 B) and the indexer tails/dead rows (12 x 2,052 B).
  **Total: 118,063,152 B (112.6 MiB) per copy of the running state.**
- An image stores the live state **plus the same state again in every turn/message checkpoint**
  (`SavedConversation::bytes()`, S `include/strata/core/conversation_cache.hpp:212-219`;
  `metadata_bytes` sums `c.gdn.size() + c.ple.size() ...`, conversation_state.cpp:97-103).
  `bytes()` counts the live state, every checkpoint and all K/V of the image, including K/V that was reused from an
  earlier entry (production runs `STRATA_SLOT_CHAIN_REUSE=1`, T16-1441g-lanes4-prod262r-vision-cpu.json).
  The fixed part is therefore 118 MB x k, where k = 1 + number of checkpoints.
- Measured number of state copies k = (snapshot_bytes - 15,268·tokens) / 118,063,152 over **all 39,336 park lines
  in E + C** (counted at the time of this revision; E is still growing). 1,843 of them have `reused_kv_bytes>0`; they
  are full images and are included. The reuse-free columns are given only for comparison
  (`capture=slot` lines always print `reused_kv_bytes=0`, generate.cpp:11622). For draftless images, 15,268 slightly
  understates k (by 1,056·tokens / 118 MB, i.e. 0.9 at 100K tokens).

| image length | images (all) | mean k | max k | fixed part (mean k) | images without reuse | mean k | max k |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 8K tokens | 32,269 | 1.93 | 6.88 | ~0.23 GB | 31,841 | 1.92 | 5.05 |
| 8K-32K | 4,080 | 3.26 | 11.74 | ~0.38 GB | 3,554 | 2.76 | 7.09 |
| 32K-64K | 1,414 | 5.44 | 12.88 | ~0.64 GB | 1,236 | 4.99 | 7.12 |
| 64K-128K | 872 | 7.85 | 13.35 | ~0.93 GB | 569 | 6.26 | 7.17 |
| > 128K (to 250K) | 701 | 9.32 | 14.14 | ~1.10 GB | 293 | 7.08 | 7.24 |

  **k reaches about 14** (E:841964: `parked 220676 tokens ... snapshot_bytes=5039059832`, reused 3.25 GB, k = 14.1).
  Only images without reuse stay at k ≤ about 7. Today's production engine in C logged C:5465
  `parked 60636 tokens ... snapshot_bytes=2236574064 reused_kv_bytes=602404992 capture=async`, k ≈ 11.
  The image at the intercept, 23 tokens, is 238,710,216 B (2 copies). At 100K tokens, E:314379:
  `parked 101071 tokens ... snapshot_bytes=2385197800`.
- **Images of 95K-107K tokens (150 lines; 59 with reuse) measure 1.99-3.05 GB, median 2.42 GB, mean 2.53 GB**
  (slot: 29 images, 2.03-2.36 GB; async: 95, 1.99-3.05 GB; untagged: 26, 2.30-2.91 GB). The largest is E:896225:
  `parked 101492 tokens ... snapshot_bytes=3049530696 reused_kv_bytes=1491781248 capture=async`.
- **Rule of thumb:** image ≈ (14.2 KB for slot images, 15.3 KB otherwise) x tokens + 118 MB x k, with k ≈ 2 for
  short images, about 8 on average at 64K-128K, and up to about 14 with chain reuse.
  A 100K-token conversation is 1.42-1.53 GB of K/V plus 0.24-1.65 GB of state, **2.0-3.05 GB, typically ≈ 2.4 GB**.
  At k = 2 the fixed state is more than half the image below about 15-17K tokens.

## 2. Tier (a): GPU slots (`--kv-resident 28672`, 16 slots)

- E:1551181: `KV streaming: 28672 of 262144 cells per QSA layer in VRAM, the K/V in 0.77 GiB of pinned RAM`.
  The 0.77 GiB is CUDA0's 3 QSA layers x 262,144 x 1,056 B = 0.773 GiB.
- Each slot session is sized for the full 262,144-token context (S generate.cpp:5086-5094: `session_bytes(g,
  o.max_context, ...)`). The authoritative K/V of every cell is in a pinned, device-mapped host pool
  (S `src/core/layer.cpp:798-806`, "KV streaming: the authoritative K/V of every cell, pinned").
  Only `kv_resident` cells per QSA layer are kept in VRAM, as a block cache. The minimum is 20,480
  (layer.cpp:666, generate.cpp:978).
- The VRAM cost of the slots: E:1551187 `16 slot sessions on CUDA0 (0.28 GiB each)` (the same on CUDA1-3), and
  E:1551191 `the slot sessions take 4.46 GiB of VRAM on CUDA0`. Each slot's GDN state (118 MB) is in VRAM, split
  over the 4 cards.
- The pinned RAM for this is already allocated. It is not part of the 128 GiB cache budget: 17 sessions (16 slots +
  the main one) x 13 attention layers (12 QSA + the MTP draft layer, which is streamed and pinned as well) x 262,144 x
  1,056 B = **61,178,118,144 B (61.2 GB, 57.0 GiB)**. Measured on the live engine (PID 3196418):
  `RssShmem: 59735764 kB` in `/proc/PID/status`, which is KiB, so 61,169,422,336 B; the service cgroup's
  `memory.stat` gives `shmem 61169422336`. That is the estimate to within 9 MB.
- Whether a request is served from VRAM, at this configuration (1441f-lanes4, `--kv-resident 28672`):
  - Production traffic, engine run 05:40-09:53 today (E:1549781-1551123): 119 `KV streaming` lines, all at 99.82 % or
    higher, median 100.00 %; the read counts grow from line to line, up to E:1551116
    `KV streaming: 100.00% of 160999041 block reads hit VRAM, 9.7 MiB read from RAM`.
  - Today's benchmark windows (C, same engine and flags): run 1 (cmpv, bench_concurrency, C:1-2134) 120 lines, median
    85.55 %; run 2 (cmpa, bench_agentic + bench_multiturn, C:2135-) 124 lines, median 77.25 %, e.g. C:7969, 8006, 8044
    `KV streaming: 78.05% / 78.47% / 78.34% of 13866-15408 block reads hit VRAM, 12.1-13.4 MiB read from RAM`.
  - The earlier `E:810468 99.90% of 3815823 block reads` is from the 2026-10-14 run of strata-g3-143v3 with
    `--kv-resident 32768`, not this configuration.

| | tokens |
|---|---:|
| K/V cells resident in VRAM, 16 slots x 28,672 | **458,752** (16 x 0.34 GiB of int8 QSA K/V) |
| what 16 slots can hold, with streaming from pinned RAM, 16 x 262,144 | **4,194,304** |
| 100K-token conversations | **16** (one per slot; every slot has room for 2.6 x 100K, but holds only one conversation) |

## 3. Tier (b): the RAM conversation cache (128 GiB / 448 slots / 16 GiB floor / min 1024 tokens)

- Budget: 131,072 MiB = 137,438,953,472 B. Three limits apply, whichever is reached first:
  1. the byte budget (`make_room`);
  2. 448 entries (each conversation takes one entry whatever its length, conversation_cache.hpp:272-277);
  3. the physical RAM floor. Before each park, MemAvailable / the cgroup must keep 16,384 MiB free, or the oldest
     entries are evicted (S generate.cpp:9110-9142, 11595-11617).
- Is 128 GiB reachable? Yes, it was reached on 2026-10-14 (the engine started at E:748142 with 131072/448 and
  kv-resident 32768). E:788196: `parked=440 bytes=137436306600 evictions=39`, which is the budget to within
  2.6 MB. Near the top the RAM floor started to bite as well: E:810562
  `skip parking (RAM floor after capture: 16384 MiB needed, 16378 MiB available)` at about 447 entries / 133.7 GB
  (E:810471). There were 1,421 "evicted ... to admit this snapshot" lines in E. At the 224 GiB cgroup limit, the
  budget and the floor bind at about the same point.
- In today's runs (the same engine and cache flags; the tier went to strata-kvspill-bench): the peak was
  C:7932 `parked=34 bytes=38597165296`, about 36 GiB. These benchmarks did not fill the cache.

| capacity at 128 GiB | value |
|---|---:|
| tokens per GiB (K/V only) | **70.3K-75.5K** (15,268-14,212 B/token) |
| upper bound, pure K/V, no fixed state | **9.0-9.7 M tokens** (137.44e9 / 15,268 - 137.44e9 / 14,212) |
| 100K-token conversations (measured 95K-107K images: 1.99-3.05 GB, median 2.42 GB) | **≈ 57** (45-69) -> **≈ 5.7 M tokens** (4.5-6.9 M) |
| 20K-token agent sessions (≈0.28-0.31 GB K/V + ~0.38 GB state at mean k 3.26) | ≈ 200 |
| short chats, 1-8K tokens (≈0.24-0.35 GB each, mean k 1.93) | the 448-entry limit binds (448 x 0.30 GB ≈ 134 GB, also about the budget): ≈ 448 conversations, ≈ 1-2 M tokens |
| the old 96 GiB / 320 figure, for comparison | 6.75-7.25 M tokens upper bound, ≈ 43 (34-52) x 100K |

## 4. Tier (c): the SSD tier (`/media/raid0/strata-kvspill`)

- What gets written: only conversations the RAM cache evicts to make room, and only those of at least 8,192 tokens
  (STRATA_CONV_TIER_MIN_TOKENS). Each is one file per stage (4 `.ses` files + `.meta`), byte for byte the RAM image
  (S generate.cpp:8861-8875). So bytes per token are the same as in §1. Examples from E:
  - E:474803: `spilled 58693 tokens (1433.5 MB, 5 files)`; the RAM image was E:453066 `snapshot_bytes=1433258880
    ... capture=slot` (draftless: k = 5.07 at 14,212 B/token, against a non-integer 4.55 at 15,268)
  - 208,485 tokens -> 4,035.2 MB (E:371780)
  - E:606972: a restore of 31,800 tokens read 690.9 MB in 136 ms.
- Limits, whichever is reached first:
  - **cap 600 GB** (default `STRATA_CONV_TIER_CAP_GB`, generate.cpp:8898, enforced at conversation_tier.cpp:411)
  - 2000 files
  - TTL 168 h
  - at least 64 GiB of disk left free
  - **300 GB written per day**
- Disk: `df -B1 /media/raid0/strata-kvspill` -> 1,102,841,810,944 B free (1.1 TB of 2.95 TB). Disk space is not the
  limit; the 600 GB cap is.
- Live now (`/metrics` -> `conversation_cache.tier`): `disk_bytes: 0, files: 0, foreign: 42` (52.1 GB from `du -sb`,
  files of earlier engine identities, kept until their TTL runs out), `written_today: 46474647820`,
  `budget_left: 253525352180`.
- Observed peak: E:696775 `files=347 ... disk_bytes=340102993328` (340 GB). On 10-14 the daily write budget ran out:
  E:810470 `written_today=299974236018 ... drops_budget=35` (first at E:624880), and up to `drops_budget=233` in E.

| capacity | value |
|---|---:|
| at the 600 GB cap, pure K/V | **≈ 39-42 M tokens** (600e9 / 15,268 - 600e9 / 14,212) |
| 100K-token conversations (1.99-3.05 GB, median 2.42 GB) | **≈ 248** (197-301; the 2000-file limit does not bind) |
| fill rate, 300 GB/day | ≈ 124 x 100K conversations/day (98-150) ≈ **12.4 M tokens/day**; pure-K/V upper bound 19.6-21.1 M tokens/day |
| if the cap were raised to the free disk (1.10 TB - 64 GiB = 1.034e12 B) | ≈ 68-73 M tokens pure K/V / ≈ 427 (339-519) x 100K |

## 5. vLLM in today's runs: GPU pool only, bf16

| run | args | log line | KV pool | tokens | max concurrency |
|---|---|---|---|---:|---|
| cmpv no-MTP | 32K, util 0.94, TP4+EP | `cmpv-nomtp-server.log:137,140` | 2.26 GiB per GPU | **150,989** | 4.61x at 32,768 |
| cmpa no-MTP | 131K, util 0.94 | `cmpa-nomtp-server.log:137,142` | 2.44 GiB per GPU | **188,092** | 1.44x at 131,072 |
| cmpm MTP | 64K, util 0.93, 8 seqs, MTP | `cmpm-mtp64k-server.log:187,191` | 1.23 GiB per GPU | **76,127** | 1.16x at 65,536 |

- `kv_cache_dtype=auto` (bf16) in all three logs. There was no CPU or disk KV tier: the only tier attempt, on 15.10
  (`cbvllm-kvram-server.log`, SimpleCPUOffloadConnector, 80 GiB), did not start (`cbvllm-1441-2026-10-15.md:22-24`).
- Peak `GPU KV cache usage`: 54.7 % (cmpv), **97.6 % (cmpa)**, **100.0 % (cmpm)**. The final prefix-cache hit rate was
  39.1 % in the cmpa agentic + multiturn run and 11.8 % with MTP: the pool was full and did not keep the earlier
  turns. For comparison, Strata's multiturn in the same window (`cmpa-strata-mt.txt`) shows
  `cached 11430` / `15868` tokens on turns 2 and 3.
- Efficiency, per-token K/V against per-token K/V: vLLM keeps bf16 K/V for the 12 full-attention layers, 2 KV heads
  of dimension 256 (`config.json` of VnimanieAI/Qwen3.8-Flash-Next-W4A16: `num_key_value_heads 2, head_dim 256`,
  12 `full_attention` layer types). At TP=4 the 2 heads are replicated to 1 per GPU, so 12 x 2 x 256 x 2 B =
  12,288 B per GPU, **49,152 B/token over the 4 GPUs, ≈ 21.8K tokens/GiB** (derived, before Mamba state). Strata's
  RAM images hold 14,212-15,268 B/token, 3.2-3.5x denser, mostly from int8 + scales instead of bf16 and no head
  replication. The measured pool rate, 4 x 2.26 GiB for 150,989 tokens = 64,287 B/token (≈ 16.7K tokens/GiB), also
  pays for vLLM's per-sequence Mamba pages (attention block size set to 784 tokens to match the Mamba page size,
  `cmpv-nomtp-server.log:115`), so it is not comparable with Strata's K/V-only rate.
- Whole conversation at 100K tokens: Strata's RAM image is 2.0-3.05 GB with all GDN state and checkpoints; vLLM needs
  4.9 GB of VRAM K/V (100K x 49,152 B) plus its Mamba state. The two sides are also different memories: Strata's
  figures are parked images in host RAM, vLLM's is its live VRAM pool.

## 6. Summary

| tier | Strata production | 100K-token conversations | vLLM today |
|---|---|---:|---|
| (a) GPU slots | 458,752 tokens in VRAM (16 x 28,672); 4.19 M tokens held in 16 slots (K/V streamed from 61.2 GB / 57.0 GiB of already-pinned RAM) | 16 (one per slot) | 76 K-188 K tokens in total (VRAM pool); at 100K: ≈ 1.9 in the 188,092-token pool |
| (b) RAM cache 128 GiB | 70.3K-75.5K tokens/GiB; ≤ 9.0-9.7 M tokens; 448 entries | ≈ 57 (45-69; ≈ 5.7 M tokens) | none |
| (c) SSD 600 GB cap | ≤ 39-42 M tokens; 300 GB/day write | ≈ 248 (197-301; + ≈ 124 per day) | none |

- **Per-token cost:** 14.2 KB/token for draftless slot images (most parks), 15.3 KB/token with the MTP draft layer
  (int8 QSA K/V + fp32 indexer key [+ draft K/V]).
- **Fixed cost:** 118 MB of GDN/PLE state per copy, k copies per image (live + checkpoints): about 2 for short
  images, about 8 on average at 64K-128K, up to about 14 with chain reuse (≈ 1.65 GB). For images without reuse k
  stays at about 7 or less. The fixed part dominates below about 15K tokens. Fewer stored checkpoints, or storing
  checkpoint GDN states as deltas, would be the lever, for short conversations and for long chain-reused ones.
- **Total, Strata's RAM tier against vLLM's VRAM pool** (different memories; vLLM had no CPU or disk tier today):
  realistically about 5.7 M tokens of 100K-token conversations in RAM, **about 30x** vLLM's largest GPU pool today
  (188,092 tokens); the pure-K/V upper bound of 9.0-9.7 M tokens is about 48-51x. At 100K tokens that is ≈ 57
  conversations against ≈ 1.9 in vLLM's pool (its logged 1.44x is at 131,072 tokens).
