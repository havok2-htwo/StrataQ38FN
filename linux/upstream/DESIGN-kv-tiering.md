# KV tiering plan: VRAM slots -> RAM conversation cache -> SSD emergency tier

Read-only code audit of Strata-143 (`g3-143`, 1797c0fb) plus the kvaudit logs. Line numbers refer to that commit.

## 0. What the code does today (findings)

### 0.1 What a park is and why it costs ~65-150 ms
- `park_current_body` (`src/program/generate.cpp:8107-8264`) snapshots the **main** sessions (`ss` plus `stages[k]->ss`), not a batch slot. In batch mode the main session holds the prompt that the **previous** admission read; `copy_to_slot` copied that prompt into its slot (`:13254`). The main session keeps it until the next request, and `:10955` parks it there whenever the new prompt does not continue `live` (`!from_live || incoming || slot_source >= 0`).
- The capture is fully synchronous and serial:
  - `conversation_snapshot_save` (`src/core/conversation_state.cpp:280`) calls `sync()`, then `conversation_checkpoint_save` (`:155`) and `conversation_kv_save` (`src/core/conversation_snapshot.cpp:147`).
  - Every part goes through `transfer` → `copy_nonblocking`, which does `cudaMemcpyAsync` plus `cudaStreamSynchronize` per call (`src/core/conversation_copy.hpp:33-39`).
  - The destination is **pageable, freshly allocated and zero-filled**: `ConversationBuffer::resize` → `segment.resize(count, value)` (`include/strata/core/conversation_buffer.hpp:62-78`), and the `std::vector` checkpoint fields are resized in `conversation_checkpoint_save`. So every byte page-faults and then goes through the driver's pageable staging.
  - Stages run one after another, with a `cudaDeviceSynchronize` per stage (`generate.cpp:8227-8238`).
- The logs fit this cost model:
  - `results/engine-ab-kvi-T16-143-64khoc-*.run.log`: a 148-token park produces a 121 MB snapshot in 63.5 ms. That is almost entirely the fixed ~118 MB running state (GDN, PLE and indexer tails) → **~1.9 GB/s effective**.
  - `kvaudit/engine-mt2-64khoc.log`: an 11K-token park is 287 MB in 147 ms (1.95 GB/s). A restore of the same size takes 35.6 ms (8 GB/s H2D).
  - **The cost is fixed per park, not proportional to the prompt.** One-shot prompts of 100 or 1.2K tokens pay ~65 ms each.
- **Why it is on the all-slot critical path.** In the pipelined mode, every line other than BSTOP (and BYIELD with NODRAIN) first runs `pipe_drain()` (`generate.cpp:10276-10282`). That covers BGEN, so all four groups stop. The request path then runs synchronously:
  1. Parking (`:10955`).
  2. Restoring from RAM (`:11013-11071`).
  3. The prompt read.
  4. `copy_to_slot` and `BADM` (`:13340`).

  The groups restart only after that. So park time, restore time and the read are all stall time for all 16 slots.
  - Multi-turn 24×4: 96 parks = **8.8 s**, plus 71 restores × 35.6 ms = **2.5 s**, out of 165 s wall.
  - One-shot: 32/64 parks at ~67/70 ms each. That explains the −6.7 % / −5.6 % (#67).
- RAM admission:
  - `ConversationCache::make_room` (`include/strata/core/conversation_cache.hpp:266`) handles the budget and the slot count.
  - The physical gate is `sample_host_memory` + `conversation_memory_admit` with `--conversation-cache-min-free-mib` (default 2560), evicting oldest-first (`generate.cpp:8199-8229`).
  - "Available" is the minimum of MemAvailable, every cgroup's `max`/`high` minus `memory.current`, and `--memory-limit-mib`. Only **clean inactive_file** is credited (`include/strata/core/conversation_memory.hpp`).
  - The 86 refused parks in the agent config (#65) came from 52.6 GiB of pinned streaming K/V (shmem/locked, never credited) plus file cache inside the 90 GiB scope. The default config (resident K/V) refused none.
  - Implication for an SSD tier: buffered writes would fill the cgroup with dirty, uncredited page cache. Use O_DIRECT, which `conversation_file.cpp` already does.

### 0.2 Batch-slot lifecycle (pipelined, 4 groups × 4 slots)
- `BSlot` (`generate.cpp:9600-9625`): `ids` = prompt plus every fed token. `cached` means idle with `ids` still held. `checks` = the turn checkpoint (with `STRATA_SLOT_CKPT_MOVE`, moved out of the main chain at admission, `:13283-13300`).
- BDONE in `pump` (`:10155-10165`) sets `sl.cached = true` (#857).
- When a group starts, `G.S` = its last active slot + 1. Every inactive slot below that becomes a **pad row** (`tok 0, pos 0`), and `if (!sl.active) sl.cached = false;` (`:10214`). So **an idle slot below a still-active slot of its group is wiped within one window**. Only top-of-group slots survive.
- `copy_from_slot` (`:9737`) is the D2D slot → main copy (0.9 ms for 20-25K tokens in the mt log, against 36 ms for a RAM restore). It is used only when `sl.cached` holds; it happened twice in the mt run.
- `slot_source` (`:10883-10912`) also considers active slots' checkpoints (SLOT_CKPT_MOVE). That is exact because windows write cells ≥ p only.
- Slots have **no draft-layer K/V** in pipelined mode (`batch_mtp` is forced off with `--batch-groups`, `:4553-4570`). `copy_from_slot` leaves the main drafter's K/V stale below `resume`. That is the documented "drafts may differ, the target's tokens never".

### 0.3 Server picker
`serve/server.py:1584-1615` `pick_slot`: with pipelined groups (`slot_gs`), the held prefix is **ignored**. It picks "fills a hole below a busy slot" first, then the spreading order. The comment at `:1593` ("the engine keeps no pipelined slot as a conversation cache") is stale since #857, but in effect true because of the pad-row wipe. `slot_held` is set at BDONE (`:1500`, `:1565`) and cleared at admission (`:1435`).

### 0.4 Session files (`.ses`, SAVE/RESTORE)
- `include/strata/core/conversation_file.hpp`, `src/core/conversation_file.cpp`. Format v1:
  - 64-byte header with magic `STRSESS\1`, model and config fingerprints and the payload size.
  - Payload: ids, images, live state, the deepest checkpoint, every K/V layer and the draft.
  - A streaming payload hash and a 16-byte trailer.
- I/O is done in 16 MiB blocks with O_DIRECT when the filesystem accepts it, through a temp file, fsync, rename and dir-fsync.
- RAM preflight (`SessionReadLimits::admit`), free-disk preflight (`--session-min-free-mib`), and identity, geometry and layer-range checks before any allocation.
- **Refuses layer-split state** (`write_impl` → `has_stage_parts`, `conversation_file.cpp:1032`; SAVE refuses `--layer-split`, `generate.cpp:10403`). However, a parked split image is already `stage0 SavedConversation` + `stage_images[k]`, each a plain single-carve `SavedConversation` with its own `layer_lo/hi` and no `stage_parts`. **Each part can be written as an ordinary v1 file** (one per stage) with no format change.
- The documented speed is 1.2 GB in 709 ms save / 898 ms restore, single-threaded.

---

## Stage 0: config-only quick win (no code)

**Change.** Add `--conversation-cache-min-tokens 2048` to `T16-143-64khoc` (new arm, e.g. `T16-143-64khocm`). `ConversationCache::wants()` (`conversation_cache.hpp:209`) then skips parking for every one-shot bench prompt (100 / ~670 tokens), and the ~65 ms fixed cost goes away. The mt bench (≥ 11K) is unaffected.
- **Exactness:** parking is never on the token path. Replies stay identical (ab_quick checks the hashes).
- **Expected:**
  - One-shot: back to the `-64kho` level (575 / 454, about +6-7 %) for prompts under the threshold.
  - Multi-turn: unchanged.
  - Trade-off: chats under 2K tokens are re-read (≤ ~0.5 s) instead of restored.
- **Cost:** none. Slightly less RAM churn.
- **Test:** `REPS=3 ab_quick.sh kvm T16-143-64kho T16-143-64khoc T16-143-64khocm` and one `kvaudit_mt.sh` run.
- **Effort:** 0.5 h (+ ~45 min GPU).

## Stage 1: asynchronous parking (removes the per-park cost for every prompt size)

### 1a `STRATA_PARK_THREADS=1`: parallel stage capture (synchronous, small)
- **Where:** `park_current_body` stage loop, `generate.cpp:8227-8238`.
- **Change:** run the 4 stages' `conversation_snapshot_save` at once on the existing `StageThreads` (the `STRATA_CKPT_THREADS` pattern, `:1629`, `:8384`). The threads sit on `--aux-cpus` and each one's `copy_nonblocking` stream is per thread and device. Also take the live checkpoint's vectors from `CheckpointPool` (warm pages; the comment at `:8020` measured 17 vs 40-60 ms for fresh pages).
- **Exactness:** the same functions copy the same bytes into the same image; only the order between devices changes. `merge_back` and the try/catch stay as they are.
- **Expected:** park 147 → ~40-50 ms (11K), 63 → ~20 ms (short). One-shot about −2 % left; multi-turn about +3 %.
- **Cost:** 0 VRAM, 0 extra RAM.
- **Tests:** batch_test16/32, T=0.7 seed 1234, interleave, overlap (token + RESUME), a `STRATA_SNAPSHOT_VERIFY=1` mt run, and a unit test in `conversation_snapshot_test.cpp` (threaded vs serial capture `==`).
- **Effort:** 2-3 h.

### 1b `STRATA_PARK_ASYNC=1`: eager capture after the admission, off the critical path
- **Idea:** the main session is not written by anything between `BADM` (`:13340`) and the next request's prompt read. Pipelined windows only use slot sessions, the main drafter is idle (`batch_mtp` off), and expert-tier swaps are not session state. So capture it **right after BADM**, while the groups decode, and **decide at the next request** (where `:10955` decides today) whether to `put` it or drop it.
- **Mechanics:**
  1. After `BADM` (and after a solo `DONE`), if `conversations.enabled() && wants(live.size())`, run the RAM checks from `park_current_body` (min-tokens, `drop_superseded`, `make_room`, the physical gate) **on the main thread**. Then start a `ParkJob`: one persistent worker per stage device.
  2. Each worker does D2H of its session's running state and K/V `[0, ceil(len/page)·page)` in 16 MiB pieces into a **pinned 2 × 16 MiB ring per device** (cudaHostAlloc; true async). An event marks each piece, and the worker copies it into `ConversationBuffer` segments. The "reuse" path stays: `take_reuse()` is handed to the job, so only pages past `unchanged_tokens` move.
  3. Host-side data (`live`, `live_imgs`, a copy of `checks` / `cs.parts`) is copied by the worker. The main thread does not mutate `checks` until it joins.
  4. **Join points** (before anything may write the main sessions or `checks`):
     - the start of the next GEN/BGEN request, before `best()`/`take()` at `:10917`;
     - `SAVE`/`RESTORE`, `VRAM`, `kvg_trim`/`kvg_ensure`;
     - the #879 flush (`:10826`), where the image is discarded;
     - `STRATA_PIPE_OVERLAP` reads;
     - QUIT.

     The join is normally free: a job takes about 5-15 ms per card, and the next admission is far later.
  5. At `:10955`: if parking is due, `put(image, held)`. Otherwise (`from_live`, `pin_sibling`, cvec mismatch) the image is dropped and its segments are freed.
- **Exactness:** the image is the same bytes as `park_current` would capture at the next request, because the source is provably unchanged in between (assert: no stream on the main sessions has work queued; keep a debug `STRATA_PARK_ASYNC_VERIFY=1` that does a sync capture as well and compares with `ConversationBuffer::operator==`). Restores use the unchanged `conversation_snapshot_restore`.
- **Expected:**
  - One-shot: −6.7/−5.6 % → within noise (≤ −1 %), for any prompt size. Residual costs are PCIe D2H contention with the hand-off and the host copy threads.
  - Multi-turn: removes 8.8 s of 165 s of all-slot stall → **about +5 % tok/s** (≈ 185). Follow-up TTFT p50 −0.2..−0.6 s, because admissions queue behind each other.
- **Cost:** 128 MiB pinned (4 × 32 MiB). RAM is the same as today, plus a transient image when it is discarded. VRAM 0. Aux CPU time ~20-40 ms per park on worker threads.
- **Tests:**
  - All exactness tools (byte-identical dumps).
  - `STRATA_PARK_ASYNC_VERIFY=1` across kvaudit_mt (every image equals the sync capture).
  - `STRATA_NAN_INJECT` mid-job (the image is discarded, no restore of a tainted image).
  - Cancel/BSTOP during a job, SAVE during a job.
  - ab_quick REPS=3 against `-64kho` and `-64khoc`; kvaudit_mt.
- **Effort:** 6-8 h.

## Stage 2: keep conversations alive across turns in batch mode

### 2a `STRATA_SLOT_PARK=1`: park the slot's conversation (reply included) before anything overwrites it
- **Idea:** at BDONE the slot holds the most useful state:
  - `ids` = prompt plus reply;
  - the running state at the end of the reply;
  - K/V `[0, len)`;
  - `sl.checks` = the turn checkpoint at the last `<|im_start|>` (`:11955-11979`).

  Today it is lost to the first pad row (`:10214`) or to the next admission into that slot. Capture it into the RAM tier with the 1b workers, reading from `bslot_ss[k][b]`.
- **Ordering, without stalling other groups:**
  - (i) In pump's group start (`:10190-10215`), skip a group (`continue`, like `ov_layout`'s wait) while a slot that would become a pad row (`b < gi*GS+G.S`, `!active`) has a capture whose events are not all complete (`cudaEventQuery`). The other groups keep the stages busy, so only that group loses ~one D2H time (5-20 ms).
  - (ii) `copy_to_slot` into that slot (an admission, or BYIELD `:11867`) joins the capture first.
  - (iii) The #879 flush discards it.
- **Image content:**
  - `live` = `sl.ids` with the slot's running state, and `checkpoints` = `sl.checks` (already host-side, with `stage_parts` → split as at `:8133`).
  - K/V comes from the slot.
  - **Draft K/V:** none (the slot has none). Add `bool draft` to `SavedConversation` (or check `kv.size()`); restore the last stage with `draft=nullptr` for such images. The main drafter below `resume` is then stale exactly as after `copy_from_slot` today: drafts may differ, tokens never.
- **Supersede:** when 2a is on, 1b skips the main-session capture for admissions that continued in a slot (`cont`), because the slot image is a superset of it. Move the turn checkpoint into the image when the slot is wiped (not copied), so RAM is not paid twice. Rely on `drop_superseded` for older copies.
- **Exactness:** the image bytes equal the slot's state at BDONE, which is exactly what the tested `copy_from_slot` path would use (the interleave test covers "a turn checkpoint taken from a slot"). The capture is complete before any write to the slot, by (i)-(iii). Restore and rewind use the unchanged functions.
- **Expected:**
  - Multi-turn: a client that re-renders the reply identically resumes **after the reply** (live match). Thinking-stripping clients resume at the turn checkpoint instead of reading from 0 when the slot was wiped. In kvaudit 400 reply tokens × 72 follow-ups ≈ 29K fewer prompt tokens (~5 % of all prompt reading) → **+3-5 % tok/s**, follow-up TTFT −0.2..−0.4 s.
  - One-shot: neutral (≤ 0.3 % from deferred group starts).
- **Cost:** RAM per image is the reply's K/V (~6 MB) plus the turn checkpoint (118 MB; moved, not duplicated). Pinned staging is shared with 1b. VRAM 0.
- **Tests:**
  - `batch_interleave_test` + `batch_groups_cache_test` with idle slots below busy ones (force wipes).
  - A restore of a slot image must give tokens identical to an uninterrupted conversation (greedy and T=0.7).
  - `STRATA_SNAPSHOT_VERIFY`.
  - kvaudit_mt counts: restores/turn and `cached` per turn should rise by ~400.
- **Effort:** 10-14 h.

### 2b `STRATA_SLOT_AFFINITY=1` (server, `serve/server.py:1584-1615`)
- **Pipelined picker order:**
  1. A free slot holding ≥ `SLOT_PREFIX_MIN` of the prompt that is a hole or the top of its group (it adds no pad row over another idle cached slot): D2D copy ~1 ms instead of a 36-100 ms RAM restore.
  2. `fills_hole`.
  3. Slots holding nothing first, then LRU (so the conversations that are held survive longest).
- **Server mirror of the wipe rule:** `held[b] = []` when a busy slot above `b` in the same group exists at BDONE, or when an admission lands above an idle `b`. Alternatively the engine prints an opt-in `BWIPE <slot>`.
- **Exactness:** the engine re-checks `sl.cached` and the prefix (`:10893-10912`), so the server only picks.
- **Expected:** −35..−100 ms per hit at admission (about 1-2 % mt). Bigger without pipelined groups.
- **Cost:** none.
- **Tests:** `serve/test_slot_alloc.py`, `test_slots.py` (new cases for the pipelined rule), kvaudit_mt `gave back` count.
- **Effort:** 3-4 h.

### 2c (optional) `STRATA_RESTORE_EARLY=1`: restore before the drain
- **Change:** at BGEN, before `pipe_drain()` (`:10276`), match `conversations.best()` and run the H2D restore of the image into the main sessions on the stage workers while the groups still decode. The main sessions are not used by pipelined windows. Join after the drain.
- **Exactness:** the same restore function, the same bytes. Validation still runs before any write.
- **Expected:** removes 71 × 36 ms ≈ 2.5 s of 165 s stall → about +1.5 %. This is also the hook for SSD prefetch (3d).
- **Effort:** 4-6 h.

## Stage 3: SSD emergency tier (user design)

Default: off; with the env unset the build is byte-identical.

| Env | Default | Meaning |
|---|---|---|
| `STRATA_CONV_TIER_DIR` | unset = off | e.g. `/media/raid0/strata-convtier`, created with mode 0700 |
| `STRATA_CONV_TIER_MIN_TOKENS` | 16384 | only conversations at least this long spill |
| `STRATA_CONV_TIER_WRITE_GB_DAY` | 200 | rolling 24 h write budget |
| `STRATA_CONV_TIER_CAP_GB` | 600 | on-disk cap |
| `STRATA_CONV_TIER_MAX_FILES` | 2000 | file-count cap |
| `STRATA_CONV_TIER_TTL_H` | 168 | expiry by last use |
| `STRATA_CONV_TIER_INFLIGHT_MIB` | 4096 | RAM a pending spill may hold |

**RAM tier sizing stays config-driven:**
- The launcher config computes `--conversation-cache-mib` from a RAM budget: 48 GiB on 128 GB now, ~110 GiB on 192 GB.
- Raise `--conversation-cache-slots` to about budget / 0.4 GB (~120). At 24K-token images (0.62 GB), 48 slots bind before 48 GiB does.

### 3a Spill on RAM eviction only; write once
- **Hook:** give `ConversationCache` an optional eviction sink. `evict_oldest()` (`conversation_cache.hpp:283`; used by `make_room` `:266` and the physical gate `generate.cpp:8216`) moves the victim to the sink instead of freeing it. `drop_superseded` and `clear()` (#879) never spill.
- **Spill conditions:**
  - the tier is on;
  - `live.ids.size() ≥ MIN_TOKENS`;
  - the 24 h budget has room for the image's bytes;
  - the cap allows it after a TTL/LRU sweep;
  - the in-flight queue is under `INFLIGHT`.

  Otherwise it is freed as today, and a counter records why.
- **Writer thread:** writes `stage0` and each `stage_images[k]` as **plain v1 files** with `session_file_write` (O_DIRECT, temp + fsync + rename, free-space preflight). Files: `DIR/<key>.s0.ses` … `.s3.ses` plus a small `<key>.meta` (ids, checkpoint id lengths, image keys, cvec, bytes, created/used, base key). The 4 stages are written in parallel. RAM is freed after the rename.
- **RAM accounting:** pending spill bytes count as "already given back" in the physical gate (`available + inflight`), bounded by `INFLIGHT`, so the gate does not evict extra entries. O_DIRECT means no page-cache charge to the cgroup (which is what refused parks in #65).
- **Write-once:** `key` = hash(model fp, config fp, layer ranges, live ids, checkpoint ids, image keys, cvec), i.e. content addressing. An image whose key already exists on disk is never written again. That covers an image that came back from SSD and is evicted again unchanged: it costs 0 bytes.
- **Index:**
  - In RAM, with the ids per entry: 4 B/token, e.g. 2000 × 32K = 256 MB, or keep only rolling page hashes and read the ids from `.meta` when a candidate matches.
  - `best()` is extended: RAM entry, slot, live and checkpoint first; then the tier, only when its match beats them by ≥ 2048 tokens.
- **Restore:**
  - `session_file_read` per stage, with `conversation_session_read_limits` per stage (layer range, geometry, sizes) and the RAM preflight, running 4 threads in parallel.
  - Assemble a `SavedConversation`, then run the normal validate/restore path at `:10920-11071`.
  - On any refusal (hash, identity, missing file), fall back to reading the prompt. Never a partial restore.
- **Exactness:**
  - File bytes = the RAM image bytes (the v1 writer/reader round trip; `conversation_file_test.cpp` already checks `==`). Payload hash plus identity plus geometry and layer-range checks refuse anything else.
  - The restored device state is byte-identical to a RAM-tier restore, which is byte-identical to never-evicted.
  - #879: a flush drops the tier entries written since the last clean point, or all of them (simplest).
- **Expected:**
  - 0 in today's 24×4 bench (25 parked = 15.8 GB, RAM suffices).
  - In overflow (e.g. 150 agents × 32K = 110 GB), every SSD hit replaces a 32K re-read (≈ 6-25 s of all-slot stall, depending on prefill rate) with ~0.15-0.3 s: ~0.73 GB per stage set, 4 × O_DIRECT streams at 6-9 GB/s, plus the H2D restore.
- **SSD writes:** a full image is 15.2 KB/token + 236 MB fixed (live state and checkpoint) → 32K ≈ 0.73 GB. At 200 GB/day that is about 270 spills/day.
- **Effort:** 14-18 h.

### 3b Append/delta images: "no full rewrites" for a conversation that grows
- **Idea:** the RAM tier already tracks a retained prefix that has not changed (`ConversationKvReuse::unchanged_tokens`, `retain`/`limit_reuse`). Carry `origin_key` and `unchanged_tokens` through SSD restore → main → slot (2a) → image.
- **Spill:** if the origin's files still exist and `unchanged ≥ 50 %` of the cells, write a **delta file**: a new magic `STRSDLT\1` with base key, base payload hash and `page_from = floor(unchanged/page)`, then the K/V pages `[page_from, cells)` per layer, the new live state and the checkpoint.
- **Restore:** read the base pages `< page_from` and the delta, then verify both hashes.
- **Chain cap:** depth ≤ 3. After that a full write, charged to the budget. TTL/cap delete whole chains.
- **Exactness:** K/V cells below the first rewrite are byte-identical (the same invariant `limit_reuse` relies on). A unit test checks that delta+base reconstruct `==` the full image.
- **Writes per grown turn:** ≈ 236 MB + new tokens × 15.2 KB (3K-token turn ≈ 0.28 GB instead of 0.73 GB).
- **Effort:** 8-10 h.

### 3c Budget, TTL, cap, metrics, dashboard
- **Budget:** a rolling 24 h byte window kept in `DIR/.budget`: an append log of (unix time, bytes), compacted daily. It survives restarts.
- **Sweeper:** on the writer thread. Expires by `used`, then LRU to the GB/file cap.
- **Engine log lines** (stderr, like `conversation cache:` today):
  - `conversation tier: spilled N tokens (B bytes, delta|full) in X ms; files=F disk_bytes=D written_24h=W budget=Bud`
  - `conversation tier: skip spill (budget|min_tokens|cap|busy)`
  - `conversation tier: restored N tokens from disk in X ms (read R MB)`
  - `conversation tier: expired K`
- **Server:** extend `ConvCacheLog` (`serve/server.py:504-558`) with a `TIER` regex, feeding `conversation_cache_view` → `/status` `conversation_cache.tier` and Prometheus `strata:conv_tier_{written_bytes_24h, budget_bytes, disk_bytes, files, spills_total, restores_total, skips_total{reason}, restore_ms}` for the dashboard.
- **Effort:** 3-4 h.

### 3d (optional) Warm restart and prefetch
- On start, scan `DIR`: headers and `.meta` with matching identity → rebuild the index. Delete files from other identities only by TTL.
- Optional `PREFETCH <ids>` control line (the server sends it while a request waits for a slot; excluded from `pipe_drain` like BSTOP). It reads the matching set into RAM in the background, so the admission restores from RAM. Builds on 2c.
- **Effort:** 4-6 h.

### Stage 3 tests
- **CPU unit tests:**
  - budget window, TTL, cap and the eviction sink (`conversation_cache_test.cpp` style);
  - per-stage v1 round trip;
  - delta reconstruct `==`;
  - fault injection: `SessionWriteOptions::fault` for ENOSPC and rename, truncated and corrupt files → refusal and fallback.
- **GPU:**
  - kvaudit_mt with a tiny RAM tier (`--conversation-cache-mib 4096 --conversation-cache-slots 4`) plus the tier at min 8K, compared with a huge RAM tier: replies must be identical (add reply hashes to `bench_multiturn.py` output), with `STRATA_SNAPSHOT_VERIFY=1`.
  - Kill the engine scope during a spill → no partial file is visible.
  - Restart → warm restore (3d).
  - Budget exhaustion → spill skipped, metric shown.

---

## Order, totals, expected end state

| Stage | Switch | Effort (agent-h) | One-shot | Multi-turn 24×4 | RAM / VRAM / SSD |
|---|---|---|---|---|---|
| 0 | `--conversation-cache-min-tokens 2048` | 0.5 | +6-7 % (bench prompts) | ±0 | less churn / 0 / 0 |
| 1a | `STRATA_PARK_THREADS=1` | 2-3 | +4 % | +3 % | 0 / 0 / 0 |
| 1b | `STRATA_PARK_ASYNC=1` | 6-8 | to within noise, any size | +5 % (8.8 s stall gone) | +128 MiB pinned / 0 / 0 |
| 2a | `STRATA_SLOT_PARK=1` | 10-14 | ≤ −0.3 % | +3-5 %, TTFT −0.3 s | +reply K/V per image / 0 / 0 |
| 2b | `STRATA_SLOT_AFFINITY=1` (server) | 3-4 | 0 | +1-2 % | 0 |
| 2c | `STRATA_RESTORE_EARLY=1` | 4-6 | 0 | +1.5 % | 0 |
| 3a-c | `STRATA_CONV_TIER_*` | 25-32 | 0 | 0 here; in overflow it replaces 6-25 s re-reads with ~0.3 s restores | ≤ INFLIGHT / 0 / ≤ budget (200 GB/day) |
| 3d | warm restart, PREFETCH | 4-6 | 0 | admission −0.1-0.3 s per SSD hit | 0 |

Recommended order:
1. Stage 0 now; it is a measurement only.
2. 1a, then 1b. Together they make the cache-on config the default without the one-shot penalty, and make Stage 0's threshold optional.
3. 2a, then 2b.
4. 3a, then 3c, then 3b, then 3d.
5. 2c whenever the SSD prefetch needs it.

Each step is its own commit with an env switch, its own exactness run and an ab_quick/kvaudit_mt entry.

## Risks and open points
- PCIe contention: D2H on the copy engines during decode with `--ple-io ram` and hand-off traffic. Measure the per-window `rows/s` engine stat line with 1b on and off.
- Pageable destination copies stay CPU-bound (page faults). If the threads are not fast enough, keep a bounded free-list of 16 MiB segments from evicted images. It must be counted in the budget, because the physical gate assumes eviction returns memory to the kernel at once (`conversation_cache.hpp:276-282`).
- 2a group deferral: if measured > 0.5 %, verify what a pad row writes (expected: cell pos 0, pooled row 0, running state). If confirmed with a hash test of cells `[1, len)` across pad windows, a D2D save of only those bytes (~30 MB/card, VRAM shadow) would remove the deferral. Needs proof first, so not part of the base design.
- The slot images have no draft K/V: solo-path draft acceptance right after such a restore is lower until the drafter catches up. Tokens are unaffected.
- `bench_multiturn.py` cannot currently detect token mismatches. Add reply hashes before Stage 2/3 exactness claims rest on it.

---

# Critique and corrected order

**KV tiering plan: adversarial review against Strata-143 @ 1797c0fb (read-only, nothing built or run)**

The plan is sound in direction. Before implementation it needs these corrections:
- **1b as written** evicts conversations too early and has the wrong ordering and join points. A corrected spec is in section C.
- **2a** loses its "superseded copy" detection.
- **2b and 3d** would break the stdout protocol.
- **3a** has an unsafe RAM accounting rule and a write-once key that is not byte-exact.
- **Two existing defects** (A1, A2) sit right in the code the plan builds on and should be fixed first.

## A. Existing defects that change the plan

**A1. Stale reused K/V after a slot give-back (upstream code; exactness; plausible, untested).**
- When a parked conversation is restored, the engine keeps its K/V pages so the next park only copies what changed (`retain`, :11064). The only correction later is `limit_reuse(read_from)` (:11170).
- Parking takes those pages with `take_reuse` (:8139), but the park returns early before that line when `!live_ok` (:8108).
- `copy_from_slot` (:10957ff) then overwrites the main session's cells `[0, slot_tokens)` with a different conversation's. It never drops the kept pages. `limit_reuse(slot_tokens)` leaves `unchanged_tokens = min(old, slot_tokens) > 0`.
- The next park then reuses host pages of the wrong conversation for `[0, unchanged)`. The result is a corrupted parked image, and wrong tokens if that image is ever restored.
- **Trigger:** request M restores image A, then ends with `live_ok=false` (cancelled mid-chunk, or `ckpt=0` from `"strata_checkpoint": false`, or a non-finite logit hit). Request M+1 then continues from a slot.
- **Fix (one line):** call `conversations.take_reuse()` before `copy_from_slot` writes, as SESSION RESTORE already does at :10558.
- **Test:** that exact sequence, with `STRATA_SNAPSHOT_VERIFY=1`, against `STRATA_SNAPSHOT_FULL_CAPTURE=1`.

**A2. With `STRATA_SLOT_CKPT_MOVE=1` (our patch, on in the default config), parking drops unrelated conversations that share the system prompt.**
- At BADM the turn checkpoint moves into the slot. The parked main-session image then keeps only the root checkpoint (`prompt_cache_every=16384`, so no periodic one below 16K).
- `drop_superseded` (`conversation_cache.hpp:312`) treats an entry as superseded when the outgoing chain holds that entry's deepest checkpoint. That deepest checkpoint is now the shared root, so every other conversation with the same ≥2048-token system prompt is dropped on each park.
- The audit benches never hit this: 0 "superseded" lines in every `kvaudit` log, because every conversation starts with a different prompt.
- **Fix:** decide supersession by token prefix. An entry is superseded only when its deepest checkpoint is longer than the outgoing chain's shallowest (root) checkpoint and the new `live.ids` starts with it. Never drop on a pinned or root checkpoint alone.
- 2a needs the same fix (see B).

**A3. The park cost is mostly page faults and zero-fill, not PCIe.**
- Restoring from RAM whose pages are already mapped runs at 287 MB / 35.8 ms ≈ 8 GB/s.
- Capturing into fresh memory runs at 287 MB / 147 ms ≈ 1.95 GB/s. `ConversationBuffer::resize(n, 0)` faults in and zeroes about 70K 4K pages, and then the D2H copy writes them again.
- Transparent huge pages are set to `madvise` on this box (AnonHugePages is 0).
- Cheap exact options: allocate without zero-fill (every byte is overwritten or the image is discarded), plus `MADV_HUGEPAGE` on the 16 MiB segments, or a bounded segment free-list. This also helps Stage 3 reads.
- Measure with `perf stat -e page-faults` before choosing.

**A4 (minor).** `copy_from_slot` with a checkpoint (`at != nullptr`) still takes the host round trip even with `SLOT_COPY_DIRECT` (:9739). A device-to-device K/V copy plus `checkpoint_restore` would be exact. The audit logs only show "all it holds", so this path is rare today.

## B. Verdicts per stage

**Stage 0 (`--conversation-cache-min-tokens 2048`): sound.**
- It is exact, and the early return comes before `take_reuse` without creating a new stale-reuse path (`wants` false implies `live` is shorter than any restored image).
- The gain assumes one-shot prompts are all under 2048 tokens.

**1a (park stages in parallel): sound, with fixes.**
- `ckpt_threads` is declared at :8332, after `park_current_body` (:8107). Declare a separate `StageThreads park_threads` before :8107.
- Each job needs its own `err` string and `reused_bytes` counter, and must catch `bad_alloc` and rethrow it on the main thread.
- The `cudaDeviceSynchronize` calls are fine here: everything is drained at :10955.
- The plan's idea of taking checkpoint buffers from `CheckpointPool` misses the main cost (the K/V segments, A3).
- Expected 147 → ~45 ms is plausible (the work is CPU and page-fault bound, and it scales per thread). This is unconfirmed.

**1b (async park): right idea, but the mechanics as written are wrong in five places.**
1. **Evicting at capture time is premature.** Today the next request first takes the image it will restore out of the cache (`incoming = take(...)`, before the park), so eviction can never hit it. Running `drop_superseded`, `make_room` and the RAM-gate evictions at BADM can evict exactly the conversation the next request restores, because oldest-first targets the one coming back. They also evict for images that are then dropped (the request continues `live`). Capture-time checks must not destroy anything; `put(image, held)` does the evictions at decision time.
2. **Workers must not call `cudaDeviceSynchronize` while groups run.**
   - `conversation_snapshot_save` calls `sync()` (`conversation_state.cpp:280`).
   - Batch windows can be waiting on a flag that only the main thread's `batch_poll` raises (`verify.cpp:3190`), so a device-wide wait couples correctness to the main thread's state.
   - The pump also captures new batch graphs lazily (ThreadLocal mode), and a device-wide sync from another thread during a capture is a known hazard.
   - Order the work once on the main thread at capture start, while still drained; workers then sync only their own private streams.
3. **The join list is incomplete.**
   - `checks` is mutated at the top of the request by the non-finite flush (:10828).
   - `kvg_ensure` (:10818) can move K/V before :10955.
   - The worker reads `checks` (today's capture deep-copies it).
   - So: cancel the job on a non-finite flush, disable async when the elastic K/V (`kvg`) is on, and wait for a "metadata copied" milestone before any mutation of `checks`.
4. **Decide first, then join or cancel.** The decision inputs (`from_live`, `incoming`, `slot_source`, `pin_sibling`) do not depend on the park.
   - After a solo GEN that is STOPped for a handover, the BGEN arrives within milliseconds and continues `live`, so today nothing is parked.
   - A blind join there would put a full capture on the critical path, which is worse than today.
5. **A dropped image loses the kept K/V pages.** Hand the completed image's K/V back as kept pages (`captured = unchanged = live.size()`).

The plan's claim that nothing writes the main session between BADM and the next request holds once capture starts after `wait_commit`, a per-device sync, and the drafter (`mtp`) being idle, with nothing in flight. I checked: batch windows read only `ss_->ple` (the shared table), and slot sessions are separate.

**2a (park the slot at BDONE): conditionally sound. Needs:**
- **Superseded detection fails.** A slot image's chain holds only the moved turn checkpoint. Its predecessor's deepest checkpoint (turn t−1) stays in the main `checks`, so per-turn copies pile up at 0.4-0.7 GB each. Needs the prefix-based supersede rule (A2).
- **The slot's checkpoints are mutated in many places:** `sl = BSlot{}` frees them, plus the `slot_ck`/`lent` moves, the non-finite clear and `ckpt_pool.give`. Do not copy them asynchronously. Capture `live` + K/V only, hold the image pending, and on the main thread move `sl.checks` into it and `put()` it only when the slot is overwritten or wiped.
- **Ordering of the slot's last window:** record per-stage events when the group leaves each stage, and have the capture wait on them. Do not rely on BDONE implying a finished commit.
- **A draftless image breaks `STRATA_SNAPSHOT_VERIFY`:** :11040 verifies `stage_images.back().kv.back()` as draft K/V, which would ERR and exit the engine.
- **Pad-row deferral** must also feed `ov_layout`'s `parks(S)` check.
- **Join (ii)** happens at BADM, after the whole prompt read, so it is almost always free.
- **Gain:** only for clients that send the reply back unchanged (no thinking stripped). Re-check the bench template. The plan's +3-5 % is an upper bound.
- **Exactness of a draftless restore** relies on Philox(seed, position) (`verify.cpp:2374`, `:3082`) plus window-row invariance. The test the plan lists covers it.

**2b (slot affinity): low value as specified, and the protocol part is unsafe.**
- At saturation the freed slot is the only free one, so affinity rarely chooses anything. Expect about 0 % in 24×4.
- An engine `BWIPE` line would land in the control `lines` queue (`server.py:799`) and confuse the request that holds the control lines. Infer wipes on the server only, or route the line explicitly and gate it on an INFO capability.

**2c (restore before the drain): sound only with worker-thread restores using stream-only syncs.**
- `conversation_snapshot_restore` and `checkpoint_restore` call `sync()`. Today that is only safe because the main thread has already drained.
- Decide with the view from before the drain; a slot that finishes during the drain is ignored for that request. That is still exact.

**3a (SSD spill): needs four fixes.**
1. **Unsafe RAM rule.** Counting pending spill bytes as already free (INFLIGHT 4 GiB) exceeds the default 2560 MiB floor inside a 90 GiB MemoryMax scope, which risks an OOM kill. Set INFLIGHT ≤ floor/2, or add it to the floor. In the synchronous RAM gate, free immediately and spill only from the asynchronous path.
2. **The write-once key is not a byte-content key.** The same tokens are not proven to give the same bytes across read paths, so restoring file A for image B could change later tokens. Key on the payload digest, or only on lineage tracked through the kept-K/V logic once the A1 fix is in.
3. **The root checkpoint (~120 MB per stage set) is duplicated in every file.** Store checkpoints content-addressed, once.
4. **API details:**
   - Compute the session identity once at startup with a silent callback; `next_file` prints SWAIT lines on stdout.
   - `conversation_session_read_limits` needs a draft-pointer overload for stages without the draft layer.
   - Make the meta file the commit record, written last after all 4 stage renames.
   - The read lands in fresh pageable memory (A3), so ~0.3-0.6 s per SSD hit is more realistic than 0.15-0.3 s.

**3b (delta files): necessary, not optional.**
- Once RAM is full, every turn of every agent spills.
- Example: 150 agents × 100 turns × 0.28 GB ≈ 4 TB/day, far over 200 GB/day.
- Add a value-based admission rule, e.g. spill only conversations that have already been restored at least once, longest-first.

**3c (budget, TTL, metrics): fine.** Append the new fields after `parked= bytes=` so `ConvCacheLog.EVENT` still matches.

**3d (warm restart, PREFETCH): unsafe as specified.**
- **Warm restart:** the config fingerprint has no binary build-id and no `STRATA_*` env, and A/B arms share the version string. A file from another arm would restore. Add exe build-id + sorted `STRATA_*` env + argv to the tier fingerprint, and wipe the directory at start unless `STRATA_CONV_TIER_WARM=1`.
- **PREFETCH:**
  - Older engines answer it with `ERR bad request`, which lands in the control queue → gate it on an INFO capability.
  - Exempt it from `pipe_drain`.
  - It must never print to stdout.

## C. Corrected Stage-1 spec

**1a `STRATA_PARK_THREADS=1`** (default off = today's code path)
1. Declare `StageThreads park_threads` before :8107.
2. In `park_current_body`, keep everything up to and including the RAM gate (:8199-8229) unchanged.
3. Replace the serial saves with `park_threads.run(n_st + 1, job)`:
   - Each job has its own `std::string e_k`, `size_t reused_k`, `SavedConversation part_k` and a caught `std::exception_ptr`.
   - job(0) does `OnDevice(0)` and `conversation_snapshot_save(part_0, view, ss, g, draft0, e_0, std::move(reuse), &reused_0)`.
   - job(k ≥ 1) does `OnDevice(stages[k-1]->dev)`, then `cudaDeviceSynchronize()`, then `save(part_k, view_k(cs.parts[k-1]), stages[k-1]->ss, g, draft_of(k-1), e_k, std::move(stage_reuse[k-1]), &reused_k)`.
4. After the run:
   - Rethrow `bad_alloc` (goes to today's skip-parking path).
   - Any false return → `err = e_k` and return false (today's ERR path).
   - Then `image = std::move(part_0)`; push `part_1..N` into `image.stage_images` in stage order; `reused_bytes = Σ reused_k`.
5. `MergeBack` stays as is. The parts are disjoint and `live`/`live_imgs` are read-only.
6. **Test:** add a deep-equality helper `conversation_image_same(a, b)`. It compares geometry, layer range, cvec, `live`/`checkpoints` (ids, imgs, gdn, ple, tails, dead, block_pos, used, pinned), `kv` metadata plus all 5 buffers via `operator==`, and `stage_images` recursively, ignoring capacity. Unit-test serial vs threaded results, then run the exactness tools, `REPS=3` ab_quick and one `kvaudit_mt`.

**1b `STRATA_PARK_ASYNC=1`** (default off; implies 1a's per-stage structure; disabled with a once-only log line when `kvg.on`)

*New API (default arguments keep today's behavior byte for byte):*
- Add `struct SnapshotCopy { bool device_sync = true; uint8_t* bounce[2] = {}; size_t bounce_bytes = 0; cudaEvent_t ev[2] = {}; const std::atomic<bool>* cancel = nullptr; };`
- Thread it through `conversation_snapshot_save` → `conversation_checkpoint_save` and `conversation_kv_save`.
- With `device_sync == false` there is no `sync()`.
- Copies run as `cudaMemcpyAsync` (D2H) into `bounce[i&1]` on the private stream, then an event record; wait for the previous event, then memcpy it into the `ConversationBuffer`.
- `cancel` is checked between 16 MiB pieces and returns false with error `"cancelled"`, which is not a CUDA error.

*ParkWorkers:*
- One persistent thread per stage device, pinned with `aux_cpus::pin_current_thread()`.
- 2 × 16 MiB `cudaHostAlloc` bounce buffers per device (128 MiB total), allocated at startup only when the env switch is on.
- Methods: `submit` (non-blocking), `wait_meta()`, `join()`, `cancel()`. The destructor cancels and joins.

*Capture start (main thread):*
1. Where: the end of every request iteration, after `DONE`/`YIELDED`/`BADM` are printed and after `copy_to_slot` and the checkpoint move (:13290-13340).
2. Start only if all of these hold:
   - the cache is enabled, `live_ok`, `!live.empty()`, `wants(live.size())`;
   - `!nan_flush`, `!kvg.on`;
   - `!pipe_inflight()` (with `PIPE_OVERLAP` groups may be running: no async, today's sync path);
   - no job pending.
3. Ordering: `ver.wait_commit`, `mtp.idle()`, then `cudaDeviceSynchronize()` on every stage device. This is about 0 ms after `copy_to_slot`'s `sync_all`.
4. Copy by value: `live`, `live_imgs`, `cvec_cached`. Then `reuse = conversations.take_reuse()`. Store `gen = main_gen`. `main_gen` is a counter incremented by every main-session writer: prompt read start, RAM/file restore, `copy_from_slot`, `session_zero`, kvg grow/trim.
5. Admission without evicting:
   - Compute the estimate exactly as :8130-8180.
   - If it is over the budget, or `conversation_memory_admit(available, additional, floor)` fails with no eviction, then give the kept pages back with a new `conversations.retain_reuse(std::move(reuse))`, start no job, and leave the request to today's synchronous park.
6. Submit per-device jobs:
   - **Phase A:** copy the complete checkpoints from `checks` as stage parts, with exactly the semantics of `conversation_checkpoints_split`: complete ones only; parts carry ids, imgs, used, pinned. Read `checks` only, then signal meta-done.
   - **Phase B:** live running state, K/V layers, and the draft layer on the last stage, with `device_sync=false`, the bounce buffers and cancel.

*Join and decision points (main thread):*
- Before any mutation of `checks` or `bs[].checks`: `wait_meta()`.
- At the top of the request (before :10811): if `nan_flush`, cancel + join + discard (kept pages discarded).
- SAVE/RESTORE (:10392), VRAM (:10309), QUIT and teardown: cancel + join + discard.
- At :10955, compute `park_due = (!from_live || incoming || slot_source >= 0) && !pin_sibling` exactly as today, then:
  - **park_due:**
    1. `join()`; a CUDA error → `ERR` and exit, as today.
    2. Require `gen == main_gen`; otherwise discard and log a bug line.
    3. Run today's post-capture RAM floor check (:8243).
    4. Call `put(image, held)`, which does `drop_superseded` + `make_room(n, held)` in today's order.
    5. Log `conversation cache: parked N tokens in <wait ms> ms; parked=.. bytes=.. evictions=.. snapshot_bytes=.. reused_kv_bytes=.. async_capture_ms=..`.
  - **Not due:** `cancel()` + `join()`. A completed image's K/V (stage 0 + stages) goes back as kept pages with `captured = unchanged = live.size()`; otherwise discard it.
  - **No job pending:** today's `park_current(held)`.

*Exactness guarantees:*
- No main-session writer runs between capture start and the join; `main_gen` asserts this.
- The metadata matches today's image.
- The kept pages are the same object today's park would take.
- `STRATA_PARK_ASYNC_VERIFY=1`: at a due join, also run a synchronous full capture (no kept pages) and require `conversation_image_same`. A mismatch → ERR + exit. This also exposes A1.

*RAM:*
- The pending image is outside the cache budget until `put`: one job at a time, plus 128 MiB pinned.
- The physical gate is checked at start, the floor again at `put`.

*Tests:*
- All exactness tools with the 16-slot env.
- `VERIFY=1` across `kvaudit_mt`.
- Forced sequences:
  - solo STOP→BGEN handover (expect cancel with a small wait);
  - non-finite injection while a job is pending;
  - BSTOP, VRAM and QUIT while a job is pending;
  - the A1 trigger.
- `REPS=3` ab_quick against `-64kho` and `-64khoc`; `kvaudit_mt` (report join-wait p50/p95).

*Expected:*
- One-shot: within noise.
- Multi-turn: about +4-5 %, from the 8.8 s park stall, minus captures that end up cancelled.
- Back-to-back admissions at startup gain nothing, but are no worse than today.

## D. Recommended order

1. A1 fix and A2 fix, each opt-in with its own test.
2. Stage 0.
3. 1a, then 1b. Optional: an `STRATA_PARK_HUGEPAGE` / no-zero-fill switch (A3).
4. 2a, with the prefix-based supersede rule and lazy checkpoint attach.
5. 3a, then 3c, then 3b.

2b: server-side only, low priority. 2c and 3d only after the fixes above.

All references are to `/media/raid0/StrataQ38FN/Strata-143` (`src/program/generate.cpp`, `src/core/conversation_state.cpp`, `src/core/conversation_snapshot.cpp`, `include/strata/core/conversation_cache.hpp`, `src/core/verify.cpp`, `serve/server.py`) and the logs in `/media/raid0/StrataQ38FN/linux/results/kvaudit/`.