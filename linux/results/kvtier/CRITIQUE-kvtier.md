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