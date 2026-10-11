# Design: cutting the fixed per-admission cost (piped batch mode, 0.1.42 base)

Read-only analysis of 2026-10-11 (workflow: 3 investigators, synthesis, adversarial critique). The critique's corrected ranking at the end supersedes the plan's ranking.


Status: this is a plan built from reading the code. Nothing was built or run for it. It combines three code readings (drain+refill, header+first window, checkpoint+copy) into one ranked list. All line numbers refer to `/media/raid0/StrataQ38FN/Strata-142` (branch `g3-142`). Per AGENTS.md, every step is opt-in behind its own env switch and defaults to off.

## 0. Premises and corrections

1. **The m0 phases were measured on the 0.1.41 binary, not on 142.** `strata-g3-mmqtab` logs "engine 0.1.41", and in that tree the D2D slot copy is on by default. The phase split below still holds for 142 with `STRATA_SLOT_COPY_DIRECT=1`, which T16-142 sets, but step 0 re-measures it.
2. **The drain costs almost nothing; the refill does.** In `pipe_drain` (generate.cpp:9694, called at 9736) the 4 in-flight groups each finish once, which is the steady-state completion rate. The real loss is the refill after BADM: about 4 stage times with no completion (23.6 ms). The model is therefore calibrated on R_eff (285 ms short / 698 ms long, which reproduces 444/327 tok/s), not on the wall stall (314/720).
3. **Most of "copy_to_slot 7.8 ms" is a host deep copy of the turn checkpoint.** It is `sl.checks.push_back(*best)` at generate.cpp:12519 (112.5 MiB of host vectors), which sits inside the "copied in" timer. The D2D part is about 0.1 ms of transfer plus about 116 per-copy stream syncs. This is inferred, and step 0 confirms it.
4. **Fixed stall per admission (wall, today):**
   - BGEN drain: 27 ms, roughly free in throughput terms.
   - Pre-read: 1 ms.
   - Turn checkpoint: 17 ms.
   - Header `read_windows` (T=4 + T=2): 23.6 ms.
   - First window (T=1): 11.5 ms.
   - copy_to_slot: 7.8 ms.
   - BADM → first BT refill: 23.6 ms.
   - Total ≈ 112 ms. Everything except the drain is real loss.

## 1. Throughput model

T = D / (1 + D·R/L), with D = 590 rows/s, L = 512, R in seconds, R_new = R − saved.

| Baseline | R short | R long | tok/s short | tok/s long |
|---|---|---|---|---|
| Today, no overlap (R_eff, calibrated) | 285 | 698 | 444 | 327 |
| `STRATA_PIPE_OVERLAP=1`, x ≈ 0.7 (R − 0.7·read; read 201 / 609 ms) | 144 | 272 | 506 | 449 |

The overlap row is a model assumption (x ≈ 0.7). It has not been measured on 142. It already includes the extra bubble that overlap adds: the restart inside the read, then the end-of-read drain at 11095.

Rule of thumb: 10 ms of R is worth about +0.85% short / +0.6% long without overlap, and about +1.0% / +0.9% with overlap.

## 2. Ranked plan

Savings are per admission, and the totals are cumulative. Throughput in the table is cumulative after the step, at 16 users.

| # | Step (env) | Δ ms (no ovl / ovl) | Cum. ms (no ovl / ovl) | tok/s no overlap, short / long | tok/s overlap x=0.7, short / long | Exact | Effort |
|---|---|---|---|---|---|---|---|
| 0 | Instrument and re-baseline on 142 | 0 | 0 | 444 / 327 | 506 / 449 | n/a | 0.5 d |
| 1 | **S1** Move the turn checkpoint into the slot | 6.5 | 6.5 | 447 / 328 (+0.6 / +0.4%) | 509 / 452 | bit-identical | 0.5 d |
| 2 | **S2** One sync per device in the direct slot copy | 1 | 7.5 | 447 / 329 | 510 / 452 | bit-identical | 0.5 d |
| 3 | **K1** Recycle checkpoint buffers | 4 (3–6) | 11.5 | 449 / 329 (+1.0 / +0.7%) | 512 / 454 | bit-identical | 0.5–1 d |
| 4 | **O2** Raise the stage-0 PLE flag at launch; **O1** no BYIELD drain when piped | 0 (enabler / conditional) | 11.5 | same | same | bit-identical | 0.75 d |
| 5 | **A** Header windows at `max_t` (8), plus pre-capture of T = 5..8 | 11 | 22.5 | 453 / 332 (+2.0 / +1.5%) | 517 / 458 | bit-identical (row invariance) | 1 d |
| 6 | **B** Header and first token in one window (admissions) | 11 (22 vs today) | 33.5 | 457 / 334 (+3.0 / +2.2%) | 523 / 463 (+3.4 / +3.0%) | bit-identical (row invariance + Philox per position) | 1.5–2 d |
| 7 | **K2** Pinned, parallel 4-stage checkpoint D2H | 10 | 43.5 | 462 / 336 (+3.9 / +2.9%) | 529 / 467 (+4.5 / +4.0%) | bit-identical | 1.5–2 d |
| 8 | **O10** Admission linger when `waiting>0` | 0 / 0 fixed (≈15–20 per back-to-back admission) | 43.5 | same | same | bit-identical (timing) | 0.5 d |
| 9 | **O3** Drain on demand at BGEN (overlap only) | – / 10 (5–20) | 43.5 / 53.5 | same | 534 / 471 (+5.6 / +4.9%) | bit-identical | 1–2 d |
| 10 | **K5/O6** Checkpoint without drain (overlap; needs K2) | – / 2.5 | 43.5 / 56 | same | 536 / 473 (+5.9 / +5.2%) | bit-identical | 1 d |
| 11 | **O7** The single merged window as a per-stage pipeline job | 10 | 53.5 / 66 | 466 / 339 (+4.9 / +3.5%) | 541 / 477 (+7.0 / +6.1%) | bit-identical | 4–6 d |
| 12 | **O5 + O8** Continuous admission, holding only `g_a`, no BADM refill | ~8 / 24 | 61.5 / 90 | 469 / 340 (+5.6 / +4.1%) | 555 / 488 (+9.8 / +8.6%) | bit-identical | 3–4 d on top of O7 |

Totals by phase:
- **Phase 1** (steps 0–6): about 5.5 days for 33.5 ms, roughly +3% / +2.2%. All exact, low risk, no protocol change.
- **Phase 2** (steps 7–10): about 4.5–5.5 days for 10 ms more without overlap, or 22.5 ms more with overlap.
- **Phase 3** (steps 11–12): about 7–10 days.

Steps interact, so the savings do not simply add up past phase 2. Treat the phase 3 numbers as ±10 ms.

---

## 3. Steps in detail

### Step 0: instrument and re-baseline (0.5 d)

Measurements to add:
- `tr()` timers around each per-stage `checkpoint_save` inside `checkpoint_at` (generate.cpp:8003-8031), split into `gpu_sync_ok` (7996), `resize`/zero-fill and D2H.
- `tr()` timers around `copy_to_slot` (12487) and around the `sl.checks.push_back(*best)` at 12519, measured separately.
- A `captured` trace line (or the existing one) checked against the admission window.

Then run `m0_driver.py` / `m0_analyze.py` against the **142** binary with T16-142 (and once with `STRATA_PIPE_OVERLAP=1`, to get the real x). Record it in the TESTLOG.

Gate: if the `push_back` turns out not to be about 6–7 ms, re-rank S1 and K1.

### Step 1: S1, move the turn checkpoint into the slot (`STRATA_SLOT_CKPT_MOVE=1`)

- **Change** (generate.cpp:12508-12519): when `best` is this request's own turn checkpoint (`best->ids.size() == turn_at`, not root or pinned), `std::move` it into `sl.checks` and erase it from `checks`. Otherwise deep-copy as today.
- **Why moving is safe:** in batch mode the main `checks` entry is pruned at the next admission of another conversation anyway (`remove_if` on non-prefixes, 10440-10442). The next turn finds the checkpoint through `slot_source`/`slot_ck` (10272-10289).
- **Saves** 6–7 ms, plus the ~60 ms first-fill copies and the ~70 ms page-fault outliers.
- **Exact:** the same bytes are restored, only from a different container.
- **Side effect:** a session SAVE issued right after a batch admission lacks the turn checkpoint (it still saves `live`). Document this.
- **Tests:**
  - `batch_interleave_test.py` case 3 (next turn from the slot) and case 7 (F from slot 1's turn checkpoint equals solo F).
  - `batch_groups_cache_test.py`: rounds 2 and 3 must still reuse ≥90%; diff OUT.json between env 0 and env 1.
  - `batch_overlap_test.py` (A's next turn, G2 from G's slot).

### Step 2: S2, one sync per device in `direct_slot_copy` (`STRATA_SLOT_COPY_BATCHSYNC=1`)

- **Change:** add an async variant of `copy()` (conversation_state.cpp:34-40, `copy_nonblocking` in conversation_copy.hpp:15-30) that does not end with `cudaStreamSynchronize`. Thread it through:
  - `conversation_checkpoint_copy` (conversation_state.cpp:197-224);
  - `conversation_kv_copy` (conversation_snapshot.cpp:223-247).

  `direct_slot_copy` (generate.cpp:9160-9188) then queues every copy on the private stream and syncs once per device. The two `sync_all` calls stay.
- **Saves** about 1–1.5 ms (≈116 stream syncs become 4).
- **Exact:** same copies in the same stream order.
- **Tests:** `batch_test16.py` (every slot equals solo) and `batch_interleave_test.py`, each with env 0 and env 1. `STRATA_STATE_HASH=1` on the admitted slot must match.

### Step 3: K1, recycle checkpoint buffers (`STRATA_CKPT_POOL=1`)

- **Change:**
  - Keep a free list of same-size `ConvCheckpoint` vector sets, fed by LRU eviction (8065), pruning (10440) and old `sl.checks` when `sl = BSlot{}` (12497).
  - Add an "into existing storage" variant of `conversation_checkpoint_save` (conversation_state.cpp:155-171), so that `resize` to the same size is a no-op: no zero-fill, no page faults.
  - Count pooled bytes against the conversation-cache budget.
- **Saves** about 3–6 ms of the 17 ms checkpoint, plus the fault spikes.
- **Exact:** every byte is overwritten by the D2H, and the size is identical.
- **Tests:** same as S1, plus a long run (≥200 admissions through `m0_driver`) to check that RSS stays flat.

### Step 4: O2 + O1, safety enabler and spurious drains (0.75 d)

- **O2** (`STRATA_PLE_FLAG_AT_LAUNCH=1`): in `batch_launch` (verify.cpp:2926-2935), right after `cudaGraphLaunch`, raise `h_flag_` when `ar_on() && ple_stage()`, as `run_slot_rows` does at 2715-2718. Keep 2952-2956 as an idempotent re-raise.
  - `h_ple_` is already complete, because `gather_batch` in `stage_batch` (2655) is synchronous.
  - This removes the hazard where a `cudaDeviceSynchronize(dev0)` after a stage-0 launch would wait forever. Steps 9–12 need it.
  - Exact: the flag only gates timing.
- **O1:** in piped mode, stop draining for BYIELD unconditionally (generate.cpp:9736; today only with the overlap env). `batch_read_piece` returns 0 when piped (batch_read.hpp:24), so the drain is pure loss. Also stop the server sending BYIELD when `slot_gs` (server.py:1300, 1402).
  - Saves about 24 ms per BYIELD sent, which only happens while shorter requests are queued.
- **Tests:** `batch_overlap_test.py` (it sends a BYIELD during B's read) with env 0 and env 1 must be token-identical. Also `batch_interleave_test.py` case 4.

### Step 5: A, header windows at verifier `max_t` (`STRATA_WINDOW_MAX_T=1`)

- **Change:**
  - In `read_windows` (generate.cpp:10652-10705), change `T = min(S, b - q)` at 10668 to `min(Sw, b - q)`, with `Sw = env ? ver.max_t() : o.spec`.
  - Resize `win/outw/nxt` (10662) to `Sw`.
  - Add `int max_t() const` to `Verifier` (verify.hpp; `max_t_` is set at verify.cpp:488 and is 8 on this config per generate.cpp:7456-7458, 7528-7529).
  - Do not touch `read_windows_pl` (10578).
  - At warm-up, pre-capture `exec_[T]`/`commit_exec_` for T = 5..8 (lazy capture today at verify.cpp:1996), so that no capture lands inside an admission.
  - `mtp.prefill` already chunks by its own `max_t` (mtp.cpp:1229-1231).
- **Saves** about 11 ms per admission: 6 header tokens take one window instead of two. Every short read of 64 tokens or less also needs half the windows (40 tokens: 10 → 5 windows, about 55 ms).
- **Exact, by row invariance:**
  - Dense and head matvecs keep each column bitwise equal to a single-column call (native_mmvq.cu:1150-1154, `g_multi_exact`).
  - GDN steps row by row (`gdn_step_norm_multi`, verify.cpp:1099/1105).
  - Each window row appends its K/V to the int8 pool before attention (`kv_append_q8_steps`, verify.cpp:1212-1214). Attention is decode-style per row over that pool, bounded by the row's step (`qsa_decode_attn_batch`, ~1321).
- **The one open point:** the QSA indexer appends all n rows (verify.cpp ~1240) before it scores per row. This relies on the step bound hiding later rows from earlier ones. That is proven for T ≤ 5 by the existing batch == solo tests (solo MTP windows), but T = 6..8 single-sequence windows are new in production (today they exist only behind `--lookup-chain`).
- **Tests:**
  - `STRATA_STATE_HASH=1` after admissions of 150, 1.2K and 5K prompts, thinking on and off. The gdn/ple/tail/pooled/kv/mtp/dead hashes (generate.cpp:12345 and following) must be equal between env 0 and env 1.
  - `batch_test16.py`, `batch_interleave_test.py` and `batch_overlap_test.py` with env 1, against solo and against the env 0 dumps.
  - Include a prompt whose header crosses a QSA block boundary.

### Step 6: B, header and first token in one window (`STRATA_ADMIT_MERGE_FIRST=1`, admissions first)

**Change, cut loop** (generate.cpp:11348-11400). When env is on and all of the following hold, skip the last `read_windows` and remember `hdr = turn_at`:
- BGEN admission (`max_new == 1` path);
- last cut is `turn_at → n-1`, `windows_ok`;
- `n - turn_at ≤ ver.max_t()`;
- no pictures, no `STRATA_LOGPOS`, not `pipe` (11475).

Then set `p = hdr` (11418) and `consumed = ids[0..hdr)` (11446), and size `window/outv/drafts` (11421-11422) to at least `max_t`.

**Change, first window** (12128):
- `T = n - hdr`, `window = ids[hdr..n)`.
- Force `a = T-1` instead of the acceptance loop (12224), then `ver.commit(T)` (12232).
- Push all T tokens to `consumed`; emit only `outv[T-1]` (12241-12248).
- `x = outv[T-1]`, `p += T` (12298-12300).
- `STRATA_DUMP_FIRST_LOGITS` should copy row T-1 (12207).
- Call `mtp.prefill(final_R_all, ids + hdr + 1, T-1, hdr)` so the MTP K/V rows match today's header windows. The draft is skipped anyway (12259, `produced_n >= max_new`).

**Scope:** admissions only. Solo GEN keeps today's path, so it would need no MTP draft-from-row-T-1 change. `mtp.draft` rejects T > `max_t_` (mtp.cpp:1285) and would feed the wrong next tokens for prompt rows (1306-1309).

**Saves** about 22 ms against today (11 on top of A; header plus first window 35.5 → ~13 ms). TTFT improves by the same amount. No protocol change: the `T`, `DONE 1` and `BADM` lines are unchanged, and `live.size() == p` (12484-12485) still holds with n tokens.

**Exact:**
- Row T-1 sits at position n-1, so its Philox counter (`pos0 + t`, verify.cpp:2155) equals today's T=1 window.
- `penalty_rows` (12180) gives row T-1 the same history tail, because `consumed` starts at `hdr`. logit_bias is identical.
- Rows 0..T-2 are sampled and discarded. Today's header has head sampling off (10660-10664); the sampler is counter-based and has no state to disturb.
- `spec_q` is never set on a first window (12197).
- The state is exact by the same row invariance as A.

**Tests:**
- Because solo GEN is unchanged, `batch_test16.py` / `batch_test32.py` (slot == solo) directly prove "equal to today".
- Add seeded sampled requests, requests with `penalty_last_n > 0` (token 1 must still match, since B keeps penalties on row T-1), and `max_new = 1`.
- STATE_HASH env 0 vs env 1. `batch_interleave_test.py` (cases 3 and 7: the next turn must restore the same checkpoint at `turn_at`). `batch_overlap_test.py`, including STOP during the read (cancel: `pp_reached` stays at `turn_at`, 12332-12341).
- `m0_analyze`: `header_windows_ms + first_window_ms` should drop from about 35 to about 13.

### Step 7: K2, pinned, parallel checkpoint D2H (`STRATA_CKPT_PINNED=1`, builds on K1)

- **Change:**
  - Back the K1 pool with `cudaHostRegister`ed memory: (prompt_cache 6 + 16 slots + 2) × 112.5 MiB ≈ 2.7 GiB, out of 128 GB RAM.
  - In `checkpoint_at` (generate.cpp:8022-8031), issue all 4 stages' D2H async, then wait once (a per-stage event), instead of sync-and-save per stage in turn.
  - Fall back to today's path on HIP or Windows, or if registration fails.
- **Saves** about 10 ms on top of K1 (17 → ~2.5 ms: about 28 MiB per link, in parallel).
- **Exact:** same bytes.
- **Risk:** PCIe contention with the mapped hand-off and PLE traffic (PLE is in RAM). Measure the decode rate during admissions.
- **Tests:** same as S1/K1, plus `batch_overlap_test.py` (the 5K prompt with a root checkpoint covers two checkpoints per admission).

### Step 8: O10, admission linger (`STRATA_ADMIT_LINGER_MS=N`)

- **Change:** at BADM (12523-12530 → main loop 9719-9738), if the server reported `waiting>0` (commit d72209a9), wait up to N ms for the next BGEN before restarting the groups. Fall back to a restart when N runs out.
- **Saves** about 15–20 ms per back-to-back admission: today the 1–2 restarted groups are drained again almost at once. It does not reduce the fixed cost of an isolated admission. It matters for bursts and for users > slots.
- **Exact:** timing only.
- **Tests:** `batch_overlap_test.py` (admissions one after another), and an `m0_driver` burst arm comparing R_eff.

### Step 9: O3, drain on demand at BGEN (overlap only, `STRATA_PIPE_LAZY_DRAIN=1`; needs O2)

- **Change:**
  - Skip the BGEN drain at 9736.
  - Call `pipe_drain()` lazily before each site that needs it:
    - `copy_from_slot`/`park_current` (10327-10333);
    - `lend`/`refill`/`res_upload` when `pf_parts` is non-empty (10750-10830);
    - `apply_pending`/`adapt_tick` with pending work (11239-11240, 9760);
    - `kvg_trim` (10450);
    - `read_windows`;
    - the non-overlapped `sp.run` fallback;
    - `overlap_prepare`, only when `ensure_batch_layouts` would capture (dry-check `has_batch_layout` first).
  - Replace the `pipe_inflight()` refusal at 10849 with that.
- **Saves** about 5–20 ms (the read-start bubble). Read it from `lost` in the overlap stats line.
- **Exact:** pre-read work touches only the main session and host state. Table changes stay gated as in `overlap_gate` (10853-10869).
- **Tests:** `batch_overlap_test.py` (all arms, including the G/H pad-up case and the STOP during a read), env 0 vs env 1 dumps identical.

### Step 10: K5/O6, checkpoint without drain (overlap; needs K2 and O2)

- **Change:**
  - Run `checkpoint_at` on a helper thread.
  - Replace `gpu_sync_ok` (7996-8001) with an event recorded after the main session's last writer (the `sp.run` chain join).
  - Postpone `read_overlapped`'s final drain (11095-11104) until just before the first verifier use, i.e. the merged window from B.
  - The #613 hang diagnosis becomes an event-error check.
- **Saves** the remaining ~2.5 ms after K2. Without K2 it would save 12–17 ms, but K2 is cheaper and also helps the non-overlap mode.
- **Exact:** slot windows never write the main session (DESIGN-overlap).
- **Tests:** `batch_overlap_test.py` with an injected fault (`STRATA_PIPE_OVERLAP_TEST_FAULT=cuda`) to exercise the error path, and `batch_interleave_test.py` case 7.

### Step 11: O7, the merged window as a per-stage pipeline job (`STRATA_ADMIT_WINDOW_PIPE=1`)

- **Change:**
  1. `Verifier::run_local`/`commit_local` without the `next_` recursion (verify.cpp:2150, 2371).
  2. A stage reservation in `pump` (generate.cpp:9587-9698): no group launches on stage k while the solo job holds it, and the job waits until `stage_group[k] < 0`.
  3. Separate hand-off rows: `hrows = batch + max_t`, with solo `hrow0 = o.batch` (verify.cpp:825/847/1606; generate.cpp:7413-7431). Today solo windows write rows 0..T-1, which belong to groups 0 and 1.
  4. On the last stage, read `h_out_`/`final_R_` and record the `mtp.prefill` event before releasing the stage (pattern: `pl_mtp_ev`, 10616-10635).
  5. Pre-capture T = 1..8 (already done in step 5).
  6. A per-job `drive.d.failed` flag, and a path through `release_gpu_waits`.
- **Saves** about 10 ms. After B only one ~13 ms window is left, and it becomes about 4 × 3 ms of per-stage occupancy while the groups keep decoding.
- **Exact:** same graphs, same inputs, same per-session order. The hand-off row offset changes no values.
- **Tests:** everything above, plus a forced pad-up arm in `batch_overlap_test.py`, and an abort during the window.

### Step 12: O5 + O8, continuous admission (`STRATA_PIPE_CONTINUOUS=1`)

- **Change:**
  - After the last solo window, restart all groups except `g_a`, the group that holds `admit_slot`. Hold `g_a` only if `admit_slot` lies below its window width `G.S` (pad rows write inactive slots below the last active one, 9670-9675).
  - Copy on a helper thread, gated by an event instead of `sync_all` (9160-9188); the hold mask goes at the pump start pick (9650-9680).
  - Release `g_a`, then BADM.
  - Together with steps 9–11, no full drain is left anywhere.
- **Saves:** with overlap, the BADM refill (~24 ms) plus copy overlap. Without overlap, about 8 ms: the read still drains, and only the refill moves to after the read, overlapping the now-short tail.
- **Exact:** the data is disjoint. Only `g_a`'s pad rows could write `admit_slot`, and `g_a` is held.
- **Tests:**
  - `batch_overlap_test.py`: G short into the last slot, then H into the slot below it, which is exactly the pad hazard.
  - `batch_test16.py`, `batch_interleave_test.py`.
  - STATE_HASH on the admitted slot.
  - `m0_driver`/`m0_analyze` comparing R_eff.

## 4. Not recommended or parked

| Option | Why not |
|---|---|
| C, deferred first token (`BADM <slot> 2`) | B already removes the first window (~22 ms, against ~11.7 for C) without a protocol change. C adds a batch row per request, adds 30–50 ms to TTFT, and is lossy for penalised requests unless they are gated out. |
| D, header as forced slot rows | Gains ~2 rows over B but adds ~150–190 ms of TTFT. Only for deep queues. |
| K3, device snapshot + lazy D2H | K2 gets the checkpoint to ~2.5 ms with less consumer-side risk. |
| K4, per-slot VRAM checkpoint | ~450 MiB per GPU for ≤3 ms left after S1 + K1 + K2. Revisit only if step 0 shows otherwise. |
| Reading straight into the slot session | Saves ≤1–2 ms once S1/S2 are in, costs 6–10 days, and is not guaranteed bit-identical. Only as part of a second prompt path later. |
| K6, `ckpt=0` | Saves ~45 ms but is **not bit-identical to ckpt=1**, and loses next-turn reuse. Client policy only (`"strata_checkpoint": false`). |
| K7, `ckpt=split` | Saves ~24 ms and is bit-identical, but loses the next-turn checkpoint. A policy choice for one-shot traffic, not an engine fix. |
| O9, pre-capture of all batch widths at init | Fixes only the 30–70 ms first-use outliers, and VRAM on CUDA3 is tight. Covered for the window sizes by step 5. |

## 5. Test matrix (every step, env 0 vs env 1)

| Tool | Covers | Steps |
|---|---|---|
| `tools/batch_test16.py` (`--layer-split 12,24,36 --pcie-frac 0 --adapt-every 1000000`, `STRATA_IQ_MT_MIN=1`) | every slot == solo, greedy; add seeded and penalised arms | all (key test for B) |
| `tools/batch_test32.py` | the same at 32 slots (>8 rows per group cap) | 2, 5, 6, 11, 12 |
| `tools/batch_interleave_test.py` | read while decoding, next turn from slot (3), BYIELD (4), slot → solo continuation (6), turn checkpoint from slot (7) | 1–7, 10 |
| `tools/batch_overlap_test.py` (`--env STRATA_PIPE_OVERLAP=1 --dump`) | overlapped admissions, root checkpoint at 5K, STOP during read, BYIELD, G/H pad-up | 4, 6, 7, 9–12 |
| `tools/batch_groups_cache_test.py` | ≥90% slot or cache reuse over 3 rounds; diff OUT.json | 1, 3, 7 |
| `STRATA_STATE_HASH=1` | gdn/ple/tail/pooled/kv/mtp/dead after admission (150 / 1.2K / 5K, thinking on and off) | 2, 5, 6, 11, 12 |
| `m0_driver.py` / `m0_analyze.py` | phases and R_eff (not wall) | after every step |

Every run goes into the TESTLOG, with the README/CHANGELOG updated, committed and pushed, per the "document every experiment" rule.

## 6. Open measurements before committing to phase 3

1. The step 0 split of checkpoint (sync, zero-fill, D2H) and copy (D2D vs `push_back`) on the 142 binary.
2. The real overlap x and the `lost` value on 142 with T16-142. This sets the overlap column and the value of O3.
3. Window cost per T (T = 1..8) with `STRATA_TRACE`, to replace the 0.15–1.45 ms per extra row estimate for the merged T=7 window.
4. PCIe interference of pinned parallel D2H (K2) with PLE-in-RAM decode traffic.

## Files

- /media/raid0/StrataQ38FN/Strata-142/src/program/generate.cpp
- /media/raid0/StrataQ38FN/Strata-142/src/core/verify.cpp
- /media/raid0/StrataQ38FN/Strata-142/src/core/mtp.cpp
- /media/raid0/StrataQ38FN/Strata-142/src/core/conversation_state.cpp, conversation_snapshot.cpp, conversation_copy.hpp
- /media/raid0/StrataQ38FN/Strata-142/serve/server.py
- /media/raid0/StrataQ38FN/Strata-142/tools/batch_test16.py, batch_test32.py, batch_interleave_test.py, batch_overlap_test.py, batch_groups_cache_test.py
- /media/raid0/StrataQ38FN/linux/results/overlap/m0/ (m0_driver.py, m0_analyze.py, runA-prod/phases.txt)
- /media/raid0/StrataQ38FN/linux/upstream/DESIGN-overlap.md, ANALYSIS-batching-levers.md
---

# Critique and corrected ranking


The review is from reading the code only. Nothing was built or run. All line numbers refer to `/media/raid0/StrataQ38FN/Strata-142`.

## Corrections to the premises

- **P1 holds, and it matters more than the plan says.**
  - The measured binary (`strata-g3-mmqtab`, BUILD.json 0.1.41, branch `g3-mmq-tab`) queued the direct slot copies asynchronously on the legacy stream (`cudaMemcpyAsync(..., nullptr)`) and synced once at the end.
  - 142 sends every copy through `conversation_detail::copy_nonblocking` (conversation_copy.hpp:15-30), which ends each copy with `cudaStreamSynchronize`.
  - The ~116 syncs the plan names therefore **did not exist in the measured 7.8 ms**.
  - The production log confirms the regression: `strata-unsloth-ud-iq4_xs.log` after the 21:19 start of `strata-g3-142` shows "copied in" at a median of about 8.9 ms (n=112), against about 7.9 ms before.
  - 142's `conversation_checkpoint_restore` also gained an extra sync (conversation_state.cpp:178).
- **P2 (the drain is almost free, the refill is the loss) is right.** Within one drain/refill cycle the loss is closer to 3 stage-times than 4. Fine within the noise.
- **P3 is partly wrong.**
  - The "copied in" timer (tc0, generate.cpp:~12487) covers more than the copy: `copy_to_slot`, `set_slot_sampling`, **`sl = BSlot{}` (frees the slot's old 112.5 MiB checkpoint)**, `sl.ids = live` and the `push_back` at 12519.
  - On the measured binary the D2D part was about 0.1 ms plus two `sync_all` calls, not ~116 syncs.
  - The 60 ms fill outliers against the 7.8 ms steady state fit glibc heap reuse: `BSlot{}` frees the old checkpoint just before `push_back` reuses that memory.
- **The model calibration is not the measured R_eff.**
  - m0 measured a median R_eff of **247 ms short / 656 ms long** (W = 2 s).
  - The plan fits 285 / 698 to the server's tok/s.
  - That leaves an unexplained **~40 ms per request at server level**: the server's BDONE→BGEN gap, stray BYIELD drains (see O1), or tail effects.
  - Step 0 must measure this residual through the server. The m0 driver alone cannot.
- **m0 does not exercise several paths:**
  - Multi-turn admissions: slot reuse, the `slot_ck` restore, windowed reads of 64 tokens or fewer.
  - Two checkpoints per admission: root + turn on a new chat whose system prompt is ≥ `--prompt-cache-root`.
  - BYIELD.

  The real-traffic value of A, K1/K2 and O1 can be higher than m0 shows. The value of S2 is lower.

## Verdicts per step

- **Step 0 (instrument): keep, and add the following.**
  - A separate timer for `sl = BSlot{}`.
  - A zero-code arm: `GLIBC_TUNABLES=glibc.malloc.trim_threshold=…:glibc.malloc.mmap_threshold=…` (large values). This shows how much of the checkpoint and copy time is page faults or trim before K1 is built.
  - A server-level R fit, for the 40 ms residual.
  - Log counts of "gave back … (its turn checkpoint)" against "(all it holds)". The current log shows 0 against 184.
  - Admissions with two checkpoints.
  - Overlap x and `lost` on 142.

- **S1, move the turn checkpoint into the slot: keep, with a fix.**
  - The saving is real: the `push_back` copy runs with all slots drained.
  - Exactness holds.
  - Two regressions were missed:
    1. **Duplicate or concurrent identical histories** (n>1, retries, parallel agent calls with the same context).
       - Today the second request finds the turn checkpoint in `checks`.
       - With S1, the checkpoint lives only in the **active** slot, and `slot_source` skips active slots (10276).
       - That request then re-reads the whole conversation.
       - Fix: also let `slot_ck` come from an *active* slot's `sl.checks` (safe because the pre-read part runs drained and KV positions below `turn_at` are never rewritten), or hold the checkpoint payload in a `shared_ptr`.
    2. **Low concurrency, next turn.**
       - Today: a host restore from the main `checks` (~15 ms).
       - With S1: `copy_from_slot(at != nullptr)` takes the **host path even with `STRATA_SLOT_COPY_DIRECT=1`** (9218: `slot_copy_direct && at == nullptr`). That is a full KV round trip through the host (40–90 ms on long conversations).
  - S1 can also move the page-fault cost into `checkpoint_at` (fresh allocations once nothing is freed at the prune), so pair it with K1, or check with the GLIBC_TUNABLES arm.

- **S2, one sync per device: keep, but reframe and fix.**
  - Against the m0 numbers it saves 0 ms. It only wins back the ~1.1 ms 142 lost.
  - Keep the copies on the private **non-blocking** stream (upstream moved off the legacy stream because of HIP graph capture). Do not go back to legacy.
  - Hazard: `refresh_residency` (conversation_snapshot.cpp:205-215) launches `kv_stream_reset`/`kv_ring_restore` on the **legacy stream** for `kv_mode` 1/2. Once the copies are asynchronous on a non-blocking stream, that kernel is no longer ordered after them.
  - Fix: record an event on the private stream and make the refresh wait for it, or sync before it when `kv_mode != 0`.

- **K1, recycle checkpoint buffers: keep (cheaper variant possible).**
  - The zero-fill from `resize` is real and on the critical path.
  - A `default_init` allocator would also skip it, but changes the element type that conversation_file and conversation_snapshot use, so the pool is fine.
  - Effort 0.5–1 d.

- **K2, pinned parallel D2H: fix by doing the cheaper version first.**
  - Problem with `cudaHostRegister` on `std::vector` storage: malloc'ed vectors share pages with other allocations, so registration fails with already-registered errors. It needs page-aligned, pool-owned slabs, which means a type or allocator change. 1.5–2 d is optimistic.
  - **K2-lite** instead: per-stage `checkpoint_save` on 4 *persistent* worker threads (one per device), pageable copies in parallel, together with K1. Expected 17 → ~4–5 ms in about 0.5–1 d.
  - The threads must be persistent: `copy_nonblocking` keeps a thread_local stream per device that is never destroyed, so short-lived `std::thread`s leak one stream per device each time.
  - Go pinned only if K2-lite measures short.
  - The PCIe-contention risk only applies to K5. Under K2 everything is drained.

- **O2, raise the PLE flag at launch: keep, but only as a prerequisite for 9–12.**
  - Raise it only when the launched graph is the zero-doorbell (ar) graph. In the doorbell graph `h_flag_` is the per-layer doorbell (`*h_flag_ = want`, verify.cpp:2996).
  - Standalone value: 0.

- **O1, no BYIELD drain when piped: keep.**
  - It is correct: `batch_read_piece` returns 0 when piped, so a piped read never yields. A BYIELD queued during the read costs a drain plus a refill right after BADM (9736 → 9749).
  - The engine-side change alone is enough.
  - m0 has 0 BYIELDs, so the value only shows through the server. It may explain part of the 40 ms residual. Effort 0.25 d.

- **A, header windows at `max_t` 8: merge into B.**
  - For admissions, B makes A's header saving redundant. B needs A's pieces anyway: T = 5..8 graphs and buffers, and the exactness proof for T = 6..8.
  - A keeps independent value for windowed reads of 64 tokens or fewer (next-turn admissions, solo GEN). m0 does not measure that. Re-tune `--short-read` once T = 8 windows exist.
  - Exactness gaps the plan missed:
    - Row invariance holds only while **every routed expert is GPU-resident**. CPU experts use the multi-token kernel from 2 tokens per expert (`mt_min` defaults to 2, native_expert.cpp:204, "can depend on how many drafts a verify window held"). Changing the window cut changes the tokens per expert, so the bits change unless `STRATA_IQ_MT_MIN=1` is set.
    - Gate A and B like `overlap_gate`: no `host_res < 0`, no loan in flight.
    - The STATE_HASH "mtp" hash may legitimately differ, since the drafter's own windows are cut differently. Do not make it a pass/fail gate.
  - The QSA block-boundary open point stays.

- **B, header and first token in one window: keep (biggest lever), with fixes.**
  - `ver.set_spec_q` at 12197 is gated on `T_mtp > 1`, **not on `first_window`**. With a merged T = 7 and `STRATA_SPEC_PROB`, sampled rows would be judged by rejection sampling against stale drafter q, which is not identical. Add `!merged`/`!first_window` explicitly. The plan's claim "never set on a first window" holds only because T = 1 today.
  - `draft_offered += T-1; draft_accepted += a` would report "6 of 6 drafts" on every admission in DONE and skew server and monitoring acceptance stats. Skip both for the merged window.
  - The emit loop at 12241 prints `outv[0]` first. Emitting only `outv[T-1]` is required, as the plan says, and `eos` and `sfx.append` must also use row T-1.
  - Print the final `PP n-1 …` line before the merged window, so server progress still reaches the end.
  - Prompt-ms and decode-ms in DONE shift by about 12 ms (metrics only).
  - Generalise the gate to "the last window holds the last ≤ max_t tokens of the final windowed segment", i.e. read `[turn_at, n-max_t)` as today and merge the rest. Otherwise a header of 8 or more tokens (thinking-off templates, or #458 tail-role turns that move `turn_at` earlier) silently loses the whole gain.
  - Same resident-experts gate as A.
  - These are verified as fine: Philox `pos0+t`, `penalty_rows` (`kHistSlots` = 4096×8), `ple_prev` after `commit(T)`, `live.size()==p`, batch rows dropping penalties (verify.cpp:2847), and solo GEN left unchanged.

- **O10, admission linger: keep, low priority.** Exactness is timing-only. Value only for bursts; in the 16-user steady state admissions are mostly isolated.

- **O3, drain on demand at BGEN: keep for later (overlap only).**
  - The saving is the read-start refill (up to ~20 ms) and is real.
  - Missing from the site list:
    - `kv_quiesce` (`cudaDeviceSynchronize`).
    - `ver.wait_commit`.
    - `session_zero`.
    - The stage-0 Verifier object, which *is* the batch verifier (`stage_verifier(0) == ver`).
  - Only after the overlap x is measured. Overlap is not in production.

- **K5/O6, checkpoint without drain: drop.** It saves ~2.5 ms after K2 (or K2-lite) for 1 d plus a helper thread on CUDA, an event-error rewrite of #613, and a postponed final drain. Not worth it.

- **O7, merged window as a pipeline job: fix by merging it with O12.**
  - On its own in non-overlap mode it saves nothing: `copy_to_slot` → `direct_slot_copy` → `sync_all` (`cudaDeviceSynchronize` on every device) drains the groups that were just restarted, so you pay a second refill (~20 ms), more than the ~10 ms gained.
  - It also shares one Verifier object (scratch, `h_flag_`, `h_seq_`, `last_t_`, `h_out_`, `head_logits_`, plan buffers) between the solo job and the batch groups on every stage. The 4–6 d estimate is low.
  - Do it only as one feature with O5/O8 (event-gated copy), and only after phase 1/2 are measured.

- **O5 + O8, continuous admission: keep as part of the merged O7+O12 feature, phase 3, overlap only.**
  - The pad-row analysis (hold `g_a` while `admit_slot` is below its window width) matches pump (9670-9675).
  - `set_slot_sampling` must stay on the pump thread (`sample_rows` reads `slot_sp_`).

- **Parked list (C, D, K3, K4, K6, K7, O9): agree.**

## New candidates (measure in step 0, not ranked yet)

- **N1, D2D `copy_from_slot` for a checkpoint source.** Copy the KV D2D up to `at` and restore only the running state from the host checkpoint. This removes the host KV round trip on next turns that resume from a slot's turn checkpoint. Today's log has 0 such cases, so it is only worth doing for thinking-mode multi-turn traffic.
- **N2, skip the drafter K/V work for BGEN admissions when `!batch_mtp`** (production does not set it). Slots never draft. The work is the header `mtp.prefill` and E-9 per chunk (`sp.on_chunk`). It is lossless for tokens. Its cost needs measuring with `STRATA_TRACE`.

## Corrected ranking

Columns:
- **Step**: what is built, with its env switch.
- **Δ ms**: per-admission saving with no overlap, against m0 (the 142 baseline is about +1 ms).
- **Days**: effort.
- **Exact**: whether outputs stay bit-identical, and under what condition.

| # | Step | Δ ms | Days | Exact |
|---|---|---|---|---|
| 0 | Instrument, GLIBC_TUNABLES arm, server-level R fit, overlap x, frequency counts | 0 | 0.75 | n/a |
| 1 | **AB**: one merged last window (B on A's T ≤ 8 infrastructure; generalised gate; fixes for spec_q, draft counters, emit and PP; resident-experts gate) | ~22 | 2–2.5 | yes, if all experts are in VRAM (else `IQ_MT_MIN=1`) |
| 2 | **K1 + K2-lite**: persistent per-stage checkpoint threads, no zero-fill | ~11–13 | 1–1.5 | yes |
| 3 | **S1 + duplicate fix**: move the checkpoint; allow `slot_ck` from an active slot, or a shared payload | ~6.5 | 0.75–1 | yes |
| 4 | **S2**: async on the private stream, event-ordered residency refresh | 0 vs m0 / ~1 vs 142 | 0.5 | yes |
| 5 | **O1**: engine ignores BYIELD when piped, no drain | conditional, ~20 per BYIELD | 0.25 | yes |
| 6 | **O10**: linger | bursts only | 0.5 | yes (timing) |
| 7 | K2 pinned, only if K2-lite is short of ~3 ms | ≤2 | 1.5+ | yes |
| 8 | O2 + O3, overlap only, after x is measured | ~10–20 (ovl) | 1.5–2.5 | yes |
| 9 | O7+O12 as one feature, overlap only | ~20–30 (ovl) | 8–10 | yes, high risk |
| – | K5/O6 | drop | – | – |

Totals for steps 0–4:
- About 40 ms in about 5–6 days, against the plan's 43.5 ms in about 7–7.5 days for phase 1 plus K2.
- Model with no overlap: 285 → 245 short / 698 → 658 long, i.e. **460 / 336 tok/s (+3.6% / +2.6%)**.
- With overlap at x = 0.7: about 527 / 466 tok/s.
- Steps 5–6 add on top only under BYIELD traffic or bursts.