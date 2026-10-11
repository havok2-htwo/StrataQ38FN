# Design: batched admissions (several queued prompts per read)

2026-10-12. Measurement, two code readings, a staged design and an adversarial critique with its own GPU checks. **The critique's verdict at the end supersedes the design**: the real lever it found is CPU pinning (`--aux-cpus`, TESTLOG #52); k > 1 queued admissions is mostly an artifact of the lockstep benchmark (fixed max_tok).

## Measurement: read time vs tokens

I found an exact way to share most of the per-admission cost, and it needs no kernel changes. The shared MoE pass you proposed is not bit-exact with the current kernels, so it can only be added later behind an explicit gate. All of this comes from reading the code at `/media/raid0/StrataQ38FN/Strata-142` (g3-142, c9bbd1fc). Nothing was edited, built or run, and no GPU was used.

## 1. How an admission flows today

**Server (`serve/server.py`)**
- `_take_control` (1182-1225) holds one control lock (`ctl`) from the moment it sends BGEN until it reads BADM. `wait_lens` (765, 1188, 1215) only feeds `_shorter_waiting` (1236) for BYIELD and the `waiting=N` key (1381).
- The BGEN line is sent at 1383-1390 and its output is read by `_control` on the shared `self.lines` queue until BADM (1394-1411). The lock is released at 1414. If the client leaves, the `finally` block reads up to BADM (1485).
- The stdout reader (791-797) sends only `BT`/`BDONE` to the per-slot queues. Everything else goes to the shared `self.lines`.

**Engine (`src/program/generate.cpp`)**
- The input loop (10016-10055) works like this: when no line is waiting it launches the slot windows again (`pump(true)`); when a line arrives it first waits for the in-flight windows to finish (`pipe_drain`). That wait is the ~26 ms "drain". Launching again after BADM is the ~24 ms "refill".
- BGEN is turned into `GEN 1` at 10084-10099 and goes through the normal single-request path, which uses the main sessions.
- In that path, in order:
  - choosing where to resume: the main session's live tokens or checkpoints, a slot holding the same conversation, or a parked conversation (10576-10637)
  - parking the main conversation (10672), copying from a slot (10676-10724), restoring a parked conversation (10730-10800), mounting a checkpoint (10818-10880)
  - cut points at the root, turn, message and pin boundaries (11729), the read loop (11757), the merged first window (11735, 11833)
  - DONE (12907), then the slot copy, checkpoint hand-over and BADM (12915-13005)
- STOP is one global flag for the whole engine (9090) and is cleared when the next request starts (10118).

The key finding: once BADM is printed, if the next BGEN line is already in the engine's input, the loop takes it straight away. Nothing is in flight, so the drain costs nothing and no refill happens. The engine pays drain and refill on every admission only because the server sends the next BGEN after it has read BADM. In that gap of a few ms the engine has already launched the windows again.

## 2. Bit-exactness of reading several prompts in one MoE pass

**Not exact with the current kernels.**
- In `src/prefill/moe_mmq_tab.cuh:395-430`, the number of tiles depends on the largest row count of any expert, and the stream-k path splits the K loop over `nsm` blocks (unless tile efficiency is at least 90%) and adds a fixup pass.
- So where the K loop is split, and the order in which a row's partial sums are added, depend on which other rows share the launch. The tile width J is also chosen from that same row count.
- TESTLOG #35 already found this ("changing MMQ's channel count changes stream-k rounding"). Dense GEMMs whose algorithm depends on the row count are a second risk.

**What the engine guarantees today** is not independence from chunk content. It guarantees that a read replays exactly the same chunks:
- `include/strata/program/batch_read.hpp:1-5`: pieces end on chunk multiples "so they read exactly the chunks a single run would".
- BYIELD parks and resumes "with the same chunks" (9408-9410, 11221-11224, `partial_from0` at 11585, resume at 10693).
- `STRATA_CKPT_REREAD` (10838-10850) checks a checkpoint only against a re-read with the same chunks.
- `ADMIT_MERGE_FIRST` relies on window rows being identical only behind the gate at 11738-11760 (zero-doorbell, all experts resident).

A shared MoE pass would therefore need a mode where each row's result does not depend on its neighbours (no stream-k, fixed J, M-independent dense GEMMs). That mode would apply to solo reads too, so outputs change against today's engine. That is the "explicitly gated" case.

**Exact alternative.** A layer split already pipelines a run's chunks across the stages (`prefill.hpp`, `set_stage`). A single 130-token chunk runs the 4 stages one after another, ~4×33 ms. If one run contains each member's own solo chunks back to back, each with that member's session, every chunk computes exactly what the member's solo read computes. Only the stage pipeline fill is shared:
- k = 3 short prompts: about (4+2)×33 ≈ 200 ms instead of 3×132 = 396 ms.
- 2 × 1.2K prompts: about (4+5)×90 ≈ 810 ms instead of 1082 ms.

## 3. Recommended design, in three tiers

### Tier 1: the server sends queued admissions in one burst (no engine change, exact by construction)

The server sends all ready BGEN lines in one write and reads their outputs in the engine's order. The engine processes them one after another with no drain or refill in between, so only one of each per burst. Each request goes through exactly the same engine path as today; the decoding slots just don't get windows between the members.

Server changes:
- **(a) Admission tickets.** Replace "hold the lock from BGEN to BADM" with a ticket queue `admit_q` (prompt, left, keys, cancel, slot, live, a result queue). The first waiter that gets the lock becomes the leader. It drains `admit_q` up to a cap, picks a slot for each ticket under `slot_cv` (1335-1366 logic), and writes all BGEN lines in one `_send` under the send lock, so no BSTOP or BYIELD can land between members. `waiting=` becomes the number of tickets behind plus `wait_lens`.
- **(b) Route output by ticket in the stdout reader (791-797).** While tickets are outstanding, every line that is not BT, BDONE or fatal goes to the result queue of the oldest outstanding ticket. That ticket is popped at its BADM, or at an ERR with no DONE (refused BGEN at 10091, bad request at 10397-10414). The reader releases the lock when the queue is empty. This avoids a member waiting on a stalled client of another member, which a "take turns reading" scheme would risk.
- **(c) Per-ticket reading.** `_control` and `_drain_control` read the ticket's own queue. `_ctl_mode`, `_ctl_result`, `_yielded`, `_ctl_erred` and `progress` move onto the ticket. `self.last` is already shared and racy with parallel slots; give it a per-ticket copy.
- **(d) No BYIELD** for members of a group of two or more (1300, 1397-1402); short prompts join the group instead.
- **(e) Cancel.** A ticket not yet sent is simply dropped. For a sent member, STOP is unsafe: it can land on the next member, because the engine clears STOP at each request start and the next member is already queued. Either send STOP only when the member is the last in the burst, or let it run to BADM and use the existing BSTOP-and-free path in `finally` (1487-1520). A BSTOP lost by `sl = BSlot{}` at 12930 is sent again there, so it recovers.
- **Effort:** about 200-300 lines of server code, 1-2 days including tests.
- **Expected gain:** about 50 ms per admission when someone is waiting (~90% of admissions per #44). The drain-plus-refill estimate gives about +3% short / +3% long. The bursts concentrate the other slots' stalls, so time-between-tokens p95 rises.

### Tier 2: read the members together into their slots, then finish each one normally (exact)

1. **Group boundary.** The server adds a key `grp=i/k` to each BGEN. Older engines skip unknown keys (10397), so against an old engine this falls back to Tier 1. In the new engine, the main loop collects the k-1 following lines from its input (same write; wait at most ~50 ms) before handling the first member.
2. **Pre-read for each groupable member,** run one member at a time in the main session for setup, then together for the read:
   - Work out the resume point and the first cut using the same code as the normal path. Factor 10576-10637 and 11666-11729 into shared helpers so the two cannot drift apart.
   - Set up the member's starting state in its own slot sessions: zero them if resuming from 0; do nothing if its own slot already holds the conversation; restore the checkpoint into the slot sessions if its own slot holds a matching checkpoint; for a main-session checkpoint or live state, park, mount in main as today, then `copy_to_slot(slot, ids[0:resume])`.
   - Only parts the solo path would read batched (longer than `short_read`) are pre-read.
3. **Shared read.** Add a `Prefill::run_segments` that takes per-member `{tokens, n, pos0, per-stage SessionState*, ple_prev}`. Each member is cut into its own solo chunks (from its own start). The chunks flow through the stage pipeline, each with its own session. This lives in `prefill.cpp`, at the chunk loop from 2154 and the stage hand-off, which belongs to the prefill area. The member's drafter build (`on_chunk`) and periodic checkpoints (`on_stage_chunk`) are skipped or handled per member.
4. **Store the result like a BYIELD.** For each member, save the checkpoint at its cut from the slot sessions into `bs[slot].checks`. Then set `bs[slot] = {ids = [0, cut), cached, partial, partial_from0 = (resume == 0), cvec}`. This is exactly the state BYIELD leaves (11579-11590).
5. **Finish each member through the unchanged normal path, back to back.** It finds its own slot as the source (`partial`, 10693). It copies the slot to main (~0.5 ms with direct copy), reads the tail and the merged first window, copies to the slot and prints BADM. Output stays untagged and in order, so the Tier 1 server works as is.

- **Not grouped** (handled as Tier 1): images, `pin=`, `ckpt=0`, a different `cvec` or `aux_cpus` from the group, a source that is another slot or a parked conversation (v1), a first cut that is the root or message boundary, `kvg.on`.
- **Effort:** engine about 400-500 lines (collecting members ~60, moving the planners into helpers ~150, pre-read ~200), plus prefill about 150-250. About 4-6 days plus the exactness tests: `batch_test16`, `batch_overlap_test`, the interleave test, and a new "group == sequential" A/B that also compares RESUME values and checkpoint sets.
- **Expected gain,** for a group of 3 short prompts: 26 + ~200 + 3×(28 + 7 + 1) + 24 ≈ 355 ms, about 120 ms per admission instead of ~220. The throughput model D/(1+D·R/L) gives about 537 tok/s short (+9%). For long prompts the gain is smaller (~−25% read time for pairs).

### Tier 3: shared MoE chunk (gated)

Tier 2 plus a row-independent MoE/GEMM mode. Short prompts are packed into one chunk of at most `--prefill` tokens; raising it to 1024 allows more members. Estimated ~+1% over Tier 2 for short prompts. Outputs no longer equal the default engine, only the same engine with the gate on. Effort 1-2 weeks, mostly kernel work.

## 4. State that must stay separate per member

| Area | Per member |
|---|---|
| Request | ids, n, max_new, slot, `req_sp` (seed, penalties → `set_slot_sampling` at 12926), `ckpt`, `pin`, `waiting` |
| Shared settings | `cvec`, `aux_cpus`, `pcie_frac`, `spec_min_p` are global while a read runs → they must be equal across a group's members (group key) |
| Cache match | resume point, `from_live`, source slot and its checkpoint, lent slot, parked conversation, `resumed_from0`, `moved_in` (resume points differ per member; each member's chunk grid starts at its own resume) |
| Cut points | `root_at`, `turn_at` (note: computed only for `i > resume`, so the pre-read must save the turn checkpoint itself), `message_at`, `pin_at`, periodic `pp_next_check` |
| Sessions | slot sessions on every stage: K/V cells, GDN/conv state, QSA state, `ple_prev`; drafter K/V (main drafter only, so skip during the pre-read; this only lowers MTP acceptance later) |
| Slot bookkeeping | `bs[b]`: ids, cached, partial, partial_from0, checks, cvec, img, stop flag |
| Server | ticket: slot, live dict, `_ctl_result`, `_yielded`, `_ctl_erred`, progress, last, cancel, slot claim |

Prerequisite for the main-to-slot copies: two members must not use each other's target slot as their source. Exclude the group's target slots as sources except a member's own slot.

## 5. How BADM, ERR and STOP map per member

- **Refused before processing** (`ERR BGEN: no such free slot`, bad request, logit_bias, no vision): ERR for that member only, with no DONE or BADM. The reader drops its ticket and the others continue.
- **Normal:** RESUME / PP / T / DONE, then `BADM <slot> 1|0`, in member order. A member only pre-read in Tier 2 prints nothing until its turn.
- **Slot copy fails during the pre-read:** clear `cached` on that slot. The member then reads from scratch in its normal pass, as the copy-failure branch at 10680 does. The others are unaffected.
- **CUDA or shared read failure:** stays fatal (`return 1` / `_Exit` at 11786-11800). The server raises EngineDied for every ticket.
- **Cancel:**
  - Engine fix: carry `bs[b].stop` across the admission (12930 resets it), so a BSTOP that arrives during the pre-read ends the member with `DONE … cancel` + `BADM b 0`.
  - The other members' results do not change: a cancelled member's chunks are still read in Tier 2, which costs time but keeps exactness.
  - An addressed `STOP <slot>` would also need the stdin reader at 9090 to stop matching only the bare `STOP` line.
- **BYIELD:** ignored for members of a group of two or more.

## 6. Risks

- **Checkpoint sets must match.** If the pre-read leaves a different checkpoint set than the solo path, later requests resume elsewhere and their chunk grid changes, so outputs differ from a solo reference even though each request is correct. The A/B must compare RESUME values and checkpoint lengths, not only tokens.
- **Slot sessions as prefill targets.** The code says they are "carved alike" (9440), but the QSA allocation, K/V capacity and `ple_prev` of slot sessions need checking before `Prefill` writes into them.
- **Time to first token.** The k-th member of a burst waits for k-1 first windows. The other slots stall for the whole burst instead of in ~220 ms pieces.
- **Same output loss as today.** TESTLOG notes after #40 that a next turn continuing from a slot can differ from a fresh solo read. Tier 2 uses that same partial-slot resume, so it inherits the issue: baseline the A/B against sequential admission, not a fresh solo read.

I did not write any files: the notes directory `/media/raid0/StrataQ38FN/linux/results/batchadmit/` does not exist yet. The next step is to measure Tier 1 on its own with `ab_quick.sh` against `T16-142fc`, with `STRATA_TRACE` showing "drain for BGEN: 0 groups in flight" for burst members.
## Design


Status: this is a plan, written 2026-10-12. Nothing in it has been built. It draws on four sources:
- the scaling measurement in `results/batchadmit/scaling/`;
- two read-only code analyses: prefill internals and admission flow;
- spot checks of the line numbers against `/media/raid0/StrataQ38FN/Strata-142` (branch `g3-142`, c9bbd1fc);
- TESTLOG #44–#50 and `upstream/DESIGN-admission-fixed-cost.md`.

As AGENTS.md requires, every stage is opt-in behind its own env switch and is off by default. All line numbers refer to Strata-142.

## 0. Summary and ranking

The ranking differs from the obvious order "shared drain/refill, then shared first window, then batched MoE". Three facts from the data drive this:

1. **The drain is nearly free.** The in-flight groups finish at the normal completion rate (premise P2 of the fixed-cost design, confirmed by its critique). Only the ~24 ms refill is lost, so sharing the drain saves almost nothing.
2. **A shared first window saves only its base cost.** TESTLOG #48 shows each window row costs 2.4–4 ms. Each member still brings its own ~7 tail rows, so only the ~11.5 ms base of the window is shared. The measurement's "+3 ms per extra member" understates the cost.
3. **The "concat" measurement is a merged-expert read (Stage 5), not a pipelined read (Stage 3).** It is still the best data available. Reading the k prompts as separate jobs through the 4-stage pipeline beats it for short prompts and matches it for long ones:
   - 4×150 tokens: about 325 ms pipelined vs 358 ms concat vs 580 ms separate.
   - 2×1.2K tokens: about 928 ms pipelined vs 901 ms concat vs 1058 ms separate.

   The pipelined read is also bit-exact by construction. The merged-expert read is not exact with the current MMQ (stream-k) and cuBLAS kernels.

| Rank | Stage | Exact | Δ stall per admission, short / long (burst k=4) | 16 users tok/s, short / long (model, cumulative) | Implementation (agent-h) | GPU (h) |
|---|---|---|---|---|---|---|
| – | Baseline T16-142fc (measured, #49) | – | R_fit 213 / 608 ms | 493 / 357 | – | – |
| 1 | **Admission hold** `STRATA_ADMIT_HOLD_MS` (engine; small server fix) | yes (timing only) | −16 / −16 | 501 / 361 (+1.5 / +1.2 %) | 1.5–2 | 1.5 |
| 2 | **Burst protocol**: server tickets send all ready BGENs in one write | yes (same engine path) | −5 / −5 (hold misses, server gap) | 503 / 362 (+2.0 / +1.5 %) | 4–6 | 1.5 |
| 3 | **Pipelined group read (P)**: `Prefill::run_jobs`, each member's solo chunks into its own slot sessions | yes (identical launches) | −58 / −97 | **533 / 389 (+8.2 / +9.0 %)** | 10–14 | 5–7 |
| 4 | **Shared first window** over the k slots | yes (ADMIT_MERGE_FIRST gate, Philox per position) | −8.6 / −8.6 | 538 / 392 (+9.2 / +9.8 %) | 4–6 | 2 |
| 5 | Merged-expert chunk (M), exact-by-replay (E1) or gated (E2) | E1 yes / E2 gated | ≈0 (worse than P) / −2…−13 | 538 / 394 | 20–30 | 8–12 |
| H | Side track (not batching): **host-grouping contention** in overlapped stages | yes (expected) | 0 / up to −170 | up to – / 405 alone (+13 %) | 2 + fix | 2 + fix |

Recommendation:
- Build stages 1 → 2 → 3, then take stage 4 if stage 3 lands.
- Diagnose Track H in parallel with stage 1. It is cheap to look at, and it sets the upper end of stage 3's short-prompt gain.
- Do not build stage 5 unless Track H reverses the comparison between P and the concat read.

## 1. Inputs and the throughput model

**Measured numbers per admission** (15 slots decoding, fixed-cost trace `results/fixedcost/final/`):

| Phase | ms |
|---|---|
| drain | ~26 (≈ free in throughput) |
| batched read, ~130 tokens | 132 |
| batched read, ~1.2K tokens | 541 |
| merged first window | ~28 (11.5 base + ~7 rows × 2.4–4) |
| checkpoint | ~7 |
| refill after BADM | ~24 |

**Read-time scaling** (dec15, `scaling/plain.summary.txt`):
- Single-chunk reads: 68 tokens in 100 ms, 143 in 145, 293 in 230, 443 in 298.
- Multi-chunk steady state: 0.335 ms per token, about 171 ms per 512-token chunk.
- Pipeline fill F ≈ 130 ms (for 1193 tokens: 529 = F + 0.335 × 1193).
- Joining k prompts reads exactly as fast as one prompt of k·N tokens (within 1 %).

**Small-chunk pipelining** (#50: `--prefill 128` / `64`, ~173-token prompt):
- 202–228 ms with 2 chunks and 234–259 ms with 3 chunks, vs ~162 ms with 1 chunk.
- So each extra small chunk costs ~40–55 ms in the pipeline, against ~25–36 ms per stage when a stage runs alone.
- That gives an overlap contention factor c ≈ 1.6–1.7, the "host grouping" growth seen in the measurement.
- This is the basis for the stage 3 short-prompt estimate.

**Model:** T = D / (1 + D·R/L).
- D = 620 rows/s and L = 512 (`ab_quick.sh` uses `--max-tok 512`).
- R_fit = 213 ms short and 608 ms long reproduce the measured 493 / 357 exactly.
- With D = 600, R_fit is 185 / 581 ms, and the relative gains below change by less than 0.5 points.
- Savings are subtracted from R_fit.

**Burst assumption:**
- Groups of k = 4 (the cap, `STRATA_ADMIT_GROUP_MAX=4`), so a fraction (k−1)/k of admissions follow a predecessor in the same burst.
- This matches the benchmark regime. `ab_quick` runs 16 users × 2 rounds, round 1 is one burst of 16, and #44 found ~90 % of admissions have a request waiting behind them.
- With desynchronized arrivals (k ≈ 1) stages 1–4 gain close to nothing. Only Track H helps an isolated admission.
- k=2 variant (for example a token cap on long prompts): short 523 (+6.1 %), long 380 (+6.5 %), stages 1–4 cumulative.

**Side effects to report in every A/B:**
- The other slots stall once per group instead of once per admission. A group stall is about 4 × 125 ≈ 500 ms short and 4 × 480 ≈ 1.9 s long, so time-between-tokens p95 rises.
- TTFT: the first member of a group waits about (k−1) × 60 ms longer, while the k-th member waits about (k−1) × 90 ms less. Mean TTFT should drop.
- Add a token budget `STRATA_ADMIT_GROUP_TOKENS` (default 2600) so long prompts group as k ≤ 2.

## 2. Exactness: what the engine guarantees today

- **No per-row invariance in prefill.** A token's bits depend on which other rows share its chunk, through three mechanisms:
  - MMQ expert grouping: `order` is split into groups of `MMQ_GROUP=16` (prefill.cpp:3283, 3369-3376, 3820), and `maxr` (3826) sets the tile width J (moe_mmq_tab.cuh:462).
  - Stream-k: below 90 % tile efficiency (moe_mmq_tab.cuh:424-431; Ampere `stream_k=true`), K-ranges are split over 82 blocks plus a fixup pass (mmq.cuh:1066-1071, 1242ff). TESTLOG #35 already recorded this.
  - cuBLAS `GemmEx(..., CUBLAS_GEMM_DEFAULT)`: the algorithm is chosen by shape, with T as a dimension (gemm.cu:668, 702-713).
- **"Batched read == solo read" holds only by deterministic replay.** Pieces end on chunk multiples "so they read exactly the chunks a single run would" (batch_read.hpp:1-5). BYIELD resumes "with the same chunks" (generate.cpp:11221-11224, 11578-11590, 10690-10696). The chunk loop is `c0 += m.T` from the segment start (prefill.cpp:2158).
- **Windows are row-invariant only under the ADMIT_MERGE_FIRST gate** (generate.cpp:11735-11760): zero-doorbell, all experts resident, `!batch_mtp`, no images, kvg, peer or swap.
- **Consequence:**
  - Stages 1–4 replay each member's solo launches exactly, or change only timing, so they are exact.
  - Stage 5 changes the launches. It is exact only through E1 replay of partial groups, or under an explicit gate (E2) that also changes solo output.
- **Reference for the A/B tests:** compare against *sequential* admission with the same engine, not against a fresh solo read. #40 notes that a turn continuing from a slot can already differ from a fresh read. Compare tokens, RESUME values and checkpoint lengths.

## 3. Stages

### Stage 1: Admission hold (`STRATA_ADMIT_HOLD_MS=N`, default 0)

This is the fixed-cost design's O10 "linger" step. It was low priority for steady state, and is promoted here because the 16-user regime is bursty (#44).

**Why it works.** After BADM, if the next control line is already in `in_lines`, the loop takes it without a pump (generate.cpp:10021-10024) and the drain finds 0 groups in flight (10034-10050). The engine pays the refill only because the server sends the next BGEN after reading BADM. That takes a few ms, and by then `pump(true)` has already restarted the groups.

**Engine changes (generate.cpp):**
- **Remember the hint.** At BADM (`std::printf("BADM ...")`, ~13000), keep `adm_waiting = req_waiting` in loop scope. `req_waiting` is parsed at 10379-10410.
- **Hold before refilling.** In the loop at 10021-10024: if `cont && adm_waiting > 0 && hold_ms > 0`, wait up to N ms for a line before calling `pump(true)`. Use a timed variant of `try_next_line` (9432), a `wait_for` on the input condition variable under `in_mu`.
  - If the line arrives, process it with no pump. The drain is then a no-op.
  - On timeout, pump as today.
  - A BSTOP that arrives during the hold is applied by the stdin reader (~10005-10017) as today.
- **Trace.** With `STRATA_TRACE`, log "admission hold: next line after X ms" or "hold timed out".

**Server changes (`serve/server.py`):**
- `waiting=` at 1381-1382 counts every queued request (`len(self.wait_lens)`). Send only the number that can be admitted now: min(waiters, free slots other than this one), excluding waiters held back by `after_epoch` (1197). Otherwise a hold waits for a BGEN that cannot come.
- Release `self.ctl` as soon as `_control` returns at BADM, when the loop will `break` and is not continuing after a BYIELD. Today the remaining `pending` tokens are yielded to the HTTP consumer first (1404-1408), and the lock is released only at 1414, so a slow consumer delays the next BGEN.

**Gain:**
- Saves the refill, ~24 ms (about 3 stage-times), for each admission that follows a predecessor: about 16 ms on average at k=4, minus a hold of a few ms.
- 493 → 501 short, 357 → 361 long.
- A miss costs N ms of idle slots on top of the normal refill. Default N = 10, which exceeds the measured BADM→BGEN gap; confirm the gap with the trace.

**Exactness:** timing only. Window contents per slot do not depend on when groups start. The existing tests cover this invariant: `batch_overlap_test` (tokens + RESUME) and the interleave test.

**Tests:**
- `batch_test16`, `batch_overlap_test`, interleave, and the 45 driver requests from #48.
- `scale_driver.py` burst arms: next BGEN 0, 5 and 30 ms after BADM, covering both the hit and the miss path.
- `ab_quick.sh` with T16-142fc vs a new config `T16-142fc-hold`.
- In the trace, check that "drain for BGEN: 0 groups in flight" appears for burst members.

**Effort:** 1.5–2 agent-h. GPU about 1.5 h (exactness suite ~40 min; ab_quick ~10 min per config; trace run).

### Stage 2: Burst protocol (server ticket queue)

This replaces "hold `ctl` from BGEN to BADM" with tickets. Stage 2 alone needs no engine change, and stages 3 and 4 depend on it: the engine must see a group's BGEN lines before it prints the first member's BADM. With hold alone, that order would deadlock.

**Changes (`serve/server.py`):**
- **(a) Tickets.** `_take_control` (1182) enqueues a ticket: prompt, left, keys, cancel, slot, live, result queue. The waiter that gets `ctl` (767) becomes the leader. It drains `admit_q` up to `STRATA_ADMIT_GROUP_MAX` and the token budget, picks a slot per ticket under `slot_cv` (logic of 1335-1366), and writes all BGEN lines in one `_send` (1388), so no BSTOP or BYIELD can land between them.
  - Add `grp=i/k` to each line. Older engines skip unknown keys (10397-10410), so this is backward compatible.
  - `waiting=` becomes the number of tickets behind plus `wait_lens`.
- **(b) Route output by ticket.** In the stdout reader (791-797), every line other than BT/BDONE goes to the oldest outstanding ticket. That ticket is popped at its BADM, or at an ERR with no DONE (a refused BGEN at generate.cpp:10091; a bad request at 10397-10414).
- **(c) Per-ticket state.** `_control` (1034) and `_drain_control` (1094) read the ticket's queue. `_ctl_mode`, `_ctl_result`, `_yielded`, `_ctl_erred`, `progress` and `last` move onto the ticket.
- **(d) No BYIELD** for members of a group of two or more (1300, 1397-1402).
- **(e) Cancel.**
  - A ticket that has not been sent is dropped.
  - A sent member must never get a bare `STOP`. STOP is one global flag (generate.cpp:9090), cleared at the start of each request (10118), so it would hit the next member. Let the member run to BADM and use the existing BSTOP-and-free path in `finally` (1485-1520).
  - Engine fix (one line, needed by stage 3): keep `bs[b].stop` across the admission (`sl = BSlot{}` at ~12930 drops it today).

**Gain:** removes hold misses and the server's BADM→BGEN gap (the fixed-cost critique found a ~10–40 ms residual at server level). Model: about 5 ms per admission, 501 → 503 short. The real value is as the enabler for stages 3 and 4.

**Exactness:** each request takes the unchanged engine path, in the same order.

**Tests:**
- Server unit tests against a fake engine (no GPU): ERR mid-burst, cancel of a sent member and of an unsent one, engine death during a burst (EngineDied for every ticket), BYIELD suppression.
- On the GPU: the exactness suite, then `ab_quick` in a short and a `--rounds 4` variant.

**Effort:** 4–6 agent-h. GPU 1.5 h.

### Stage 3: Pipelined group read (P), `STRATA_ADMIT_GROUP=1` (+ `_MAX`, `_TOKENS`)

**Idea.** On a layer split, a stage already hands chunk c to the next stage and goes on with chunk c+1 (prefill.cpp:4095-4101, `hand[2]`). A one-chunk prompt leaves three of the four cards idle at any time (~132 ≈ 4 × 33 ms). Stage 3 feeds the k members' solo chunks back to back through the pipeline. Each chunk is bound to its own member's slot session on that stage, so every chunk runs exactly the launches the member's solo read would run.

**Changes, prefill (`include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp`):**
- **API.** Add `struct Job { core::SessionState* ss; const int64_t* tokens; int64_t n, pos0, chunk; ple_prev; int id; }` and `bool run_jobs(const Job*, size_t, std::string&)` next to `run` (hpp:135). `on_chunk` (147) and `on_stage_chunk` (153) gain a job id, or get job-aware twins.
- **Per-chunk session.** `Impl::ss` is set once in `init` (prefill.cpp:706, 1047); bind it per chunk instead. Every `ss.` user then follows the binding: `gdn_state`/conv (2636-2647), QSA rope/`kv_append`/indexer/`qsa_prompt_attn_batch` (2670-2868), PLE history (2538).
- **Job loop.** The chunk loop `for (c0 = 0; c0 < n; c0 += m.T)` (2158) becomes an outer job loop. Each job steps by its own solo chunk: the `C = sp.chunk()` that `read_part` uses (generate.cpp:~11545), and `request_chunk`/`equal_chunk` (generate.cpp:6435).
- **PLE prefetch.** The prefetch future (2141-2156) and `ple_prev` (2109, 4157) become per job. The prefetch of "the next chunk" may belong to the next job.
- **Stage hand-off.** The hand-off lambda (4097-4099) captures the job and binds `next_`'s session before `run_impl`. `drain_pipeline` (4244) runs once at the end.
- **Geometry check at init (refuse, then fall back to sequential).** Slot sessions per stage must match the main session's `max_cells`/`max_blocks` (1215), `gdn_ord0` (2636) and QSA pools. They are carved alike (generate.cpp:9440), but this is unchecked.

**Changes, engine (`src/program/generate.cpp`):**
- **Collect the group.** After the BGEN parse (10084-10099), if `grp=0/k` is set, take the k−1 following BGEN lines from `in_lines` under `in_mu` (pattern at 11522-11524). Wait at most 50 ms for a missing member, then fall back.
- **Shared planners.** Factor the resume planner (10576-10637) and the cut planner (11666-11729, `cuts` at 11729) into helpers that both the solo path and the group path call, so they cannot drift apart.
- **Eligibility.** A member falls back to sequential (Stage 2 behaviour) if any of these holds:
  - images, `pin=`, `ckpt=0`, or `kvg.on`;
  - a different `cvec`/`aux_cpus`/`pcie_frac`/`spec_min_p` from the group (these are global while a read runs);
  - its source is another slot or a parked conversation;
  - its first segment is a root/message cut, or is ≤ `short_read` (the window path).

  The group's target slots are excluded as sources, except a member's own slot.
- **Pre-read.** For each member, set up its starting state in its own slot sessions:
  - zero them for resume 0;
  - do nothing if the slot already holds the conversation;
  - otherwise park, mount the checkpoint or live state in main as today, then `copy_to_slot(slot, ids[0:resume])`.

  Then call `run_jobs` over every member's first segment [resume_i, first_cut_i). The drafter's `on_chunk` work is skipped: slots never draft (`!batch_mtp`), so this is lossless. Save each member's cut checkpoints with the same save call the solo path uses. Because `turn_at` is computed only for i > resume, the pre-read must save the turn checkpoint itself.
- **Hand over like a BYIELD.** Leave each slot exactly as a BYIELD does (11578-11590): `ids = [0, cut)`, `cached`, `partial`, `partial_from0`, `cvec`, `checks`. Also store `preread_at = cut`.
- **Finish each member back to back on the unchanged path.**
  - The partial-resume branch takes its own slot (10690-10696).
  - Copy slot → main (~0.5 ms, direct copy).
  - Read the rest and the merged first window.
  - DONE, then BADM (12907-13005).
  - **Assert `resume == preread_at`.** A mismatch is legal output but breaks the equality with sequential admission: count it in the trace and fail the self-test.
- **Failure handling.**
  - A failed copy into a slot clears `cached`, and that member reads from scratch, as in the branch at 10680.
  - A CUDA error stays fatal, as at 11786-11800.

**Gain** (trace basis 132 / 541 ms):
- **Short.** Group read ≈ 132 + (k−1) × 33 × c, with c ≈ 1.65 from #50. That is 295 ms at k=4 (vs 528 separate), or 74 ms per admission, saving 58. At k=2 it saves 39.
- **Long.** Group read ≈ F + k·S with F ≈ 130 and S ≈ 411: 1774 ms at k=4, i.e. 443.5 per admission, saving 97. This agrees with the concat measurement (0.811 × 541 = 439). At k=2 it saves 65.
- **Cumulative:** 533 short (+8.2 %), 389 long (+9.0 %).
- If Track H brings c to about 1, short rises to ~544.

**Exactness:**
- Every chunk of member i sees the same tokens, T, `p0`, session state, stage and kernels as its solo read. Stream-k geometry, J, MMQ grouping and cuBLAS shapes depend only on that chunk's own rows, so the bits are identical.
- Concurrency puts different members on different GPUs (one stage per card), and one stage's chunks are serialized on that stage's stream.
- Host-side grouping is deterministic (sorted by expert id).
- All experts are resident in T16, so no loan or streaming order is involved. Keep the loan sized for the largest job for other configs.
- Slot → main copies and checkpoint saves are byte copies.

**Tests:**
1. **`STRATA_ADMIT_GROUP_SELFTEST=1`.** After the group pre-read, re-read each member solo into main and `memcmp` every stage's state: K/V cells up to `cut`, GDN/conv, QSA and indexer, `ple_prev`. Any difference is an ERR.
2. **Group == sequential A/B** (driver over stdin, greedy and seeded). Compare tokens, RESUME values and checkpoint length sets per member, grouping on vs off. Mix fresh prompts, multi-turn resumes (`bench_multiturn.py`) and mixed lengths.
3. Existing suites: `batch_test16`, `batch_test32`, `batch_overlap_test`, interleave, and `prefill_mmq_tab_test` (sanity).
4. Cancel inside a group: BSTOP during the pre-read ends with `DONE … cancel` + `BADM b 0`, and the other members are unchanged.
5. `ab_quick` with GROUP_MAX 2 / 4 / 8 and token budget 2600, plus `--rounds 4`. Report tok/s, TTFT p50/p95, and time between tokens p95.

**Effort:** 10–14 agent-h: prefill 4–5, engine 5–7, review and fixes 2. GPU 5–7 h, mostly the A/B matrix and multi-turn runs.

### Stage 4: Shared first window

**Change.**
- After the stage 3 group read, split the per-member finishing pass at the first window (generate.cpp:11830-11845).
- Run every member's tail read up to `merge_at`.
- Run one batch window over the k slots: the verifier's `run_slots`/`run_slot_rows`/`commit_slot_prefixes` (verify.hpp:152-177), merge_T ≤ `kVerifyMaxT` rows per slot.
- Then print each member's T/DONE/BADM in order. The protocol is unchanged.
- Each member must pass `merge_from` (11741-11760); otherwise the group falls back to per-member windows.
- Sampling uses the member's `req_sp` per slot (`set_slot_sampling`, ~12926) before the window.

**Gain:** the 11.5 ms window base is shared: 8.6 ms per admission at k=4. Cumulative 538 short (+9.2 %), 392 long (+9.8 %).

A variant that needs the ride-along protocol saves the base even at k=1: the first window rides in the next decode window, and the first token arrives as BT with `BADM b 2`. It needs per-slot drafter K/V and protocol work, so defer it.

**Exactness:**
- Rows are identical in any window under the ADMIT_MERGE_FIRST gate; that is the gate's premise, and #48 tested it.
- The first-token draw uses Philox per position, as in step B of the fixed-cost design, so the slot sampler and the main sampler draw the same value.

**Tests:** the stage 3 set, plus seeded sampling (temperature > 0) with first-token equality against sequential admission.

**Effort:** 4–6 agent-h. GPU 2 h.

### Stage 5: Merged-expert chunk (M), not recommended

**What it is.** Per-sequence HC/GDN/QSA/router/shared work, then one host grouping and MMQ over all members' rows, then per-sequence write-back.

**Data.**
- **Short:** the k=4 concat read of 358 ms is *slower* than P (about 295–325 ms). Packing four 150-token prompts into 512 + 81 still walks the 4 stages almost serially, while P overlaps them.
- **Long:** 2–13 ms per admission over P, from packing partial chunks.

**Exactness cost:**
- **E1 (exact).** Run full 16-expert groups in merged launches with a new forced one-block-per-tile flag in `launch_mul_mat_q_tab`. Replay the partial groups that ran stream-k solo with their solo geometry (up to ~15 experts per sequence per layer). Per-sequence cuBLAS calls with each sequence's own T, using a convert-once FP16 weight.
- **E2 (gated).** `STRATA_MMQ_TILES=1` forces one block per tile everywhere. Solo output changes at rounding level, which needs a quality check.

Either way, J-invariance must first be proven by extending `tests/cuda/prefill_mmq_tab_test.cu`: the same rows with different co-rows must match bit for bit.

**Effort:** 20–30 agent-h. GPU 8–12 h.

**Revisit only if** Track H shows that P stays host-bound after a fix while M's single host grouping per layer would not be.

### Track H (parallel, not batching): host-grouping contention

**Observation.**
- In multi-chunk reads, each card's time per 512-token chunk grows from ~89 ms when it runs alone to 145–165 ms when 3–4 stages overlap.
- All of the growth is in the "host grouping" phase: 5 → 60–68 ms per stage-chunk.
- That phase is the GPU idle gap while the host syncs and groups experts at each MoE layer (`PfTimer` semantics, prefill.cpp:1940-1948; host syncs such as 2930 and 3132; grouping tables in mapped pinned memory, 782).
- #50 shows the same effect for small chunks: c ≈ 1.65.

**Suspects:**
- CPU contention between the 4 stage threads, the PLE gather threads and `--aux-cpus` pinning;
- host reads of mapped pinned grouping tables across PCIe;
- the sync spin/yield policy;
- memory bandwidth taken by `--ple-io ram`.

**Diagnosis (1–2 agent-h, 1–2 GPU-h):**
- `STRATA_PREFILL_TIMING` on a 2.4K read with 2 vs 4 stages;
- `--aux-cpus` and `taskset` variants;
- an nsys profile of one overlapped chunk.

**Potential.** If the overlap per-chunk time returns to ~95–100 ms:
- a 1.2K read drops from 541 to ~370 ms, and long reaches ~405 tok/s (+13 %) on its own, with no batching;
- stage 3's short-prompt gain moves to its upper end (c → 1).

## 4. Order of work and GPU budget

1. Stage 1 implementation, with the Track H diagnosis in parallel: about 3 agent-h and 3 GPU-h. Decide what to do about H.
2. Stage 2, server only, with the fake-engine tests: about 5 agent-h and 1.5 GPU-h.
3. Stage 3: prefill `run_jobs` first, then the self-test, then the engine group path: about 12 agent-h and 6 GPU-h. This is where most of the gain is.
4. Stage 4: about 5 agent-h and 2 GPU-h.
5. Stage 5 only if H and P data justify it.

GPU time is the bottleneck: ~10 min per ab_quick config, ~40 min per exactness suite. Each step needs a TESTLOG row (#51 onward), and each A/B reports tok/s, TTFT p50/p95 and time-between-tokens p95, plus the trace counters "drain … 0 groups in flight", hold hit/miss, `resume == preread_at` and group-size histogram.

## 5. Risks and open questions

- **Checkpoint sets must match** sequential admission. Otherwise later turns resume at different points and their chunk grids differ. The A/B compares RESUME and checkpoint lengths, not only tokens.
- **Slot sessions as prefill targets.** QSA allocation, K/V capacity and `ple_prev` of slot sessions are unchecked; the stage 3 init check refuses and falls back.
- **Group stalls grow time-between-tokens p95.** Keep the cap and the token budget. #44 showed queue rules can make the median worse, so judge each stage on p50/p95 too, not only tok/s.
- **The model assumes burst k = 4.** The two-round `ab_quick` overweights bursts. Confirm with `--rounds 4`; in desynchronized traffic the gains of stages 1–4 approach 0.
- **The c ≈ 1.65 short-prompt estimate is inferred** from #50 (128/64-token chunks), not measured for 150-token jobs. The stage 3 self-test run should log per-group read time to confirm it.
## Critique and verdict


I checked the design against the code and ran three GPU tests, with no engine edits. The cheapest win is not batching at all: it is a config flag that fixes the slowdown the design calls Track H. With that flag, the 16-user long-prompt A/B rose by about 5–6 %. Stage 4 cannot be built as designed. Stage 3's gain only appears in the benchmark's lockstep bursts, and only once the flag is on. The GPUs were back under 300 MiB at the end.

### What the measurements show

**1. k = 4 is an artifact of the benchmark.** I put a timestamp proxy between the server and the engine (`qprobe/`) and logged `waiting=` on every BGEN:
- Speed through the proxy matched baseline (short 491.6 / long 360.0 tok/s).
- After round 1 the queue held at 3 waiting for most admissions (34 of 64 BGENs with 4 rounds).
- The cause is lockstep decoding. Every request is capped at 512 tokens and batch windows give each slot one token per window. So the 4 slots of a decode group finish (BDONE) in the same window and their users come back together.
- With mixed output lengths or EOS, arrivals spread out. My queue simulation then gives k ≈ 1 after the first burst. Stages 1–3 gain about nothing in that kind of traffic.
- The k=4 assumption holds only for `ab_quick`, and `--rounds 4` does not change that. The A/B needs an arm with mixed `max_tok` values.

**2. The server leaves almost no gap between admissions.** When a request is waiting, the next BGEN arrives 0.2–1.0 ms after BADM, not the 10–40 ms the design assumes.
- The real per-follower loss is different from the design's "refill". The engine restarts one decode group before the BGEN lands, then drains it: 18–22 ms in 44 of 46 admissions, for only 4 rows.

**3. Track H is CPU pinning, and a config flag fixes it** (`pipecheck/`, `auxab/`).
- The host thread is pinned to CPU 0. Threads it creates inherit CPU 0 while `--aux-cpus` is off, which is the default. That includes each prefill stage's `std::async` thread and the PLE gather. The engine had 19 threads limited to CPU 0.
- With `--prefill 144`, which stands in for Stage 3's k one-chunk jobs, each extra pipelined chunk costs about 110 ms (c ≈ 3, not 1.65). The design's estimate for Stage 3 is wrong without the fix.
- With `--aux-cpus 64-127` (no code change), the stage threads move off CPU 0:

| Read (ms) | Without `--aux-cpus` | With `--aux-cpus 64-127` |
|---|---|---|
| 4 chunks of 144 tokens | 486 | 272 |
| 8 chunks of 144 tokens | 953 | 432 |
| 1.2K-token prompt | 529 | 457 |
| 2.4K-token prompt | 911 | 673 |

- Through the server (16 users, 2 rounds, greedy):
  - Short: 495.3 tok/s vs 491.6–493.1, so unchanged.
  - Long: **377.6 tok/s vs 357.3–360.0, about +5–6 %**.
  - Long-prompt time to first token, p50 / p95: 2.65 / 7.98 s vs 3.01–3.02 / 9.06–9.08 s.
- I have not yet checked that this flag leaves output unchanged; the code comments say it should.

### Verdict per stage

| Stage | Verdict | Why |
|---|---|---|
| **H**: `--aux-cpus 64-127` | **Keep, do first** | Run `batch_test16` and the driver equality check, then make it part of a new default config. Gives +5–6 % long with no batching, so it also helps desynchronized traffic. Upstream fix: pin each stage or helper thread off the host CPU even when the feature is off. |
| 1: admission hold | **Keep, with corrected numbers** | The miss case is rare (gap under 1 ms), so a 2–10 ms hold works. It saves about 15 ms per follower: roughly +1 % short, +0.4 % long, and only in bursts. The other changes (sending only admittable waiters, releasing `ctl` at BADM) are still correct. |
| 2: ticket protocol | **Keep only as the enabler for Stage 3** | Its own gain is about 0, since there is no gap to remove. Cancel handling is fine as designed: no bare STOP to a sent member, keep `bs[b].stop` across admission, ERR pops the ticket. |
| 3: pipelined group read | **Build only after H, and only if bursty traffic matters** | See the Stage 3 section below. |
| 4: shared first window | **Drop** | `kVerifyMaxT = 8` is the limit on rows in one window, for all slots together (`verify_kernels.hpp:21`; batch windows hold at most 8 slots). The merged first window already uses 7–8 rows for a single member, so k members cannot share one window. The only variant left is the deferred ride-along. |
| 5: merged-expert read | **Drop** | With H on, pipelining (272 ms for 4×144) beats the joined-prompt read (351 ms for 600 tokens), and it is exact without extra work. |

### Stage 3: what holds and what to fix

**Gain with H on:**
- Short, k=4: about 77 ms of read saved per admission (272 ms grouped vs 580 ms separate).
- Long, k≤2 (a 2.4K read used as the proxy): about 120 ms saved per admission.
- Model estimate in the burst benchmark: about +9 % short and +10 % long over the H baseline.
- Without H, the short gain shrinks to about 23 ms per admission.

**Exactness holds.** Each chunk is launched alone with its own token count and position, so stream-k, MMQ grouping and cuBLAS shapes match the solo read. The pipeline hand-off is one chunk per stage per thread, and the `single_chunk_` flag only matters for the helper mode, which is off.

**Gaps in the design:**
- `checkpoint_at` saves only the main session's state (`ss` / `stages[k]->ss`). Stage 3 needs a version that saves from the slot sessions, plus a defined handoff with `STRATA_SLOT_CKPT_MOVE` (the turn checkpoint moving between main checkpoints and the slot's checkpoints).
- Skipping the drafter's per-chunk work leaves the main drafter's K/V stale below the cut. Tokens are unaffected; a later solo continuation would draft less well.
- Mixing a long and a short prompt in one group makes the short one wait for the long read. Group only similar lengths or order shortest first; the `GROUP_TOKENS` cap alone is not enough.
- Time to first token for the first member rises by about (k−1) × 40 ms for short prompts and about 215 ms for long prompts at k=2.

### Corrected ranking and recommendation

- **Go:** H first (exactness check, then the default config and a TESTLOG row), then Stage 1.
- **Conditional go for Stage 2+3** (about 15–20 agent-hours): only after H is on, and only if an `ab_quick` arm with mixed `max_tok` (for example 256–768) still shows k > 1 often enough to justify it.
- **No-go:** Stages 4 and 5.

I did not add the TESTLOG rows (#51–#53); they are still owed under the "document every experiment" rule.

Files are in `/media/raid0/StrataQ38FN/linux/results/batchadmit/`:
- `review-notes.txt`
- `qprobe/` (`strata-wrap`, `proto.log`, `qanalyze.py`, `engine.log`, `run.sh`)
- `pipecheck/` (`pipe_driver.py`, `pf144/`, `pf144-aux/`, `pf512-aux/`, `run.sh`)
- `auxab/` (`T16-142fc-aux.json`, `short.txt`, `long.txt`, `sweep.jsonl`, `engine.log`)