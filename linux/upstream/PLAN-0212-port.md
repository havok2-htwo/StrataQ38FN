# Port plan: production (engine 3af0006d + server 22f4781f) onto upstream v0.2.12

Inputs: the 7-chunk commit classification, upstream reports A and B, and `/media/raid0/StrataQ38FN/linux/configs/production.json` (read again for this plan). I changed nothing and ran nothing on the GPUs. The only thing I checked live was the worktree list: `/media/raid0/StrataQ38FN/Strata-212` already exists (branch g3-212 @146df3c4, clean, no build).

## 0. Strategy in one paragraph

**Do not run a 3-way merge of our branches into v0.2.12.** Rebuild our production on top of v0.2.12 instead, by feature group, using cherry-picks with `-x` and resolving each conflict by hand.

Why:
- About half of the 267 trial-merge conflicts come from upstream's reworded port of our own 0.1.43 series (v0.1.45 alone gives 127 hunks).
- Those commits are already in v0.2.12, so we skip them and take upstream's code.
- What is left are about 22 production switches of ours that upstream does not have. These are ported as about 11 patch groups onto a fresh branch.

Because the work is grouped by switch, every group stays a self-contained, reviewable unit we can later submit upstream.

The hard groups (pipe overlap, spec lanes, fair reads, the server admission rework) will take several days. **Tonight's window should therefore decide where porting is worth it.** It measures:
- upstream's alternatives on our box (PIPED_CONCURRENT, PIPED_MTP, PW_WITH_BATCH, sm_86 defaults, IQ4_W4A8, DAILY), and
- a partially ported v0.2.12 build,

against production. It should not try to bring up a fully ported build. Production returns to ll10b after the window either way. A deploy needs your "deploy ja".

---

## 1. Feature table

Status values: **U=** upstream has it the same. **U~** upstream has an equivalent with a small delta. **ALT** upstream has a different mechanism. **NP** not in upstream, needs a port. Prod: a value means the switch is set in production.json or the systemd unit.

| # | Feature / switch | Status vs v0.2.12 | Prod | Decision | Commits to cherry-pick (in order) | Difficulty |
|---|---|---|---|---|---|---|
| F1 | >8 slots on a split (`--parallel 16`), 8 rows/group window | U= (ee2a907e, 6ab4c312) | yes | take upstream | none (`tools/batch_test16/32.py` → `tools/batch_test.py`) | – |
| F2 | SLOT_COPY_DIRECT, SLOT_COPY_BATCHSYNC | U= (now PARALLEL_STACK_ENV default) | 1 / 1 | take upstream, keep explicit env | none | – |
| F3 | MMQ_TAB build option + MMQ_DIRECT + slot guard | U~ (7d9b50dc dropped our NVFP4 `if (f.tail_off) continue;`) | 1 | take upstream; re-add the 1-line NVFP4 guard (prefill.cpp ~1594) | delta of 7cdd921f (+ the clause from fd243de9 in docs) | trivial |
| F4 | MMQ_GROUP_SORT | U= (moved to mmq_resident_sort.hpp) | 1 | take upstream | none | – |
| F5 | ADMIT_MERGE_FIRST, CKPT_POOL, CKPT_THREADS, SLOT_CKPT_MOVE, BATCH_SELF_COMMIT, QSA_SLOTS, HANDOFF_PIPE=8 | U= | all set | take upstream; CKPT_POOL/SLOT_CKPT_MOVE/SELF_COMMIT/QSA_SLOTS are now defaults; the 0.1.45 "SLOT_CKPT_MOVE do not use" note is obsolete | none | – |
| F6 | PIPE_BYIELD_NODRAIN | U~ (no `pipe_overlap_env()` in the condition) | 1 | take upstream; re-add `pipe_overlap_env() \|\|` with F15 | delta from 89dcda9e / d8e371d5 | trivial |
| F7 | GR_NORM_FAST on CUDA | U~ (fec7d1ca; not in the sm_86 table) | 1 | take upstream; optionally carry the ctest gr_parity_norm_fast | (CMake hunk of bfca9a6a) | trivial |
| F8 | HC_Q8 trimmed split | U~ (abf2953b) | no | take upstream | none | – |
| F9 | SERVE_STOP_BOUND (#4677) | U~ (243fdb21, but wrapped in `#if STRATA_USE_HIP` and missing `produced_n < max_new`) | default on | take upstream; delta: remove the 3 HIP gates and add the guard | delta of f81292e1 | trivial |
| F10 | park segment reserve (53f72571) | U= | yes | take upstream | none | – |
| F11 | security / parser / robustness ports (8181522f, c08fd2ec, 3f6960f1, fc27296d+b145a4a2, 77794e34+b72f56e0, e69cbfa5) | U= | yes | take upstream; drop our duplicate tests | none | – |
| F12 | STREAM_WRITE_TIMEOUT ProgressWriter (#4271 part 2) | NP (upstream still has the sendall bug) | yes | port ours | 10015835 | trivial |
| F13 | ADMIT_SHORT_FIRST=8192 | U= (ed4fa42a) | yes | take upstream; re-add the `counted` image-gate condition in F21 | none | – |
| F14 | low-level levers: QSA_SLOTS_STREAM=1 | NP | 1 | port ours, then **A/B against ATTN_DAILY tonight** | 038994f4, aeb0cc4f | trivial (but useless while DAILY=1) |
| F14b | ATTN_VBATCH=1 | NP; ALT = ATTN_DAILY (sm_86 default, which also turns QSA_SLOTS off) | 1 | **A/B tonight**: DAILY vs VBATCH+QSA_SLOTS(_STREAM); port only if VBATCH wins | 31af7dff, 0d785e9d, 70b4488e | hard (16 hunks in qsa_decode_attn.cu) |
| F14c | GR_DOWN_PF=2 (+ GR_FAST=1, MMVF_ROWS=1 already upstream) | NP; ALT GR_V3 (not bitwise, sm_120 only) | 2 | port the pre-split form | b628d059, 6d513178 + 096efada (squash) | moderate |
| F14d | EARLY_SAMPLE=1 | NP | 1 | port, rebased on `slot_row_draw`; **measure on its own with T>0** (never measured) | 2ac15656, 389ef5c8, 67ec083d | moderate |
| F14e | GR_DOWN_SPLIT, GR_UP_SPLIT, ROUTE_PLAN_FUSE | NP | no | **drop** (TESTLOG #78) | – | – |
| F14f | SPLIT_OWN_RESERVE_MIB (enables the 11,24,37 rebalance, +4.9/+3.6 %) | NP, merges cleanly | no | carry, re-test the rebalance on the new build | 9a842828 | trivial |
| F15 | **PIPE_OVERLAP + _KVS** | NP; ALT #2326 PIPED_CONCURRENT + #2050 PIECE (off with SPLIT_MTP_BATCH; TESTLOG #90 never engaged) | 1 / 1 | **A/B tonight**: upstream PIPED_CONCURRENT with SPLIT_MTP_BATCH=0 vs ours. Preferred long-term: extend PIPED_CONCURRENT so it runs beside SPLIT_MTP_BATCH and staged sources, using our PLE setters | 020ed391, c7e8fe1e, 377f1b33, 74a3b4f2 + c6939188 + 90909f7c (one unit), 15ccf028 + aff75ebb, 5fb49a53 + df7f0409. Drop 56690405 and 7fcea102 (MAX_QUEUE, waiting=N) | hard |
| F16 | **SOLO_PIPELINE=1 + SPEC_LANES=4 (LC_SHARE)** | NP; ALT PW_WITH_BATCH (d17c52e7, only lifts a gate) and PIPED_MTP #4121 (needs `--batch-mtp`) | 1 / 4 | **A/B tonight**: PW_WITH_BATCH and PIPED_MTP at 1/2/4 users vs prod (ours: +45 % at 2 users, +38/+26 % at 3/4). Port ours unless upstream comes within ~5 % | 81015495 + 0238f7f8, 7db06746 + 1edbfdb9, 25221fc9, a069ac58, abdc40fb, ba107591, f60f4ea5, ee403737 (after F15) | hard |
| F17 | **PARK_ASYNC (ours: capture beside slots)** | **NAME CLASH**: upstream f73b3c86 is a different VRAM-stage park | 1 | port ours **renamed `STRATA_PARK_CAPTURE`** (+ PARK_CAPTURE_THREADS/_VERIFY), rebased on ParkSource (#2526); never set `STRATA_PARK_ASYNC=1` on a v0.2.12 engine | b1daa17a | hard |
| F18 | SLOT_PARK (+ SUPERSEDE_PREFIX) | NP; ALT #2526 park_slot (returns early with `--mtp` without `--batch-mtp`, which is our config) | 1 | port ours beside park_slot; reconcile with `supersedes_richer` (#4420); no double park | 9e64a578, 12f2715d (also clear `dev_tag`) | hard |
| F19 | CONV_MMAP + GATE_EVICT (#3115) | NP | 1 / 1 | port ours | 823ed5e6 + 630b7606, 16c85fc4 (park half first, slot-capture half after F18) | moderate |
| F20 | CONV_TIER_DIR (SSD) + /metrics tier | NP | DIR, WARM, MIN_TOKENS 8192, WRITE_GB_DAY 1000, CAP_GB 500 | port ours; the fingerprint must cover the upstream env defaults | 177472ea, 4341ea6f | moderate |
| F21 | PARK_BORROW on a layer split | ALT (c9e571e0 is gated on `stages.empty()`, so it never runs on a split) | 1 | **rebuild as an extension of upstream's `borrows()` / `guard()`** (split path + our `2*match < len` rule); counters in ConvCacheLog | 09aedda6 + 324b1d80, f8fcc276 | hard |
| F22 | SLOT_CHAIN_REUSE + GIVEBACK_CHECKS | NP | 1 / 1 | port ours (after F18) | d9927ecf + bbf90765 | hard |
| F23 | MSG_CKPT_TAIL=8192 | NP; ALT `--prompt-cache-tail` (single GPU only) | 8192 | port ours beside the prompt-cache-tail selection (only one tail alive) | 7f86f97c | moderate |
| F24 | **Vision**: batch guard, per-slot M-RoPE (MROPE_SLOTS=1, REGION_SKIP), server image gate + dead-encoder isolation, image diag line | NP (bug #1242 still in v0.2.12) | MROPE_SLOTS=1 | port ours, and it is a must: without it image rows corrupt batch output | 674fbe41, 77e9b9e1 + 12242dcb, f8a41fff, server 1b9247b1 + 1ce30723, docs f2ef22b3 (rewrite) | hard |
| F24b | IMAGE_HISTORY | NP | no | **drop** (db073efd, 6a379e65) | – | – |
| F25 | **Fair reads, engine**: READ_YIELD=1 | NP (upstream only has the #656 BYIELD) | 1 | port ours; a yield point on every read path, including upstream's piece path and the PIPED_CONCURRENT thread | 45073129, 6c9607f5, baadb797 (after F15) | hard |
| F26 | **Server admission**: YIELD_SHORT=2048, YIELDS_MAX=4 (env), ADMIT_BY_NEW=1, top-row reservation, YIELDNO | NP | systemd env | port ours with a **redesigned entry record** (named fields, compatible with upstream's `[plen, t, new, prompt, cancel]` / COALESCE) | 5549a72b, bda03b22, c50cdf98 + 9db053af + 90f73293 (as one rewritten unit), serve/admit_est.py, test_read_yield.py | hard |
| F27 | GET /v1/load (+ GET /load, LOAD_PEAK_FILE, LOAD_REF_TOK_S) | NP; upstream POST /v1/load = model load | yes (scripts poll it) | port ours; GET only, never shadow POST; consider a `/v1/load_status` alias for upstream | 44d8ff4b, 3850a79e (`load_counts` also counts coalesce/SOLO_AGAIN) | moderate |
| F28 | PREFIX_DIAG=1 | NP | systemd env | port (self-contained module plus a ~31-line hook) | bba60a17 | moderate |
| F29 | test race fix conversation_snapshot_test | NP | infra | port (clean) | f17f3eec | trivial |
| F30 | Genesis parser replay | tool | infra | carry; vendor the old `frontend.py` (90f73293) as a fixture | 07bfcded | trivial |
| F31 | Stage-2 background reads (READS, SERVER_READS, FG_BELOW, quantum…) | NP; ALT PIPED_CONCURRENT | **off** | **drop** from the port; keep on side branch `g3-reads-side` | f6f2a9c6 … 3af0006d, 99602e75, c87b2d7b, d9e25df3, 22f4781f | – |
| F32 | STRATA_TRACE admission timing | U= | no | take upstream | none | – |

Production switches that stay set but are now upstream-owned: MMQ_DIRECT, GROUP_SORT, BYIELD_NODRAIN, ADMIT_MERGE_FIRST, CKPT_POOL, CKPT_THREADS, SLOT_CKPT_MOVE, SLOT_COPY_*, SPLIT_MTP_BATCH, SELF_COMMIT, NAN_GUARD=0, QSA_SLOTS, HANDOFF_PIPE, KV_PREFETCH, GR_FAST, GR_NORM_FAST, MMVF_ROWS.

Switches that need our port: PARK_ASYNC (rename), SLOT_PARK, CONV_TIER_*, MROPE_SLOTS, SLOT_CHAIN_REUSE, GIVEBACK_CHECKS, MSG_CKPT_TAIL, QSA_SLOTS_STREAM, ATTN_VBATCH, GR_DOWN_PF, EARLY_SAMPLE, PIPE_OVERLAP(_KVS), SPEC_LANES, SOLO_PIPELINE, READ_YIELD, PARK_BORROW, CONV_MMAP, GATE_EVICT. On the server side: ADMIT_BY_NEW, YIELD_SHORT, YIELDS_MAX, PREFIX_DIAG.

---

## 2. Branch layout, patch groups, config changes

### Worktrees and branches
- `/media/raid0/StrataQ38FN/Strata-212` (exists, g3-212 @ v0.2.12): **stock** arm. Build it twice:
  - (a) with setup.py flags only (MMQ_TAB off, like the release);
  - (b) the same plus `-DSTRATA_MMQ_TAB=ON` (fair stock arm with MMQ_DIRECT/GROUP_SORT working).
  Name the binaries `engine/strata-212-stock` and `engine/strata-212-stocktab`.
- New `/media/raid0/StrataQ38FN/Strata-212m`, branch **g3-212m** from 146df3c4: the **port** branch. One commit, or a few, per group. Each message says `port <our shas>` and cites TESTLOG #, author havok2-htwo noreply. Push to our fork only; never push upstream without approval.
- Side branch `g3-reads-side`: F31, not built.
- Keep the production worktrees (Strata-ll10-borrow, Strata-143-prod) untouched as references for resolving conflicts.

### Patch groups in cherry-pick order

| Group | Content | Depends on | Ready tonight? |
|---|---|---|---|
| **P0** | Build/doc glue: production build script (flags below), tools call `tools/batch_test.py` instead of batch_test16/32 (or carry ours as wrappers) | – | yes |
| **P1** | Small deltas that are also upstream PRs: F3 NVFP4 guard, F9 stop-bound CUDA ungate + guard, F12 ProgressWriter, F29 test sync, F14f SPLIT_OWN_RESERVE_MIB, b628d059 gr_parity sizes | – | yes |
| **P2** | Kernel levers: F14 QSA_SLOTS_STREAM, F14c GR_DOWN_PF, F14d EARLY_SAMPLE (on `slot_row_draw`). F14b VBATCH only if tonight's A/B says so | P1 | yes (VBATCH no) |
| **P3** | RAM tier base: F19 CONV_MMAP (merge with DefaultInit/`resize_for_overwrite` Segment), GATE_EVICT park half | – | maybe |
| **P4** | F17 PARK_CAPTURE (renamed, on ParkSource) → F18 SLOT_PARK + 12f2715d → GATE_EVICT slot half | P3 | no |
| **P5** | F20 CONV_TIER + metrics | P4 | no |
| **P6** | F21 borrow on a split (extension of upstream borrow) + f8fcc276 | P4, P5 (tier exclusion), later P9 (yield exclusion) | no |
| **P7** | F22 SLOT_CHAIN_REUSE/GIVEBACK, F23 MSG_CKPT_TAIL | P4 | no |
| **P8** | F24 vision engine (guard, regions + gate); audit **every** v0.2.12 position writer (PIPED_MTP, PIECE, BCO coalesced prefill, layer-major, slot drafters, SLOT_COPY_DIRECT) | – (independent, can run in parallel) | no |
| **P9** | F15 PIPE_OVERLAP(_KVS) incl. the BYIELD_NODRAIN condition, or the PIPED_CONCURRENT extension, depending on tonight | P1 | no |
| **P10** | F16 SOLO_PIPELINE + SPEC_LANES (recalibrate kVramSafeMib / LC_LAZY_MIB against the v0.2.12 VRAM layout; refuse lanes when PIPED_MTP is on) | P9 (ee403737) | no |
| **P11** | F25 READ_YIELD engine | P9, P4 (parked conversation carry) | no |
| **P12** | Server: F26 admission/yield (one rewritten entry structure), F27 /v1/load, F28 PREFIX_DIAG, F24 server image gate + diag, F30 Genesis replay | engine INFO lines from P11 and P8 | partly (F27/F28 standalone) |

Effort estimate: P0–P2 about 1 day. P3–P7 2–3 days. P8 1–2 days. P9–P11 3–5 days (less if we adopt upstream alternatives). P12 1–2 days.

A full-parity build is realistic for a second night window, about a week out.

### Build command for g3-212m (and stocktab)
Pin to CPUs 0-63 so the build does not compete with production's `--aux-cpus 65-127`.

```
cmake -G Ninja -S $WT -B $WT/build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_MMQ_TAB=ON -DSTRATA_GGML_DIR=/media/raid0/StrataQ38FN/Strata/third_party/llama.cpp \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
```

After configuring, check the CMake log:
- `STRATA_MMQ_TAB on: pointer-table MMQ ...` must appear;
- `STRATA_MMQ_TAB off: mmq.cuh differs` must not.

`-DSTRATA_WINDOW_MAX_T=16` is a later lever with a VRAM cost; it is not used tonight.

### Production config changes for a v0.2.12-based engine (new file `configs/T212-port.json`; production.json stays unchanged)
- **Remove** `STRATA_PARK_ASYNC=1`. **Add** `STRATA_PARK_CAPTURE=1` once P4 exists. Until then the stock and partial arms run without park capture.
- **Pin, for parity with production** (identity and baseline arms):
  - `STRATA_GDN_TC=0`, `STRATA_GDN_CONVL2=0`, `STRATA_PROMPT_ATTN_IMMA=0` (all bit-changing);
  - `STRATA_AUTO_NOBORROW=0`, `STRATA_PF_W16_CACHE_MIB=0`;
  - `STRATA_SM86_PERF=0`, or `STRATA_ATTN_DAILY=0` alone if the rest of the sm_86 table proves exact and helpful;
  - `STRATA_BATCH_GDN_GROUPS`, `STRATA_BATCH_ATTN_ROWS`, `STRATA_CKPT_POOL_READY` set explicitly per arm.
  - `STRATA_DEFAULTS=026` only for a "0.2.6 behaviour" identity arm: it also reverts GR_UP_WIDE, GR_V3, PLE_AUTO_RAM.
- **Keep set explicitly** even though upstream now defaults them: CKPT_POOL, SLOT_CKPT_MOVE, SLOT_COPY_DIRECT, SLOT_COPY_BATCHSYNC, BATCH_SELF_COMMIT, QSA_SLOTS (the server's `setdefault` respects them, and explicit values document intent). **Keep** GR_NORM_FAST, BYIELD_NODRAIN, ADMIT_MERGE_FIRST, CKPT_THREADS, GROUP_SORT, HANDOFF_PIPE=8, MALLOC_* (the arena problem is still unfixed upstream).
- Top-level keys:
  - `"tool_call_recovery": false` at first (upstream default flipped to true), then A/B it with the Genesis replay and agents.
  - Review `undeclared_tool_calls` (default `text`).
  - Penalties now count reply tokens only (#4590); `STRATA_PENALTY_PROMPT=1` restores the old window if a client depends on it.
- Server systemd env is unchanged in name. `STRATA_YIELDS_MAX` only works with our P12 port: upstream hardcodes 2.
- Check that `--mtp` still selects MTP: drafting.py `blocked_by` returns true for slots > 1. Confirm via the startup log tonight.
- Scripts (deploy-*.sh, window-*.sh, prefix_fanout.py, vllm_facade tests) need GET /v1/load. With stock v0.2.12 they fail, so the stock arms tonight must not rely on it: use `/metrics` or the INFO lines instead.

---

## 3. Risks and conflict hot spots

1. **The sm_86 table silently disables our levers.** `STRATA_ATTN_DAILY=1` makes `pre75_attn()` true. That (a) skips VBATCH and (b) makes `qsa_decode_attn_slots()` return false, so QSA_SLOTS and QSA_SLOTS_STREAM fall back to per-row launches without any error.
   - Advice: the port arm must log which attention path is active; add a told-once INFO line.
   - Decide DAILY vs VBATCH+slots by measurement. If DAILY wins, give `attn_chunk_slots_kernel` a pre75 body instead of carrying VBATCH.
2. **PARK_ASYNC name collision.** Our env would switch on upstream's VRAM-stage park (it needs 1.5 GB of free VRAM, which our cards do not have) and lose ours.
   - Advice: rename ours. Add a startup warning when `STRATA_PARK_ASYNC=1` and `STRATA_PARK_CAPTURE` are both set.
3. **Bit-changing upstream defaults** (GDN_TC/CONVL2 from the server, IMMA, AUTO_NOBORROW) break every identity comparison. Pin them as listed in section 2. Treat them as separate quality-checked arms (`tools/force_check.py`, KL), never in identity gates.
4. **generate.cpp (90 hunks, 14.2k → 21.5k lines).** Re-place code by function, not by line. Anchors:
   - PLE open / batch setup (P9);
   - the BGEN/BGENI parser next to `geni && !o.vision` (P8);
   - the parked-restore block with upstream's `borrow` (P5/P6);
   - park_slot at ~14430 (P4);
   - the RAM gate at ~12185 (P3);
   - the message checkpoint at ~17865 next to prompt-cache-tail (P7);
   - read_part / `concurrent_read_off()` / batch_read.hpp pieces (P9/P11).

   Run host tests after every group. Keep each group building on its own.
5. **verify.cpp / mtp.cpp (vision).** Every new position writer (PIPED_MTP, PIECE, BCO, layer-major, slot drafters) must add the region base. If one is missed, image slots corrupt silently.
   - Advice: grep every `mrope`/`pos` write and every `stage_batch*` call, then rerun `tools/vision_batch_check.py` (including promotion, resumed image turns, all groups) as a hard gate.
6. **Arena carve (SPEC_LANES).** Upstream added verifier buffers (arena_b_, dc_arena_, DFlash2 and PIPED_MTP save buffers). The carve signature and the poison exclusions must cover them. Lane VRAM fit must be recalibrated, or lanes fail at boot or OOM under load.
7. **VRAM headroom.** The new buffers, the FP16 W16 cache and CKPT_POOL_READY all cut into the 1.6–2.6 GiB per card left after the expert cache. Record "free MiB per card" at boot for every arm; an abort criterion follows in section 4.
8. **RAM / oomd.** CKPT_POOL_READY=4 adds about 0.5 GB, beside the 124 GiB conversation cache.
   - Watch memory pressure. systemd-oomd kills under user@1000 at 50 % pressure.
   - Run test servers outside user@1000 slices, or watch `oomctl`.
   - No MemoryHigh next to production.
9. **server.py (29–31 hunks, plus a structural rework).**
   - The entry layout clash (`e[3]`/`e[4]`) is the main trap. Use named fields (a small dataclass or dict) so COALESCE_PREFILL and our ADMIT_BY_NEW/YIELD fields cannot shadow each other.
   - Send BYIELD via `_send_for(born, ...)` (#4451).
   - Parse YIELDNO next to YIELDED.
   - `admitted()` must fire on the COALESCE preslot path too.
   - Use upstream's `serve/test_admit_order.py` and add our cases (add/add conflict).
10. **Two parking mechanisms** (ours SLOT_PARK and upstream park_slot) and **two borrow mechanisms**. Avoid double parks: make park_slot defer when our SLOT_PARK capture is pending for that slot. Extend upstream borrow rather than keeping a second one.
11. **Client behaviour changes** (tool-call recovery, undeclared tools, fenced calls not run, penalties). Gate them with the Genesis replay (07bfcded) and a short agent run before any deploy.
12. **Build.** CUDA arch default 120, MMQ_TAB silently off, `BUILD_TESTS` default OFF. Use the fixed build script from P0 with a CMake log check.

---

## 4. Night window plan (GPU from 23:45 CET)

**Before 23:45, no GPU use:**
- set up the worktree;
- port P0–P2 (P3 if it fits);
- build `strata-212-stock`, `strata-212-stocktab` and `strata-212m-p2` on CPUs 0-63 with nice.
- Prepare the arm configs (`configs/T212-*.json`, port 8091, own log files).
- Prepare the window script `linux/window-212.sh`, modelled on window-1441 and window-reads2. It must restore production in a `trap` on any exit.

Open question for you: are CPU-only host tests (ctest without a card visible, `serve` unittests) before 23:45 already "testing"? Default: run them at 23:45 while production is still serving.

Coordinate with the docs/bench session before stopping production. Production has to be back by the agreed end time; the default plan is 23:45 to about 04:15.

| Time | Step | Gate / abort |
|---|---|---|
| 23:45–00:00 | Record production: binary sha, config, `/metrics`, GET /v1/load, RSS, VRAM. CPU tests: `CUDA_VISIBLE_DEVICES= ctest -j16` on the 212m build; `python -m unittest discover -s serve` on 212m (production still running) | stop if our host tests fail (fix tomorrow; the stock arms can still run) |
| 00:00 | `systemctl` stop strata.service; check that all 4 GPUs are free | – |
| 00:00–00:30 | GPU ctest subset, one card, on 212m: prefill_mmq_tab_test, expert_cache_slot_guard, qsa_slots_parity, qsa_slots_stream_parity, gr_parity_* (down_pf2/3, norm_fast), sampler_same, conversation_snapshot*, conversation_copy_streams, batch_rows_parity, qsa_merge_daily_parity. Pin `STRATA_ATTN_DAILY=0` for our parity tests | any mismatch in our parity tests → drop that lever from tonight's arms |
| 00:30–01:15 | **Identity**: dump `tools/batch_test.py` and `batch_test16 --batch 16 --n 16 --long-tokens 40000 --dump` with production ll10b; compare (btcmp/bootcmp) with 212m-p2 in parity pins. Run `batch_interleave_test`, `batch_groups_cache_test` (SLOT_CKPT_MOVE on and off), `cancel_park_test`, `early_close_test`, `stop_resend_test` against 8091. Note: 212m-p2 lacks park capture / SLOT_PARK / tier / borrow / overlap / lanes, so the comparison is token-level only, not cache behaviour | token mismatch in exact arms → stop the perf arms of that binary |
| 01:15–03:30 | **Performance A/B** on 127.0.0.1:8091, text only. Each arm: boot (wait for ready), short + long HTTP (existing ab-fam scripts), multi-turn (bench_multiturn), 1/2/4-user per-user decode, cold 32K join beside 4 streams. About 12 min per arm; close calls get a repeat (r2) | see the abort list |
| 03:30–03:50 | Spare / r2 for the closest pairs | – |
| 03:50–04:15 | **Restore production**: start strata.service with the unchanged production.json + ll10b. Check `/v1/models` on 8090, one chat, one image request, GET /v1/load, `/metrics` `conversation_cache.tier`, the log ("conversation tier", lanes, overlap INFO). Write the results into TESTLOG | hard deadline; restore wins over any remaining arm |

**Arms** (all text-only on the stock binaries, since they still have the batch M-RoPE bug):

| Arm | Binary | Settings |
|---|---|---|
| A | ll10b (production) | production.json as is (reference) |
| B | 212-stocktab | production env minus PARK_ASYNC, parity pins, sm86 off |
| C | B | + upstream stack defaults (BATCH_GDN_GROUPS=1, BATCH_ATTN_ROWS=1, CKPT_POOL_READY=4) + sm_86 table on (DAILY=1) |
| D | C | + `STRATA_IQ4_W4A8=1` (exact, direct prompt lever for IQ4_XS; check the interplay with MMQ_DIRECT) + `STRATA_SELECT_TWO_PASS=1` |
| E | C | upstream overlap: `STRATA_SPLIT_MTP_BATCH=0` + `STRATA_PIPED_CONCURRENT=1` (+ PIECE 16384). Measures the join pause against A's overlap. Check the log that it engages |
| F | C | `STRATA_PW_WITH_BATCH=1` (1-user decode vs A's SOLO_PIPELINE) |
| G | C | `--batch-mtp` + `STRATA_PIPED_MTP=1` (2/4-user decode vs A's SPEC_LANES). Drop it if `--batch-mtp` does not fit the VRAM |
| H | 212m-p2 | parity pins + DAILY=0 + QSA_SLOTS_STREAM + GR_DOWN_PF=2 + EARLY_SAMPLE (our levers on the new base) |
| H2 | 212m-p2 | DAILY=1 (sm_86 table), QSA_SLOTS_STREAM off: the DAILY-vs-VBATCH/slots decision (VBATCH itself only if it is ported; otherwise H vs H2 isolates the slots/stream part) |
| H3 | 212m-p2 | best of H/H2 + IQ4_W4A8 + the stack defaults (candidate for the future production) |
| opt | – | an EARLY_SAMPLE on/off pair at T=0.7 seeded; the non-exact GDN_TC/IMMA arms only with force_check, lowest priority |

**Metrics per arm:**
- aggregate tok/s at 16 concurrent;
- TTFT p50/p95/max (short and long);
- follow-up TTFT p50 (multi-turn);
- per-user decode at 1/2/4 users;
- cold-join pause (s);
- prefix reuse % and conversation-cache hits/borrows (where the build has them);
- peak RSS;
- free VRAM per card at boot and peak;
- identity status;
- the INFO lines showing which paths engaged (DAILY/slots, PIPED_CONCURRENT, PIPED_MTP).

**Abort criteria:**
- Exact arm not token-identical to A → stop that arm and keep its log.
- CUDA error, illegal address, NaN guard, engine crash, or hang over 120 s without tokens → stop the arm, `nvidia-smi` check, next arm.
- Free VRAM under 300 MiB on any card, or an OOM at boot → stop the arm.
- Memory pressure near 40 %, or RSS over 200 GiB → stop at once (oomd hazard).
- Anything listening on 8090 other than production, or any test traffic to 8090 → stop.
- Two arms in a row crashing, or the clock past 03:50 → go straight to restoring production.

**What the night decides:**
- F14b: port VBATCH, or give the slots kernel a DAILY body.
- F15: port PIPE_OVERLAP, or extend PIPED_CONCURRENT (only if E engages and comes close to A's join pause).
- F16: port lanes, unless G/F come within ~5 % of A at 1/2/4 users.
- Whether IQ4_W4A8 and the sm_86/stack defaults join the future production configuration.

Port only what upstream does not match.

---

## 5. Upstream submission candidates afterwards

Submit as havok2-htwo noreply, with your text approval before posting.

**Small fixes:**
1. NVFP4 guard for MMQ_DIRECT (1 line, a correctness fix).
2. SERVE_STOP_BOUND on CUDA plus the `produced_n < max_new` guard (fixes `outp[-1]`).
3. ProgressWriter for #4271 part 2 (a slow but still reading client no longer fails after 60 s).
4. conversation_snapshot_test sync race (flake 4/10 → 0/30).
5. ConvCacheLog counting upstream's own `borrowed` lines.

**Correctness, high value:**

6. Batch + vision: per-slot M-RoPE regions + the mixed-row guard + the server image gate (#1242 still reproducible). 15/15 image checks, 491 tok/s with images under load.

**Measured cache and memory wins:**

7. Borrow on a layer split (extends c9e571e0): reuse of prompts ≥100K 73.3 → 91.7 %, 118 borrows, 0 failed (TESTLOG #94).
8. CONV_MMAP + GATE_EVICT (#3115 closed unfixed): peak RSS 148 → ~90 GiB (#95).
9. SSD conversation tier CONV_TIER_DIR: +25 % under forced overflow, neutral otherwise (#72).
10. SLOT_PARK for `--mtp` without `--batch-mtp` (park_slot's gap): +2.6/+2.9 % multi-turn, follow-up TTFT 22.9 → 20.6 s (#70).
11. Park capture beside slots (ours, renamed): +7.2/+6.1 % one-shot, park stall 70 → 20 ms, 0 VRAM (#68). Position it against linxuhao's PARK_ASYNC as the zero-VRAM variant.
12. MSG_CKPT_TAIL / SLOT_CHAIN_REUSE / GIVEBACK_CHECKS on layer splits. The gains need a clean single-feature figure from TESTLOG before submitting.

**Scheduling and fairness:**

13. READ_YIELD + YIELD_SHORT + ADMIT_BY_NEW: short TTFT max 20.2 → 1.19 s, 16-short burst 17.1 → 4.9 s, follow-up TTFT p50 11.8 → 3.1 s (#87–#89).
14. PREFIX_DIAG (diagnostics), GET /v1/load as a load report (propose the path `/v1/load_status`).

**Throughput (depend on tonight's A/B):**

15. QSA_SLOTS_STREAM: +4.3/+2.8 % HTTP (#77), plus a DAILY-compatible slots kernel if DAILY wins.
16. GR_DOWN_PF: +1.4/+1.8 % in the gr arm (#77).
17. SPLIT_OWN_RESERVE_MIB + the 11,24,37 rebalance: +4.9/+3.6 % (#79), after re-measuring on v0.2.12.
18. Overlap read beside SPLIT_MTP_BATCH (as an extension of PIPED_CONCURRENT): join pause 7.7 → 0.35 s at 32K, 19.3 → 0.29 s at 100K (#80/#81).
19. SOLO_PIPELINE + SPEC_LANES vs PIPED_MTP: +45 % at 2 users, +38/+26 % at 3/4 (#82/#84), submitted only if it still beats upstream on our box.
20. EARLY_SAMPLE: measure on its own first; there is no figure yet.

**Not to submit:** the stage-2 READS series (worse tails, −5 % throughput), the GR splits, ROUTE_PLAN_FUSE, IMAGE_HISTORY, PIPE_OVERLAP_MAX_QUEUE.

## Relevant paths
- Port plan inputs and scratch: `/tmp/claude-1000/-media-raid0-StrataQ38FN/07837ea6-13aa-4738-916e-a62b73a0cbb0/scratchpad/inv/` (releases.md, server-0212.py, mt_3af0006d.txt, mt_22f4781f.txt, conflicts.txt, cmake_v0212.txt)
- Stock worktree: `/media/raid0/StrataQ38FN/Strata-212`. Proposed port worktree: `/media/raid0/StrataQ38FN/Strata-212m` (branch g3-212m, not created)
- Production references: `/media/raid0/StrataQ38FN/Strata-ll10-borrow` (engine 3af0006d), `/media/raid0/StrataQ38FN/Strata-143-prod` (server 22f4781f), `/media/raid0/StrataQ38FN/linux/configs/production.json`
