# Code-Analyse: Batching-Hebel für Strata auf 4× RTX 3090 (2026-10-11)

Vier Analysten haben je einen Hebel im Code untersucht, je ein Gegenprüfer hat jede Zeilenangabe nachgeprüft. 
Zeilennummern beziehen sich auf Strata v0.1.41 plus die beiden lokalen Patches (Branch `g3-batch-over-8`). 
Die Gegenprüfer-Spalte ist maßgeblich.

| Hebel | Analyst (Tage) | Gegenprüfer (Tage) | Urteil |
|---|---|---|---|
| A: interleaved / chunked prefill. The cheapest variant that removes most of the blocking i | 3–8 | 4–10 | optimistic |
| B: MTP drafts inside pipelined batch windows with a layer split (--batch-mtp across 4 GPUs | 6–14 | 8–16 | optimistic |
| C: per-window overhead of the pipelined batch windows (pump / batch_launch / batch_poll /  | 4–12 | 5–14 | optimistic |
| D: quick wins without deep engine work (env/CLI knobs, --batch-groups, data-parallel insta | 0.5–2 | 0.5–2.5 | optimistic |

## B: MTP drafts inside pipelined batch windows with a layer split (--batch-mtp across 4 GPUs, one draft row per slot)

### Mechanismus heute

ONE GPU (--batch-mtp): at startup (generate.cpp:4006-4018) every slot gets its own MtpDrafter. It shares the main drafter's weights (draft layer and draft head, MtpDrafter::load with `shared`, mtp.cpp:244-262), but its draft K/V and buffers are private. It is loaded against stage 0's slot session bslot_ss[0][b] and limited to set_max_drafts(1). In serve (6818-6845) the CUDA0 verifier is initialised with max_t=kVerifyMaxT=8 (verify_kernels.hpp:21) and set_batch_graph_limit(64), because slot rotation produces many row layouts and old graphs are evicted LRU. Each slot also gets a residual buffer and binds to `wt`/native_head. batch_step (8389-8510) gives every active slot two rows: [x at p, draft[0] at p+1]. A window therefore holds at most 4 slots, and with more slots `static next_slot` rotates them through 8-row windows (8402-8423). The window runs through run_slot_rows. That path chains through next_ for a split (verify.cpp:2624-2720). The window kernels already handle consecutive rows of one slot: GDN conv/recurrence per slot group (verify.cpp:993-1031), and an indexer-tail snapshot plus per-row attention over the slot's own K/V (1113-1119, 1224-1236). The host accepts the draft only if pick==draft, it is not EOS/stop, and the slot has room (8444-8450). commit_slot_prefixes(keep) then writes keep per slot into the mapped staging h_commitb_ and launches the commit graph. That graph applies the GDN state, the indexer append and the PLE history for only the kept prefix (verify.cpp:2728-2766, capture_commit_batch 2389-2468), and it chains to next_. Drafting runs per slot and serially: a cudaMemcpy of the 2 final_R rows, then MtpDrafter::draft, which ends in cudaStreamSynchronize (generate.cpp:8482-8493, mtp.cpp:1245-1372). Admission builds the first draft with draft_first(ver.final_R_all()) (10982-10990). copy_to_slot/copy_from_slot copy the draft K/V between the solo drafter and the slot drafter (8334-8344, 8378-8388).
WHY IT IS REFUSED WITH A SPLIT: the option gate turns it off for any multi_gpu/split_same/stage setup (3871-3878). The #1413 warning at 8521-8525 says the pipelined path "builds one row per slot". In verify.cpp, stage_batch rejects duplicate-slot rows whenever next_ != nullptr ("grouped slot rows do not support a layer split yet", 2545-2548) and compares hbase+S to the slot count instead of the hand-off row count (2526). The structural reason is the pipeline itself. PGroup/pump (8527-8640) launches group g as rows base..base+S-1, one per slot, via Verifier::batch_launch (verify.cpp:2783-2804). That function launches the commit graph directly behind the window ("right behind it: every row is kept", 2790; header verify.hpp:173-177: "a batch window keeps every row, so the commit needs no host decision"), and it advances the PLE ple_prev on the host at launch time. With a split, the accept decision exists only after the last stage samples. Meanwhile stage k has started another group's window, and that window overwrites the window-row-indexed intermediates the commit needs: qkv_L_, h_L_, gate_L_, beta_L_, idx_raw_L_, hist_snap_ (allocated verify.cpp:528-554, read by the commit at +first*C, 2413-2447). It also overwrites the per-verifier "last window" fields last_rows_/last_t_/last_pos_b_/row_base_ (2606-2612). The draft layer and its head live on the last stage's GPU (mtp.load under OnDevice(last_st->dev), generate.cpp:3997-4004, bound to last_st->wt/head at 6820-6822). The per-slot drafters are still created against CUDA0 objects: bslot_ss[0], `wt`, native_head (not loaded with a split). Today's numbers: per the log, CUDA3 holds 6141 of 6144 experts and has only ~270 MiB free after the window captures. The draft layer takes 839 MiB plus the draft head 213 MiB on CUDA3.

### Nötige Änderungen

- **src/program/generate.cpp:3867-3922 (batch_mtp gate, g3 pipe_cap block)** (small, ~25 lines): Allow batch_mtp with a pipelined split where each stage has its own GPU (keep refusing split_same and the helper GPU). Enforce 2*(batch/groups) <= kVerifyMaxT, i.e. at most 4 slots per group and 16 slots on 4 GPUs. Keep the existing kVerifyMaxT clamp consistent with the g3 patch.
- **src/program/generate.cpp:4006-4018 (slot drafters) and 6829-6845 (slot_mtp_rows, bind)** (small, ~30 lines): Create the slot drafters under OnDevice(last_st->dev) and load them against bslot_ss.back()[b], so the draft K/V geometry and device are the last stage's. Use bslot_ss[0][b] as the PLE session. Allocate the residual rows on the last GPU. bind(last_st->wt, &last_st->head, r, err, &mtp), which also satisfies the shared-head check in mtp.cpp:620-629. Add OnDevice to the draft K/V copies in copy_to_slot/copy_from_slot (8334-8344, 8378-8388) and to draft_first at admission (10982).
- **src/program/generate.cpp:6718-6722 (hand-off size), 6745/6757 (stage verifier max_t)** (small, ~10 lines): Hand-off must hold 2*batch rows, because group g writes rows [g*2*GS, (g+1)*2*GS). Stage verifiers need max_t=kVerifyMaxT under batch_mtp (today min(batch,8) plus spec). Give the stage verifiers set_batch_graph_limit too.
- **src/core/verify.cpp:2523-2613 stage_batch** (small, ~25 lines): Drop the 'grouped slot rows do not support a layer split' guard. Check hbase+S against the hand-off row capacity (new member), not slots_.size(). Add a defer flag that skips the default-keep write into h_commitb_; otherwise the next launch races the deferred commit graph's copy_i32_from_mapped of m_commitb_.
- **src/core/verify.cpp:2783-2804 batch_launch + include/strata/core/verify.hpp:173-190** (medium, ~120 lines): New batch_launch_rows(rows,S,hbase,tokens,pos,defer): window graph only, no commit, no host ple_prev advance. New batch_commit_deferred(group, rows,S,pos,tokens,keep): writes keep and positions into h_commitb_ for the group's slots, launches commit_bm_[key] asynchronously on the stage's cs_ (stream order puts it before the group's next window on that stage), and advances ple_prev by `keep` tokens. The rows/pos/tokens of each group's last window are stored per group, not in last_rows_/last_t_/last_pos_b_.
- **src/core/verify.cpp:2309-2344 init_slots, 2346-2468 capture_batch/capture_commit_batch** (medium, ~100-150 lines): Per-group stash of the commit intermediates: qkv_L_/h_L_ (C floats per row) and gate_L_/beta_L_ (HV) for this stage's 9 GDN layers, idx_raw_L_ for its 3 QSA layers, and the hist_snap_ rows on the PLE stage. Either add D2D copies at the end of the deferred window graph into stash[hbase/(2*GS)] and capture the commit graph to read from there, or offset those writes by row_base_ as the hand-off already does (verify.cpp:752, 774, 1528). Size is groups x 8 rows, a few MB per stage. tail_snap_b_ and the slot states are already per slot, so they are safe to defer. This is the part that decides exactness.
- **src/program/generate.cpp:8513-8650 pump / PGroup / pipe_drain** (medium-large, ~200-250 lines): MTP mode. PGroup carries 2*GS rows (rows[], first[], tok[], pos[]) at a FIXED S=2*GS (pad rows), so each stage has exactly one graph key per group; a variable G.S (8610) would thrash graph captures on CUDA3. On the last stage: the accept rule from batch_step (8444-8450), 1-2 BT lines per slot, EOS/stop/max_new/context handling, then batch_commit_deferred on all n_pipe stages (each under OnDevice), then copy the slots' 2 final_R rows from the last stage's R_ into slot_mtp_rows and draft. The group may only restart once its drafts are in. pipe_drain (8647, 8659) must also flush pending commits and drafts before the prompt path touches sessions.
- **src/core/mtp.cpp:1404-1484 (chain_launch/chain_poll) and generate.cpp pump** (v2 medium ~100 lines; v3 large ~300-500 lines): v2: drafts that do not block the host. A synchronous draft() per slot (cudaStreamSynchronize, mtp.cpp:1320) stalls every stage's hand-on for about 4 x 0.6 ms per group-step. Use chain_launch/chain_poll per slot drafter (needs set_force_capture before load and prepare_chain) and poll them in pump. Optional v3: a batched multi-slot draft forward in MtpDrafter (one pass over up to 4 slots' rows with per-slot K/V, like the verifier's batch_rec_ path), so the draft head and dense weights are read once per group instead of once per slot.
- **tools/batch_test16.py, tools/batch_interleave_test.py** (small-medium, ~80 lines Python): Exactness: --batch-mtp with a split, greedy, with STRATA_IQ_MT_MIN=1, --pcie-frac 0, --adapt-every 1000000. Every slot must match its solo tokens. Also cover a seeded sampled run, admission mid-flight, BYIELD, BSTOP, and the return from slot to solo path, which copies draft K/V on the last GPU.
- **serve/server.py:766,1103,1345 (BT parsing), 2198-2219 (PARALLEL_MAX)** (small, 0-10 lines): No protocol change. The server already accepts several BT lines per slot per window (single-GPU --batch-mtp). At most a log message about the 16-slot ceiling with MTP on 4 GPUs.
- **docs/BATCHING.md 'Limits', docs/MULTI_GPU.md** (small): Document the split support, the 4-slots-per-group ceiling, and the VRAM on the last GPU.

### Blocker und Invarianten

- The commit has to be deferred. Pipelined batch_launch commits all rows right behind the window (verify.cpp:2790; header verify.hpp:173-177). The keep decision exists only after the last stage samples, while stages 0-2 are already running other groups.
- Intermediates are indexed by window row. qkv_L_/h_L_/gate_L_/beta_L_/idx_raw_L_/hist_snap_ (verify.cpp:528-554) are read by the commit at +first (2413-2447), and the next group's window on the same stage overwrites them. They need a per-group stash or row_base_ offsets. A mistake here does not crash: it silently corrupts GDN/indexer state while the text stays coherent, so only bit-exact batch-vs-solo tests catch it.
- Per-verifier 'last window' state (last_rows_, last_t_, last_pos_b_, last_tokens_, row_base_; stage_batch 2606-2612) is per stage, not per group. The deferred commit must carry its own rows, positions and tokens.
- Mapped staging race: stage_batch writes default keep into h_commitb_ (2593-2602), and the commit graph copies m_commitb_ when it starts executing. With commits queued asynchronously, only the decision may write h_commitb_.
- Hard 8-row window: kVerifyMaxT=8 is a kernel constant (verify_kernels.hpp:21, iq_kernels.cu:1237). brow_[8], last_rows_[8], b_out_[8] and rows[8] in batch_launch follow it. With token plus draft that is 4 slots per group, so at most 16 slots on 4 GPUs. 32 slots with MTP would need kVerifyMaxT=16, a separate project.
- Graph keys: bkey = (hbase, rows..., ar_off) (verify.hpp:286-297). Duplicate-slot rows create new layouts. Each window+commit pair costs 20-30 MiB (#997 note, verify.cpp:2492-2495). CUDA3 has ~270 MiB free after the captures (engine log), and a capture synchronises the stage stream (verify.cpp:2373). S must be fixed at 2*GS per group (pad rows) to avoid re-capture thrash.
- VRAM on the last GPU: the draft layer (839 MiB) and draft head (213 MiB) are already on CUDA3, and its cache holds 6141 of 6144 experts. Each slot drafter adds private draft K/V plus buffers: the #1413 warning says ~0.9 GB per slot; at 32K int8 it is probably much less; the 'shared draft weights, X MiB of private state' log line gives the real figure. The slot drafters load before the expert cache is sized, so CUDA3's cache shrinks and the evicted experts fall back to the CPU doorbell path, which slows the last stage. Likely needs an explicit split with fewer layers on CUDA3 (e.g. 12,24,37; CUDA1/2 have about 1.7-2 GiB of spare expert room) or a draft K/V ring (--mtp-window).
- Drafting is on the critical path. Group g's next stage-0 window needs its new draft, and per-slot draft() blocks the host (mtp.cpp:1320), which stalls every stage's hand-on. Asynchronous or batched drafting is needed for most of the gain.
- Pad rows of idle slots write their state, so those slots lose the conversation cache (sl.cached=false, generate.cpp:8612). Fixed-S MTP windows make this more frequent.
- pipe_drain before every admission (generate.cpp:8659) has to flush pending deferred commits and drafts. The slot/solo transitions (copy_to_slot/copy_from_slot draft K/V, draft_first at admission) must all run on the last GPU.
- Exactness invariants from BATCHING.md: STRATA_IQ_MT_MIN=1, --pcie-frac 0, --adapt-every 1e6. Sampled rows use Philox(seed, position) (sample_rows), and the draft is accepted only if it equals the sampled pick. This must stay bit-identical to solo for greedy and seeded sampling.
- Single-user speed does not change: a request alone keeps the solo path with spec-4 MTP (121 tok/s); a slot left alone still returns to it.

### Erwarteter Gewinn (Analyst)

Measured basis: 4 groups x 2 rows = 345 tok/s, so 5.8 ms per stage-step; 4 groups x 4 rows = 603 tok/s, so 6.6 ms. The marginal cost is ~0.4 ms per extra row and most of the stage time is fixed overhead. With MTP a group window carries 4 slots x (token + draft) = 8 rows, projected ~8.2 ms per stage-step (measurable without new code using 32 plain slots, 4 groups x 8 rows, batch_test32). First-draft acceptance alpha at temperature 0.7 is about 0.65-0.8: the solo log shows 295/453 = 65% over chains, BATCHING.md shows 140/172 = 81% greedy. That gives 1.65-1.8 tokens per slot-window. One draft per slot costs ~0.5-0.7 ms: it reads 111 MiB dense Q8, the 213 MiB draft head and the active experts, then synchronises.
PURE ENGINE DECODE, 16 slots: (v1) synchronous per-slot drafts take about 4x0.6 ms per group-step and stall every stage: 16*1.72/(4*10.6 ms) = ~650 tok/s, +5-15%. (v2) drafts on the drafter stream, not blocking the host: CUDA3 gets ~1.4 ms more bandwidth per group-step and becomes the bottleneck: ~720 tok/s, +12-28%. (v3) batched multi-slot drafter plus a split with fewer layers on CUDA3: ~8.6 ms per step, ~800 tok/s, +25-40%. That matches the doc's +31-39% for 2-4 rows per window. At 8 users (2 active slots per group) the relative gain is larger: pure ~470-510 vs 345, +35-45%.
THROUGH THE HTTP SERVER the admission stall dilutes it. A = 512/360 - 512/603 = ~0.57 s per short request (log: a ~100-token prompt read in ~330 ms, slot copy 48-86 ms, plus the pipeline drain). Throughput = 512/(A + 512/D). Short prompts: v1 ~377 (+5%), v2 ~391 (+9%), v3 ~410 (+14%) against 360. Long prompts (A = ~1.2 s): +3/+6/+10%, about 257-275 against 250. At 8 users (281 measured, A = ~0.34 s): ~350-380, +25-35%.
Caveat: per window, a draft row is worth only alpha of a real row. With 32 users, 32 plain slots fill the same 8 rows and give more tokens (MTP-16 is about 0.83-0.9x plain-32), as long as the 32 slot sessions do not push experts out of VRAM; through the server, 32 slots measured worse (342 vs 365). So B is mainly a multiplier for 4-16 users. Alone it does not reach vLLM's 710. Only after lever A removes the admission and prefill stall (server approaching the 603 pure figure) does B lift it to ~720-800, i.e. to or slightly past vLLM.

### Erster Prototyp-Schritt

Under a day, without the pipeline: use the non-pipelined split path (--batch-groups 1). It already chains duplicate-slot rows and the keep[] commit through every stage (run_slot_rows forwards next_ at verify.cpp:2655/2714, commit_slot_prefixes at 2765, batch_step at generate.cpp:8389-8510). About 40-60 lines unlock it: (1) let batch_mtp through the gate at generate.cpp:3871-3878 for a split where each stage has its own GPU; (2) stage verifiers get max_t=kVerifyMaxT (6745/6757); (3) in verify.cpp, remove the split guard at 2545-2548 and check hbase+S against the hand-off rows instead of slots_.size() (2526); (4) create the slot drafters and their residual rows under OnDevice(last_st->dev), loaded against bslot_ss.back()[b] and bound to last_st->wt/&last_st->head (4006-4018, 6829-6845), with OnDevice around the draft K/V copies. Build it as a separate binary (strata-g3-mtp) and run tools/batch_test.py with --batch 4 --extra "--batch-groups 1 --batch-mtp --pcie-frac 0 --adapt-every 1000000" and STRATA_IQ_MT_MIN=1. 4 slots x 2 rows = 8 rows, one layout, no rotation. Results: (a) bit-exactness against solo across 4 stages; (b) the 'strata batch: ... ms/window = run + commit + emit' line, where run = the 8-row window over 4 stages, emit = ~the serial drafting cost of 4 slots, and avg rows gives alpha; (c) the per-slot drafter VRAM from the 'shared draft weights' log line. Alongside it, with no code change: tools/batch_test32.py (4 groups x 8 rows) for T_stage at 8 rows. Plug these into tok/s = 16*(1+alpha)/(4*T_stage8 + T_draft) before committing to the 6-14 day pipelined version.

## C: per-window overhead of the pipelined batch windows (pump / batch_launch / batch_poll / per-stage graphs)

### Mechanismus heute

HOST LOOP: generate.cpp:8652-8660 is a pure busy loop: try_next_line (mutex), then pump(true). There are no sleeps anywhere on the hot path. The only sleep_for is in release_gpu_waits (verify.cpp:292), an error path. pump (generate.cpp:8545-8642) first polls every stage with batch_poll (8556). A finished group is marked for stage k+1, and on the last stage its BT lines are printed (8565-8592). It then launches waiting groups from the last stage down to stage 0 (8594-8632), calling batch_launch at 8626. With 4 groups on 4 stages this is a lockstep pipeline: every tick waits for the slowest stage, and stage 0 is launched last.

batch_launch (verify.cpp:2783-2804) first runs stage_batch (2523-2615) on the host. That does map lookups and a capture check, refresh_ar (a scan of 6144 entries), positions and steps, the commit staging, and on stage 0 a synchronous PLE gather (verify.cpp:2589). The gather is 16 random rows per token = 64 rows per group, read with O_DIRECT from the 26.8 GB n-gram table (ngram.cpp:507-550), with no prefetch on the batch path. batch_launch then issues TWO cudaGraphLaunch calls back to back, exec_bm_ and then commit_bm_ (2789-2790), followed by a cudaStreamQuery.

batch_poll (2806-2880) behaves differently by mode. For an all-resident stage (ar_on) it raises the PLE flag (2815-2819), which the stage-0 graph spin-waits on at layer 1 (826). Then it does cudaStreamQuery on cs_ and copy_ (2862-2866). On the last stage, sample_rows (2768-2781) runs per-row sample_tokens plus cudaStreamSynchronize for sampled requests, which blocks the pump.

HOST ROUND TRIPS PER TICK: 4 stage_batch calls, 8 graph launches, about 4x2 cudaStreamQuery per pump pass, and 4 OnDevice switches (cheap). The hand-off goes through mapped pinned host buffers (generate.cpp:6717-6730): the graph tail writes them (verify.cpp:1527-1532) and the next stage's head reads them (773-777). That is about 51 KB per row, about 200 KB per group, about 50 us per tick.

ZERO-DOORBELL: yes. bkey includes ar_off_ (verify.hpp:286-289). All 4 stages log "100% VRAM resident: zero-doorbell graph" with 16 slots, so the experts are planned on the device (resident_plan, verify.cpp:1325-1329) and grouped directly (1450-1454) with no host service per layer.

GPU SIDE (the real cost): record_window (verify.cpp:701ff) in batch mode (batch_rec_) loops per row/slot:
- GDN: gdn_conv_l2_multi and gdn_step_norm_multi per slot (994-1001, 1022-1031). gdn_step uses a grid of only h_v = 48 blocks for one row on 82 SMs (verify_kernels.cu:912ff).
- QSA, per row: tail snapshot (1116), KV append (1157-1176), indexer append (1186-1193), scores, top-k, resolve and 2-kernel attention (1225-1236).

Then a SECOND graph, capture_commit_batch (2378-2468), runs for every tick:
- per slot per GDN layer: gdn_conv_commit, plus gdn_step_norm_multi run again on the 3 MiB fp32 state (2401-2421);
- per QSA layer: tail restore and indexer re-append (2425-2441);
- PLE history copy (2443-2448).

The pipelined path keeps every row anyway. The solo T=1 window avoids all of this with one_token_self_commit (713, 2268), but that path is disabled for batch_rec_.

Counts: about 360 nodes in the verify graph plus about 100 in the commit graph, per stage per tick (4 rows). PDL is sm_90+ only (verify_kernels.cu:1278), so it cannot help on the sm_86 3090s.

NUMBERS:
- 8 slots (4x2 rows): 345 tok/s, a 5.80 ms tick. 16 slots (4x4 rows): 603 tok/s, a 6.63 ms tick. So each extra row costs about 0.41 ms, which is about its peak-bandwidth cost (about 10 extra experts x 12 layers x 2.22 MiB plus 81 MiB of GDN state, about 0.38 ms). Row-proportional work is already efficient. The intercept, about 5 ms per tick, is fixed per-window cost.
- From GGUF sizes, the real bandwidth floor for 4 rows is: dense about 80 MiB per GDN layer and 74 MiB per QSA layer, about 0.94 GiB per stage; experts about 1.04 GiB; GDN state including the commit re-run 0.32 GiB. That gives about 2.6 ms at 936 GB/s, and about 3.1 ms on stage 3, which also runs the Q6_K head (497 MiB). So the assumed 1.5-2 ms bound is too low. The real removable gap is about 2.5-3 ms per tick: latency-bound graph nodes, the redundant commit graph, max-of-4 lockstep imbalance (head on stage 3, PLE on stage 0), and a host share of only about 0.1-0.3 ms.
- No STRATA_VERIFY_PROFILE run exists in the logs, so the split between GPU and host is not measured yet.
- 32 slots is not comparable: the solo runs show about 40x more cache misses (2959 vs 69), so slot sessions pushed experts out of VRAM and onto the CPU.

### Nötige Änderungen

- **no code: run tools/batch_test16.py with STRATA_VERIFY_PROFILE=1 (profile_report, verify.cpp:1631) plus ~40 lines of timers in pump (generate.cpp:8545-8642) and batch_launch/batch_poll** (small, ~40 lines of instrumentation): Measure per stage: GPU ms per window inside the verify graph, commit graph time, busy vs idle (completion seen to next launch), stage_batch time including the PLE gather, and sample_rows time. This shows which stage sets the tick and how much of 6.63 ms is graph work vs commit vs host/idle.
- **verify.cpp:record_window (713, 994-1031, 1116, 918) + Verifier::batch_launch (2783-2804) + capture_batch/bkey (verify.hpp:286)** (medium, ~80-150 lines): Batch self-commit for the pipelined no-MTP path, where every row is kept. Set self_commit per batch row: gdn_conv_l2_multi(commit=true), and gdn_step_norm_multi with keep=one_ (the same kernel template the batch commit already uses for 1-row groups, 2420-2421). Skip the tail snapshot and the PLE history snapshot. Drop the commit_bm_ launch. Add a variant bit to the graph key. Keep the old commit graph for batch_step, --batch-mtp and commit_slot_prefixes.
- **verify.cpp:sample_rows (2768-2781) and batch_poll (2868-2871)** (small, ~40-60 lines plus sampler entry point): Sample all sampled rows in one launch with per-row parameters, and drop the cudaStreamSynchronize: poll the stream or an event instead, so the pump is not blocked on the last stage (matters for server traffic at temperature > 0).
- **verify.cpp:stage_batch (2575-2590) -> batch_poll (2815-2819)** (small, ~30 lines): Take the stage-0 PLE gather off the launch path. Launch first, then gather and raise flag 1: the graph already waits on m_flag_ before copying m_ple_ at layer 1 (826). Alternatively prefetch through PleTable's prefetch API once the group's tokens are known.
- **config only: --layer-split** (small, no code): Unload the head stage: try 12,24,37 / 12,25,37 / 13,25,37 with --trim-stage-weights. Stage 3 carries the 497 MiB Q6_K head (~0.55-0.8 ms) and sampling. Check that all 24,576 experts plus 16 slot sessions still fit (stage 2 had 17.9 GiB of room; 13 layers of experts need ~14.4 GiB).
- **src/kernels/cuda/verify_kernels.cu (gdn_conv_l2_multi, gdn_step_norm_multi), kv_q8.cu (kv_append_q8_step), native_qsa_indexer.cu, qsa_select.cu (scores/topk), qsa_decode_attn.cu, plus the call sites at verify.cpp:994-1031, 1116, 1157-1193, 1225-1236 and the commit at 2401-2441** (large, ~600-1000 lines across 5-6 kernel files plus parity tests): Multi-slot kernel variants: one launch per layer over S rows with per-row device pointer tables (state, conv history, page table, pools, tail, idx buffers), filled at capture. Slot sessions are fixed allocations, so captured pointers stay valid. Each block keeps one row, so the arithmetic stays bitwise per-row. This cuts ~100-150 serialized tiny nodes per stage at 4 rows, ~2x that at 8 rows, and raises GDN-step occupancy from 48 to 48xS blocks. Needs parity tests in the existing *_parity style.
- **(optional, low value) generate.cpp pump + verify.cpp stage_batch staging buffers** (medium-large, ~200-300 lines; not recommended before the items above): Device-side hand-off: pre-launch stage k+1's next graph behind a wait_flag_ge on a mapped ready-counter written by stage k's tail. Needs double-buffered h_step_/h_pos_/h_commitb_/h_tok_ per window parity, and must still work with the doorbell (ar_off) variant that needs host service per layer. Saves only ~30-80 us per stage per tick.

### Blocker und Invarianten

- Exactness gate (batch = solo, token-identical): tools/batch_test16.py and batch_test32.py, batch_interleave_test.py and parking_test.py, run with STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 1000000. Every change above must keep the row arithmetic bitwise identical.
- Self-commit is only valid when every row is kept. That holds for the pipelined path (batch_launch launches the commit 'right behind it: every row is kept', verify.cpp:2790). It does not hold for --batch-mtp/commit_slot_prefixes or for batch_step on one GPU, so both graph variants must coexist and the variant must be part of bkey (verify.hpp:286-289).
- The GDN self-commit must hit the same kernel template the batch commit uses for 1-row groups: gdn_step_norm_multi_kernel<false,...> with n_keep and t_out_begin 0 (verify.cpp:2420-2421; verify_kernels.cu:912ff). Its y output must also equal the all_out variant the batch verify uses today. That is plausible (solo T=1 self-commit is on by default) but must be tested.
- QSA indexer: the verify appends in place after snapshotting the tail, and the commit restores the tail and re-appends from the commitb positions (verify.cpp:1116, 1186-1193 vs 2425-2441). Skipping the commit needs the step_ kStepPos position to equal the commitb position, which must be checked.
- PLE: ple_history_advance already writes each slot's history in place in the verify (verify.cpp:~915). batch_launch's host-side ple_prev update (2793-2798) must stay.
- Fixed 8-row arrays (rows[8], brow_[8], last_rows_[8], b_out_[8]; kVerifyMaxT) cap a group window at 8 rows. The local g3 patch relies on hbase = group * rows-per-group into a hand-off sized per slot (generate.cpp:6717-6722).
- ar_on() can flip per window (prompt loan, VRAM shrink, swap: refresh_ar, verify.cpp:1660-1673). The doorbell variant then needs host service per layer inside batch_poll, so any device-side hand-off or pre-launch scheme must still let the host serve layers.
- No NVLink/P2P on the 3090s: the hand-off must stay in mapped pinned host memory. PDL (programmatic dependent launch) needs sm_90 (pdl_supported, verify_kernels.cu:1278), so there is no launch overlap on sm_86.
- A single host thread serves 4 GPUs and stdin. Any new BGEN line drains the pipeline first (pipe_drain, generate.cpp:8647-8661). That is Lever A's territory and limits what Lever C can show at server level.
- VRAM: new graph variants are small, but each capture syncs the stream (capture_batch at 2346-2376, only at first use of a layout). 32 slots already spills experts out of VRAM (solo misses 2959 vs 69), so per-row savings only pay off at 16-24 slots on this box.

### Erwarteter Gewinn (Analyst)

Pure batch decode, 16 slots, 4x4: the tick is 6.63 ms (603 tok/s). The realistic floor for the slowest stage, from GGUF sizes and real kernel efficiency, is about 3.6-4.2 ms (dense ~0.94 GiB + experts ~1.04 GiB + GDN state 0.32 GiB per stage, plus the 0.49 GiB head on stage 3). The 2->4-row slope (+0.41 ms per row) shows row-proportional work is already near bandwidth, so all gains come from the ~5 ms of fixed cost per window.

Estimates:
- Self-commit (removes ~100 nodes per stage, ~108 MiB of duplicate GDN reads and one graph launch): -0.4 to -0.8 ms.
- Stage balance, sampling and PLE off the critical path: -0.2 to -0.6 ms, only where that stage is the max.
- Multi-slot kernels: -0.3 to -1.0 ms.
- Pure host/pump fixes: at most 0.1-0.3 ms (2-5%). The host round trip is NOT the main problem.

Realistic tick 5.0-5.6 ms, so about 710-800 tok/s pure decode (+18-32%). At most ~1000 tok/s if everything reached the floor (not realistic).

Server level: decode is about 60% of wall time with short prompts (360/603) and about 41% with 1.2K prompts (250/603). That gives about +10-17% for short prompts (360 -> ~400-420 tok/s) and +6-10% for long ones (250 -> ~265-275). Lever C alone does not reach vLLM's 710/400. It multiplies with Lever A (prefill mixed into the windows), which attacks the 40-60% non-decode share. It also makes 8 rows per group (24-32 slots) worthwhile if VRAM allows.

### Erster Prototyp-Schritt

Under a day, once the GPUs are free:
(1) Run tools/batch_test16.py exactly as for the 603 tok/s figure, with STRATA_VERIFY_PROFILE=1 and no code change. Read the 'strata batch GPU stages, stage k (ms/window)' lines to get GPU time per stage inside the verify graph vs the 6.63 ms tick. Add ~40 lines of steady_clock timers in pump/batch_launch/batch_poll: per stage, busy (launch to done) vs idle, stage_batch ms (stage 0 includes the PLE gather), and sample ms. This tells which stage is the max and how much is commit, host, or idle.
(2) Same session: put a quick hack behind STRATA_BATCH_SELF_COMMIT=1. In record_window, set self_commit for batch rows (gdn_conv_l2_multi commit=true, gdn_step_norm_multi keep=one_) and skip the tail/hist snapshots. In batch_launch, skip the commit_bm_ launch. Add the flag to bkey. Rerun batch_test16: the slots must stay IDENTICAL, and the gain shows up as tok/s.
(3) Also with no code: rerun with --layer-split 12,24,37 and 12,25,37 to see whether the head stage is the bottleneck.

## A: interleaved / chunked prefill. The cheapest variant that removes most of the blocking is an overlapped admission: the batched prompt read runs on a worker thread while the main thread keeps pumping the pipelined decode groups. Two fixed admission costs get cut alongside it: the host-bounced slot copy and the per-admission expert-cache loan on CUDA3. Real vLLM-style prompt rows inside the batch windows were investigated and rejected as the first step (8-15 days, structural blockers).

### Mechanismus heute

WHAT HAPPENS TODAY WITH --batch 16 ON THE 4-GPU SPLIT (piped, groups auto)

1) Piped mode is always on here. Auto groups resolve to one group per stage (generate.cpp:3906-3912), so piped = batch>0 && groups>1 && n_pipe>1 (generate.cpp:8520). The main loop (8652-8662) pumps groups with pump(true) (8545-8646):
- non-blocking Verifier::batch_launch over contiguous rows base..base+S-1 (verify.cpp:2783-2804)
- batch_poll (verify.cpp:2806-2876), which on an all-resident stage is just a cudaStreamQuery

2) Every BGEN drains the pipeline first. Any input line other than BSTOP calls pipe_drain() (generate.cpp:8659). BGEN is rewritten to 'GEN 1' (8680-8694) and runs the full solo request path on the MAIN sessions (ss + stages[k]->ss) on the main thread. Nothing pumps until the request path returns.

3) Prompt read. The cuts loop (9890-9943) does the following per segment:
- lend() cache slots (9653-9702)
- read_part() for batched segments, or read_windows() for segments <= --short-read (default 64, generate.cpp:613, windows_ok 9413-9419). read_windows uses the same stage Verifiers as the batch groups.
- checkpoint_at the turn boundary
- refill (9944)

The critical line is generate.cpp:9723: `if (o.batch <= 0 || piped || ...) return sp.run(...)`. In piped mode the whole segment is ONE sp.run. The STRATA_BATCH_DECODE_SHARE loop (9780-9786, which calls batch_step, the non-piped window function) and the BYIELD scan (9731-9779) never execute. A BYIELD that arrives meanwhile is discarded later at 8670. The bypass dates from #465 (commit cab62c9), so it was deliberate.

Consequence: experiment 3 (share 2.0/4.0) cannot change anything. linux/configs/Q3-b12-share4.json has no --batch-groups, so it runs piped. docs/BATCHING.md's claim that the slots decode between the prompt's chunks does not hold for --batch-groups.

4) The prompt path itself is a chain of per-card Prefill objects. Stage 0 runs on the caller's thread; each later stage runs on a thread via next_ (prefill.hpp:151-156, prefill.cpp:1965ff, run+drain 4074-4085).
- A single-chunk prompt (<=512 tokens) runs the 4 cards strictly one after another.
- on_chunk on the last stage's thread builds the MTP draft KV, prints the PP lines and takes checkpoints (generate.cpp:7309-7372).

5) After the first token (a solo verify window on `ver`), BADM (10962-11004) calls copy_to_slot (8312-8346). Per stage, sequentially, it does conversation_checkpoint_save/restore plus conversation_kv_save/restore with cudaDeviceSynchronize between them. Those are synchronous device-to-host-to-device copies through host vectors (transfer() = cudaMemcpy, conversation_snapshot.cpp:112-120; conversation_state.cpp:154-198). The engine log shows 'copied in 86.9-88.6 ms' for every admission.

6) Prompt-buffer loan on CUDA3. CUDA0-2 keep their own prompt buffers, but CUDA3 'prompt path borrows 166 of its 6144 slots (0.39 GiB)' (strata-unsloth-ud-iq4_xs.log:32193-32197; own-buffer rule at generate.cpp:6441-6465). So every admission lends, then refills about 0.39 GiB into CUDA3's cache, read from the mmap'd GGUF.

7) Server side. One admission at a time under self.ctl (server.py:741), held from BGEN to BADM (server.py:1290-1330). pick_slot fills holes inside a group first (1424-1455). No change is needed there for the recommended variant.

8) Measured admission cost (log 32260-32330): prompts of 145/248/264/273 tokens are read in 366/451/492/496 ms. That is single-chunk and serial over the 4 cards. Add ~88 ms for the copy, ~12 ms for the first-token window, and a drain plus pipeline refill bubble of roughly 2x3 group-steps x 6.6 ms ≈ 40 ms.

Modelled with 16 slots, a 603 tok/s decode ceiling and 512 tokens per request: throughput = 603 / (1 + 603·R/512), where R is the all-slot stall per admission. The measured plateaus give:
- short prompts: 360 tok/s means R ≈ 0.57 s
- 1.2K prompts: 250 tok/s means R ≈ 1.2 s
The log components above add up to the short-prompt figure, so the model is consistent.

### Nötige Änderungen

- **src/program/generate.cpp: read_part lambda (9721-9795)** (medium, ~100-140 lines): New piped branch, gated (env STRATA_PIPE_OVERLAP=1, then default-on once validated): when piped && batch_on(), start sp.run(ids+a0, b0-a0, a0, e) on a worker thread (std::async). On the main thread, loop: pump(true); scan in_lines for BSTOP (copy of 9733-9745); sleep about 50 us. Then join and pipe_drain() before returning, so lend/refill, checkpoint_at, read_windows, the first-token window and copy_to_slot all still run with the pipeline drained. Gate it off when any pf_part lends (p.first>=0), when experts are not all resident or ar_on(), with kv_mode != 0 (--kv-resident), with images, and while adaptive swaps are pending.
- **src/program/generate.cpp: pump (8545-8646)** (small, ~20 lines): Add an 'overlap' flag. While it is set: skip apply_pending(false) at group start (8622), so no adaptive swap rewrites cache slots the prefill reads; and do not reset progress().busy / the timing line when all groups go idle mid-read (8633-8644). Keep the BGEN drain at 8659 so the pre-read work (prompt cache, copy_from_slot, park_current, kvg_trim, lend) stays serial.
- **src/core/verify.cpp: init (600-606) plus a setter** (small, ~15-25 lines): Create the batch stream cs_ with high priority (cudaStreamCreateWithPriority, as mtp.cpp:424 already does), at least for the stage verifiers of a split running --batch. That way the short, latency-bound decode kernels are scheduled ahead of the prefill's MMQ blocks on the same GPU. Leave the ext_stream_ (pipeline-windows) path alone.
- **src/program/generate.cpp: copy_to_slot / copy_from_slot (8312-8390), plus a new helper in src/core/conversation_state.cpp / conversation_snapshot.cpp** (medium, ~120-200 lines): Device-to-device session copy: gdn_state, ple_hist, idx_tail/dead/block_pos/pooled, and the QSA K/V pools (int8 data plus scales) up to `upto`, with cudaMemcpyAsync on each stage's own device. Issue all 4 stages first, then sync once. Keep the existing host path as the fallback for kv_mode 1/2 (--kv-resident) and the q4 KV ring. Expected 88 ms down to a few ms per admission.
- **src/program/generate.cpp: own-buffer rule (6441-6465) / config** (small, ~20-40 lines or config only): Stop the per-admission CUDA3 loan and refill in piped batch mode. Option 1: count the batch slot sessions into CUDA3's headroom so it keeps its own 0.39 GiB buffers (verify the cache still holds 6144/6144). Option 2: config-only, e.g. a smaller --max-context for the slots or --prefill 256 on that card. This is also the precondition for exact and thread-safe overlap: otherwise decode windows on CUDA3 run 166 experts on the CPU pool through the non-thread-safe ExpertSource during the read.
- **src/program/generate.cpp: windows_ok (9413-9419)** (small, ~5-10 lines): In piped batch admissions, read the short tail segment (the turn header after turn_at, <= --short-read) through the batched path instead of read_windows. read_windows uses the shared stage Verifiers, which forces a drain. Alternatively, recommend --short-read 0 for batch servers.
- **tools/batch_interleave_test.py, tools/batch_test16.py** (medium, ~100-150 lines Python): Piped split variant (--extra '--layer-split 12,24,36 --batch-groups N --pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow', STRATA_IQ_MT_MIN=1): 12-16 slots decoding while prompts of 150, 1.2K and 5K tokens are admitted, every slot compared with its solo tokens. Step 4 (BYIELD) must be skipped or expected to be ignored in piped mode, because BYIELD is inert there.
- **serve/server.py (1230-1320)** (small, ~10 lines): Optional only: stop sending BYIELD when slot_gs (pipelined groups), since the engine ignores it there. Optionally send the next BGEN without waiting for BADM; that is negligible. Admission serialization (ctl lock, 741) can stay.
- **docs/BATCHING.md 'How the server uses the slots' / Limits** (small): Document that decode-between-chunks and BYIELD do not apply with --batch-groups, and describe the new overlap.

### Blocker und Invarianten

- Decode-share experiment 3 is void as planned. In piped mode read_part returns a single sp.run (generate.cpp:9723), so STRATA_BATCH_DECODE_SHARE (9708-9711, 9780-9786) and BYIELD (9731-9779, discarded at 8670) never run. Q3-b12-share4.json uses auto groups and is therefore piped (3906). It can only differ from baseline by noise. Testing the share at all needs --batch-groups 1, which loses the pipeline.
- Interleaving alone gives no throughput. Reading in pieces with decode in between serializes the same GPU work in a different order: only TTFT and fairness change. Only work running concurrently on idle or underused GPUs raises tok/s.
- Stage-exclusive overlap ('prefill on card k, decode on the others') gains about nothing. Every decode token needs all 4 stages, so while a single-chunk prompt occupies card k for its ~75-120 ms, all groups pile up behind k. A serial 4-stage pass then stalls the pipeline about as long as today. The gain only comes from true same-GPU co-execution: gain ≈ (1-f) x read time, where f = the fraction of decode speed lost on a card while the prefill runs on it. f is unknown and is the first thing to measure.
- The stage Verifiers are shared between the batch groups and the admission's solo windows: read_windows for segments <= --short-read 64 (9413-9419, 9535ff), the first-token window, and pipeline windows. These must run with the pipeline drained, so overlap is limited to sp.run.
- Pad rows: a group window spans its slots up to the last active one and writes the state of inactive slots below it (8613-8618). Never copy_to_slot or read into a slot while its group is in flight. The drain before BADM preserves this.
- The expert cache must not change while the other side reads it. This covers lend/refill/res_upload (9653-9702, 9627-9649), apply_pending in pump (8622) with --adapt-async 1 as in the production config, and VRAM elastic. ExpertSource is not thread-safe (prefill.cpp:3086-3087 comment). CUDA3 currently lends 166 slots per admission (log:32197). Gate the overlap on 'no stage lends, all experts resident, no pending swaps'.
- --kv-resident (KV streaming, in the production config) and conversation parking: the slot and main sessions would stream concurrently. Gate these off in the prototype; audit before enabling.
- Graph capture uses cudaStreamCaptureModeThreadLocal (verify.cpp:2349), so capturing a new row layout on the main thread while the prefill thread issues CUDA calls is allowed. But every new layout costs 20-30 MiB of VRAM (instantiate_evicting, verify.cpp:2498-2521), and CUDA3 is tight (~0.9 GB free at start), so eviction and recapture stalls are possible.
- stdout: PP lines from the last stage's thread (7322) and BT/BDONE lines from the main thread will interleave. Each is a single printf, and the server already demultiplexes BT/BDONE during admissions (non-piped interleave). The watchdog/progress busy flag is reset by pump at 8643 and must not drop while a read is in flight.
- Exactness: with all experts resident, no loan, no swaps and --pcie-frac 0, the change is timing-only, so batch == solo stays bit-exact (same graphs, same prefill chunks). With a loan, windows during a read run the lent experts on the CPU, which rounds differently, as the docs already state for the non-piped path.
- TTFT trade-off: an overlapped read shares its GPUs, so the admitted request's own TTFT rises. Admissions stay one at a time: server ctl lock (server.py:741), plus a single main session and a single Prefill chain per card in the engine.
- Real vLLM-style chunked prefill inside batch windows is blocked:
- kVerifyMaxT = 8 rows per window (verify_kernels.hpp:21, referenced in 15 places under src/kernels)
- grouped rows of one slot are refused with a layer split (verify.cpp:2545-2548)
- one graph pair per row layout (capture_batch keyed by rows, verify.cpp:2346-2376)
- window arithmetic differs from the prefill path, so exactness against solo GEN breaks
- the BGEN/BADM protocol would need redesigning
With only about 4 spare rows per group-step, a 150-token prompt would take ~38 group-steps (~1 s TTFT) and a 1.2K prompt ~8 s. Estimated 8-15 days, not the cheap path.
- Concurrent admissions (two prompts read at once) would need a second main session plus a second prompt path per card (+0.39 GiB buffers and a session per card), a relaxed server ctl lock, and duplicated prompt-cache/checkpoint bookkeeping. Little gain over the overlap, because the GPU work is identical.

### Erwarteter Gewinn (Analyst)

Model: throughput = 603 / (1 + 603·R/512), where R is the all-slot stall per admission. Measured: R ≈ 0.57 s (short prompts, 360 tok/s) and R ≈ 1.2 s (1.2K prompts, 250 tok/s).

(1) Fixed-cost cuts alone: device-to-device slot copy (-80 ms), no CUDA3 loan/refill (-20..50 ms), no bubble from the header windows.
- Short: R ≈ 0.45-0.47 s, about 390-400 tok/s (+8-11 %).
- Long: R ≈ 1.07 s, about 265 tok/s (+6 %).

(2) Adding the overlapped batched read. The batched part is about 250-350 ms of the short admission and about 1.0-1.1 s of the 1.2K admission. Saving (1-f) x that time:
- Short: f = 0.3-0.6 gives R ≈ 0.25-0.34 s, about 430-470 tok/s (+20-30 % over 360).
- Long: f = 0.5-0.8 gives R ≈ 0.55-0.87 s, about 300-365 tok/s (+20-45 % over 250).

The co-execution factor f is the main uncertainty. Grounds for optimism: both workloads are latency-bound (decode 6.6 ms per stage-step against a ~1.5-2 ms bandwidth floor; a short prompt takes ~90 ms per card for only ~150 tokens). Grounds for caution: 512-token chunks are denser, so f is likely higher for long prompts.

Hard ceiling for this lever: R → 0 gives 603 tok/s (today's pure batch decode). So lever A cannot reach vLLM's ~710 on short prompts by itself; that needs the decode ceiling (levers 5/6). On 1.2K prompts it can roughly close the gap to vLLM's ~400 only if f ≤ ~0.5; otherwise prefill efficiency is the next limit (Strata ≈ 2.1-2.5 ms of GPU time per prompt token summed over cards, against ≈ 1.1 for vLLM). Single-user speed is unchanged: the solo path is untouched and the overlap is gated on piped && batch_on().

### Erster Prototyp-Schritt

Under one day, env-gated, two measurements.

(0) Five minutes, no code: confirm the decode share is inert. Compare Q3-b12-share4 with a share-0.5 run of the same config, or grep the engine log for 'the prompt was read in N parts'; it never appears in piped mode. Then drop experiment 3 or re-scope it.

(1) Code: in read_part (generate.cpp:9721-9723), add a branch for piped && batch_on() && getenv("STRATA_PIPE_OVERLAP"):
- run `auto fut = std::async(std::launch::async, [&]{ std::string e2; bool ok = sp.run(ids.data()+a0, b0-a0, a0, e2); return std::make_pair(ok, e2); });`
- on the main thread, loop `while (fut.wait_for(0s) != ready) { if (!pump(true)) fail; scan in_lines for BSTOP; sleep_for(50us); }`
- then `pipe_drain()` and return the result
- add a bool that makes pump skip apply_pending and the idle reset while the read is in flight

(2) Run it with 12 and 16 slots, --short-read 0 (the whole prompt goes batched), no --kv-resident and no --adapt-async, and confirm in the log that no stage lends (otherwise add --no-prefill-borrow, or test at 12 slots where CUDA3 has more headroom).

(3) Measure:
- bench_concurrency.py with --prompt-tok 100 and the 1.2K prompts at --max-tok 512 and 16/24 users, against the same build without the env
- f directly: with STRATA_TRACE, the batch rows/s logged during reads against pure decode at 603
- exactness: batch_test16.py plus one admission-while-decoding run against solo with STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 1000000

If f ≤ ~0.6, build the full version: stream priority, device-to-device copy, no CUDA3 loan, tests, upstream PR. If f ≈ 1, stop: the lever is then worth only the fixed-cost cuts (~+10 %), and effort should go to the decode ceiling and prefill efficiency instead.

## D: quick wins without deep engine work (env/CLI knobs, --batch-groups, data-parallel instances, server.py scheduling)

### Mechanismus heute

Measured logs confirm the cost model. In the pipelined batch path, every new request stops all slots for its whole prompt read.

(1) What piped means. --batch-groups defaults to auto on a layer split and picks one group per stage (generate.cpp:3904-3915), so 16 slots become 4 groups of 4. That sets piped=true (generate.cpp:8520). pump() (8545-8645) then pushes the groups through the 4 stages with batch_launch/batch_poll (verify.cpp:2783-2880). A window holds at most 8 rows: rows[8] in batch_launch, and stage_batch requires S <= max_t_ (verify.cpp:2526). The g3 patch therefore caps slots at 8 x groups (generate.cpp:3888-3899, 3916-3922).

(2) Admission. The server takes one admission at a time under its ctl lock (server.py:1118-1157). In the engine, every stdin line except BSTOP first drains the whole pipeline (generate.cpp:8658-8659). The BGEN prompt then runs through the normal GEN 1 path. read_part returns sp.run(...) for the whole prompt when piped (generate.cpp:9723), so it is never split into chunks with decode windows in between. As a result, STRATA_BATCH_DECODE_SHARE (9708-9711, used 9780-9786) and BYIELD (dropped at 8670, never taken in piped mode) do nothing on this setup. The logs agree: E1 (share 1.0) gave 246.3 tok/s and F2 (share 2.0) gave 246.0. Even where it works (groups=1), the decode share only time-slices prefill against decode. The GPUs do the same work one after the other, so it can only shift waiting between users, never raise throughput. After the read, copy_to_slot (8312-8348) copies the state into the slot through the host: conversation_checkpoint_save/restore and conversation_kv_save/restore with std::vector buffers, plus 3 cudaDeviceSynchronize per stage. That takes 36-40 ms in the logs.

(3) Measured cost per admission (G16 engine log, 4x4):
- 162-token prompt: read in about 455 ms, copy 36 ms, first token 12 ms, plus drain and refill. About 0.57 s per admission, during which all 16 slots stall. That is 18.1 s of 45.3 s wall time (40%).
- 1.2K-token prompt: read in about 1083 ms, copy 40 ms. About 1.17 s per admission, 58% of wall time.
- Pure decode runs one group-step per 6.63 ms, about 603 rows/s.
- Model: throughput ~= L / (L/R + T_adm). With L=512 and R=603 it predicts 361 and 253 tok/s; measured 357/361 and 248/253.

(4) Where the fixed cost of a short prompt read comes from (about 360 ms of it). CUDA0 and CUDA3 do not have the free VRAM the own-buffer rule needs (buffers + 1.5 GiB, generate.cpp:6446-6460). So they lend 170 and 167 expert-cache slots to the prompt path ("borrows 170 CUDA0 cache slots", G16 log). Before the next window they refill them (refill 9627-9645) with fill_slot_queued: a cudaMemcpyAsync on stream 0 from the pageable GGUF mmap (expert_cache.cpp:723-737). The engine log shows about 0.7 GB of expert reads per short prompt and about 1.8 GB per long one ("704 MB of expert reads", "1827 MB"), all on the critical path. Each prompt is also read in 2 parts at the turn boundary, with a checkpoint at that boundary (9829-9905).

(5) Last stage not fully resident. CUDA3 holds 6141 of 6144 of its layers' experts. It therefore lacks the zero-doorbell graph (verify.cpp:629-650, 1657-1673), and batch_poll serves 12 per-layer doorbells from the host (verify.cpp:2820-2860). The configs use "layer_split": "auto" without --trim-stage-weights, so every card carries the dense weights of all 48 layers (4.39 GiB). With auto, CUDA0 can only be trimmed if the split is given explicitly (generate.cpp:2624-2640).

(6) Rows per group. From Q2 (2 groups of 8: 2060 group-steps, 16351 rows in 60.9 s, same 0.57 s admissions), a stage-step costs about 2.8 ms + 0.95 ms per row: t4 = 6.6 ms, t8 ~= 10.4 ms. Two groups therefore leave 2 of 4 GPUs idle: pure decode 16/(4 x 10.4 ms) ~= 385 tok/s, against 603. Measured: 266-272 tok/s, against 357-365 for 4x4 (-25%).

(7) server.py:
- pick_slot ignores held prefixes in piped mode (server.py:1424-1440). Its comment says "the engine keeps no pipelined slot as a conversation cache", but engine 0.1.41 does keep them (generate.cpp pump, "a pipelined slot is a cache again"). Multi-turn chats therefore re-read their whole history while all slots stall.
- The server still sends BYIELD (server.py:1234, 1314). The engine ignores it but drains the pipeline for that line.
- The CPU pool honours the inherited affinity (pool.cpp:232-234). Without taskset, two instances both pin 63 workers to logical CPUs 1-63.

### Nötige Änderungen

- **linux/configs/G16-*.json (config only)** (small, 2 config lines): Set "layer_split": "12,24,36" (explicit, so CUDA0 is trimmed too) and add --trim-stage-weights. This frees about 2 GiB of foreign dense weights per card. All 4 stages should then keep their own prompt buffers (no loan, no refill of 0.7-1.8 GB from pageable mmap per admission), and CUDA3 should reach 6144/6144 with the zero-doorbell graph. Check the log for: 'loads the dense weights of layers 0-11 only', 4x 'keeps its own prompt buffers ... no loan', 4x '(100% VRAM resident: zero-doorbell graph)', and a per-request blob-read count that stays flat.
- **linux/series_q.sh, configs Q3-*/G*-share*** (small): Drop STRATA_BATCH_DECODE_SHARE from piped experiments: generate.cpp:9723 bypasses it, so Q3-b12-share4 (12 slots, auto 4x3) will only reproduce H12 (~360 tok/s). Q2 (2x8) is already measured at -25%. Do not pursue fewer groups.
- **configs after step 1** (small): With the freed VRAM, keep 4 groups and raise rows per group: "parallel": 24 (4x6), optionally --max-context 16384 (slot sessions ~0.15 GiB/slot/stage at 32K int8 KV). Keep all 4 cards at 100% residency and own buffers; the earlier G32 lost because of VRAM pressure.
- **bench_concurrency.py payload / one-shot clients (server.py:970-972 already maps it)** (small): Send "strata_checkpoint": false for one-shot requests (ckpt=0). The prompt is then read in one part, with no turn checkpoint and no separate header windows. Do not use for multi-turn agents.
- **serve/server.py:pick_slot 1424-1440 (+ serve/test_slot_alloc.py)** (small, ~10-15 lines + test): In piped mode (slot_gs), prefer a free slot whose slot_held is a prefix of at least SLOT_PREFIX_MIN tokens, before the hole-filling order. The engine re-checks sl.cached, which pad rows clear (generate.cpp pump group start), so this is safe. Fix the stale comment. Big win for multi-turn agent traffic, none for the one-shot benchmark.
- **serve/server.py:1227-1234 and 1310-1315** (small, ~2 lines): Do not send BYIELD when self.slot_gs (piped). The engine cannot take it (generate.cpp:9723, 8670), and it costs an extra pipeline drain (8658-8659).
- **src/program/generate.cpp:copy_to_slot 8312-8348 / copy_from_slot 8350-8395 (optional)** (small-medium, ~60-100 lines): Replace the host round trip (conversation_checkpoint_save/restore + conversation_kv_save/restore into std::vector, 3 device syncs per stage) with same-device cudaMemcpyAsync D2D of GDN state/conv, PLE history, indexer tails and the QSA K/V prefix, issued on all 4 devices concurrently with one sync each. Byte-identical result.
- **dp_launch.sh, configs Q1-dp2-inst*.json (only if DP2 is pursued)** (medium, 1-2 days): Give each instance its own CPUs: taskset -c 0-31,64-95 / 32-63,96-127 (pool.cpp:232-234 honours the mask). Replace --resident-experts with --mmap-experts (one shared page-cache copy) or cap it with --resident-budget-gib. --resident-experts pins a private copy of the experts each process cannot keep on its GPUs (~26-30 GB each), and --shared-expert-arena is refused together with mmap (generate.cpp:2034-2037). Raise or remove MemoryMax=90G: the 93.7 GB GGUF (PLE table + experts) page cache is charged to the scope. Use an explicit split of 24 plus --trim-stage-weights. Production also needs a sticky balancer: a small proxy hashing the system prompt and first user message, with a least-outstanding fallback (~150-250 lines), plus metrics aggregation for the G3 monitoring.

### Blocker und Invarianten

- In piped mode (the default on a layer split) an admission always drains the pipeline (generate.cpp:8658-8659) and reads the whole prompt in one sp.run (9723), with all slots stalled. No config knob changes this. Real overlap (prefill on idle stages, or chunks mixed into windows) is lever 4 and needs engine code.
- Time-slicing (STRATA_BATCH_DECODE_SHARE, BYIELD) cannot raise throughput even where it works. It only moves waiting time between users, because prefill and decode still run one after the other on the same GPUs.
- A window holds at most kVerifyMaxT = 8 rows: batch_launch rows[8] (verify.cpp:2786) and stage_batch S <= max_t_ (verify.cpp:2526). The g3 patch caps slots at 8 x groups. --batch-mtp is refused on a split (generate.cpp:3874), and the piped path builds one row per slot (8521-8525).
- A stage keeps its own prompt buffers only if free VRAM >= buffers + 1.5 GiB (generate.cpp:6446-6460). CUDA0 can only be trimmed with an explicit --layer-split (generate.cpp:2633-2640). The zero-doorbell graph needs 100% residency per stage (verify.cpp:629-650); one missing expert switches the whole stage to the doorbell path.
- Exactness: trim and own buffers change which experts a card keeps resident, so output need not match earlier runs bit for bit (docs/BATCHING.md, #340 note in generate.cpp). Re-run tools/batch_test16.py and tools/batch_interleave_test.py with --pcie-frac 0 --adapt-every 1000000 and STRATA_IQ_MT_MIN=1 (batch = solo).
- VRAM: slot sessions take ~0.15 GiB per slot per 12-layer stage at 32K with int8 KV. 32 slots previously lost (G32: 340 tok/s), because sessions pushed experts out or forced loans.
- DP2: each 2-GPU instance holds only ~49% of the 24,576 experts (~56% with trim). Prompt chunks then stream or stage the non-resident experts; chunks of 1024 tokens or more stream all of them (prefill.cpp:142-146). The CPU pools and the 4-channel DDR4 are shared, and single-user speed per instance drops below today's 121 tok/s. That conflicts with the user's goal of keeping single-user speed.
- DP4 with --batch-mtp: about 0.9 GB per slot for drafters (generate.cpp:8521-8525 warning) plus 4.4 GiB of untrimmed dense weights per card leaves room for ~20% of the experts. RAM cannot hold 4 pinned copies of the experts the GPUs don't keep. Not viable.
- Avoid STRATA_DF_BRANCH: on Linux it hung windows when NVML queries came in (#905, verify.cpp:614-620), and the G3 monitoring polls the GPUs.

### Erwarteter Gewinn (Analyst)

Model fitted to the logs: throughput ~= L / (L/R + T_adm), with L = 512, R = 603 rows/s and T_adm = 0.57 s (162 tokens) or 1.17 s (1.2K tokens). It predicts 361 and 253 tok/s; measured 357-365 and 248-253.

- Trim + explicit split (own buffers, no refill, CUDA3 at 100%): removes the ~0.7 GB (short) / ~1.8 GB (long) pageable re-read and part of the 2-part overhead. Estimated T_adm 0.57 -> ~0.43-0.47 s and 1.17 -> ~0.9-1.0 s. Short ~361 -> 390-415 tok/s (+8-15%), long ~250 -> 275-295 (+10-16%), plus 0-8% decode if CUDA3's doorbell path is the slowest stage.
- ckpt=0 for one-shot requests: -30-60 ms per admission, +3-6% short.
- 24 slots in 4 groups of 6 once VRAM allows: pure decode ~603 -> ~700 (t(6) ~= 8.5 ms); through the server about +8-10% short.
- D2D slot copy (optional code): -25-30 ms per admission, +3-5%.
- Combined realistic ceiling for lever D: short ~430-470 tok/s, long ~290-310 tok/s, against vLLM's 710 and 400. vLLM level needs admissions that overlap decode on the hardware (lever 4).

Measured or predicted negatives:
- 2 groups of 8: measured -25% (266-272 vs 357-365), explained by the per-row cost (2.8 + 0.95·rows ms per stage-step) with half the GPUs idle.
- Decode share: 0% (inert in piped mode).
- DP2: expected 250-350 short / 170-240 long. Prompt reads with ~half the experts non-resident push T_adm to ~1-1.5 s short and ~2-3 s long. CPU misses lower R per instance, and single-user speed drops to roughly 70-90 tok/s. It could only win if prefill and decode on 2 GPUs stayed as fast as today, which the residency numbers rule out.
- DP4 + --batch-mtp: not viable.

### Erster Prototyp-Schritt

Copy configs/G16-pf512-share2.json to D1-trim.json and change three things: set "layer_split": "12,24,36", add "--trim-stage-weights" to args, and remove the share env. Run the same G16-short and G16-long sweep (u=16, 24; --prompt-tok 100 / default 1.2K; max-tok 512).

Check the engine log for:
- 'CUDA0 loads the dense weights of layers 0-11 only'
- 4x 'keeps its own prompt buffers ... no loan'
- 4x '(100% VRAM resident: zero-doorbell graph)'
- per-request 'blob reads from the file' no longer rising by about 260-340 per prompt
- 'prompt 16x tokens ... read in X ms' (today ~455 ms; target ~300-350)
- 'copied in' (~36 ms)
- the 'strata batch (pipelined ...)' rows/s

Add one extra run with STRATA_VERIFY_PROFILE=1 to get per-stage ms/window and see whether CUDA3 is the slowest stage. If the 4 cards then have >= ~1.5 GiB spare, repeat with "parallel": 24 (and --max-context 16384 if needed), and once with "strata_checkpoint": false in the bench payload. About 1-2 h of GPU time; GPUs were not touched for this analysis.

## Gegenprüfung: C: per-window overhead of the pipelined batch windows (pump / batch_launch / batch_poll / commit gra (optimistic, 5–14 Tage)

My estimates (no profile or nsys data exists yet):
- Pipelined self-commit: removes about 100 nodes, ~108 MiB of re-read state and one graph launch per stage. Realistic saving 0.2-0.5 ms per tick, not 0.4-0.8.
- Async sampling and moving the PLE gather off the launch path: 0.1-0.4 ms, and only where that stage is the slowest. Sampling helps only at temperature > 0, so it shows up in server runs, not in batch_test16.
- Layer-split rebalancing: 0-0.3 ms. One layer is about 0.5 ms, coarse next to the head's ~0.6 ms.
- Multi-slot kernels: 0.2-0.5 ms. The analysis's own 0.41 ms/row slope caps what removing per-row nodes can save.

Realistic tick: 5.3-6.1 ms instead of 6.63, so 650-750 tok/s pure decode (+8-25%). The cheap subset without multi-slot kernels gives roughly +5-15%.

At server level (Amdahl, decode about 60% / 41% of wall time):
- Short prompts: 360 to roughly 380-415 tok/s (+5-15%).
- 1.2K prompts: 250 to roughly 258-272 tok/s (+3-9%).

Lever C alone does not reach vLLM (710/400). The upside is larger only if the 1-row-per-group cross-check shows that most of the ~2.5 ms batch-vs-solo gap is path overhead rather than kernel scaling or memory throttling.

**Nicht bestätigte Behauptungen:**
- Row-proportional work is already near bandwidth (0.41 ms per row), so all gains come from the roughly 5 ms fixed intercept. — The fit uses two points from separate runs (8 vs 16 slots), which is too little to separate the terms. The claim also contradicts the analysis's own -0.3 to -1.0 ms for multi-slot kernels: those kernels remove per-row nodes, and per-row nodes sit in the slope, not the intercept. If the slope really is at bandwidth, the multi-slot savings at 4 rows are capped at about 0.1-0.3 ms.
- STRATA_VERIFY_PROFILE shows how much of the 6.63 ms is graph work vs commit vs host. — Each profiled section adds a 1-thread gpu_stamp kernel (verify_kernels.cu:1226-1244), roughly 300 extra serialized nodes per stage. Profiling also turns off the shared-expert side stream (verify.cpp:1280 and 1487, '!prof_on_') and does a synchronous cudaMemcpy per window in batch_poll (collect_profile, 1868-1871). It does not stamp the commit graph at all. It distorts exactly the latency-bound regime under study.

**Übersehene Blocker:**
- HIP: one_token_self_commit() is hard-disabled under STRATA_USE_HIP (verify.cpp:173-182), and sh_stream is also off on HIP. Upstream's main pipelined-batch rig is 4x R9700 (BATCHING.md:40). A batch self-commit or new multi-slot kernels must therefore be CUDA-gated or validated on AMD, which cannot be done on this box. That lowers the PR's value and acceptance odds upstream.
- The SYCL port keeps its own copy of the batch path (sycl/src/core/verify.cpp contains batch_launch/capture_commit_batch, plus sycl/src/kernels/cuda/verify_kernels.dp.cpp). Engine changes diverge from it and need re-migration per INTEL_ARC.md.
- batch_step also runs inside the pipelined engine during prompt reads (decode share, generate.cpp:9784) through run_slot_rows + commit_slots on the same Verifiers. A STRATA_BATCH_SELF_COMMIT hack that is not keyed by caller or variant would run the self-committing graph and then commit_bm_ again: a double commit that corrupts conv/GDN state. batch_test16 (short prompts, no chunked reads) would not catch this; batch_interleave_test would.
- The non-MTP batch_step never advances next_slot (generate.cpp:8399-8423: it is advanced only when batch_mtp). With 16 slots, only the first 8 active slots decode between prompt chunks while slots 8-15 starve. This is Lever A, but it distorts the server-level numbers Lever C is judged against.
- Sampling: the split sampler refuses to run under stream capture (sampler.cu ~1275, !stream_capturing) and takes one SamplerParams per launch. A batched sampler with per-row params needs a new kernel interface that keeps the Philox(seed,position) draws identical, plus an exactness run with batch_test.py --keys temperature=0.7. batch_test16 is greedy-only, so it never exercises this path, and the 603 baseline excludes the per-tick sample sync that the temperature-0.7 server runs pay.
- Measurement confounders: (a) the profiler perturbation described above; use cudaEventRecord between the exec_bm_ and commit_bm_ launches, plus nsys --cuda-graph-trace=node (nsys is installed at /usr/local/bin/nsys). (b) GDDR6X memory thermal/clock throttling: in the pipelined batch every 3090 is busy 100% of the time, against about 25% per card in solo. Check clocks with nvidia-smi before attributing the gap to software overhead.
- A cheap zero-code cross-check is missing. Solo without MTP (81 tok/s) costs about 3.1 ms per stage per token for one row, while the batch path costs 5.8 ms at 2 rows. A run with --batch 4 --batch-groups 4 (one row per group) isolates batch-path overhead (commit graph, host, lockstep) from multi-row kernel scaling. This tells whether Lever C or a kernel-efficiency lever owns the gap.
- The ~5 ms intercept is not explained by the items listed. Dense reads (~1.1-1.4 ms), the head (~0.6 ms), the commit (~0.3 ms), host (~0.2-0.4 ms) and imbalance (~0.3-0.5 ms) leave about 1-2 ms unattributed. That remainder may sit in dense GEMV/MoE-grouped kernel efficiency at n=4 or in hc-read kernels, which no proposed change touches.

I checked every cited file:line, and the code references are accurate. The analysis reads the mechanism correctly: pump, two graph launches per window, the synchronous stage-0 PLE gather, the per-row sample+sync, the commit graph re-running the GDN recurrence, and zero-doorbell on all stages. The central refactor is sound on CUDA. I traced the kernels (verify.cpp:1017-1031 vs 2410-2421; verify_kernels.cu:207-290, 912-966): for 1-row groups, a batch self-commit hits the same gdn_step_norm_multi_kernel<false,false> template the commit already uses, with y bitwise equal to the all_out path. The QSA open question is resolved because kStepPos == the commitb position (qsa.cu:710). The PLE history is already advanced in place (verify.cpp:913).

Where the analysis is too optimistic or incomplete:
(1) The savings per item are not grounded in a measurement, and the slope/intercept argument contradicts the multi-slot-kernel estimate.
(2) The proposed first measurement, STRATA_VERIFY_PROFILE, adds about 300 stamp kernels per stage, turns off the shared-expert stream, syncs once per window, and leaves out the commit graph.
(3) Not mentioned: HIP disables one_token_self_commit, and upstream's pipelined rig is AMD.
(4) Not mentioned: the pipelined engine itself calls batch_step during prompt reads, so a caller-agnostic env hack would double-commit.
(5) Not mentioned: the greedy batch_test16 baseline vs temperature-0.7 server runs.

Recommended first day once the GPUs are free:
(a) Zero code: run batch_test16 and a --batch 4 --batch-groups 4 run (one row per group) and compare their ticks with solo's ~3.1 ms per stage. Log nvidia-smi clocks, power and memory temperature during the runs. Also try --ple-io ram or mmap to bound the PLE gather.
(b) About 30 lines: put a cudaEventRecord around exec_bm_ and commit_bm_ in batch_launch, and steady_clock timers in stage_batch and sample_rows. Or use nsys --cuda-graph-trace=node.
(c) Self-commit prototype gated as (CUDA && one_token_self_commit && pipelined caller), with a separate bkey bit. Validate with batch_test16/32, batch_interleave_test, parking_test, and batch_test.py --keys temperature=0.7.
(d) Small fix for Lever A: rotate next_slot in non-MTP batch_step.

Effort: about 2-4 days for instrumentation, self-commit (also applicable to non-MTP batch_step), async sampling event and PLE reorder. The full lever with multi-slot kernels and parity tests is 5-14 days, without HIP/SYCL validation.

Files: /media/raid0/StrataQ38FN/Strata/src/core/verify.cpp, /media/raid0/StrataQ38FN/Strata/src/program/generate.cpp, /media/raid0/StrataQ38FN/Strata/src/kernels/cuda/verify_kernels.cu, /media/raid0/StrataQ38FN/Strata/src/kernels/cuda/sampler.cu, /media/raid0/StrataQ38FN/Strata/src/kernels/ngram.cpp, /media/raid0/StrataQ38FN/Strata/include/strata/core/verify.hpp, /media/raid0/StrataQ38FN/Strata/docs/BATCHING.md, /media/raid0/StrataQ38FN/Strata/sycl/src/core/verify.cpp, /media/raid0/StrataQ38FN/linux/results/batch_test_16.txt, /media/raid0/StrataQ38FN/linux/bench_concurrency.py

## Gegenprüfung: B: MTP drafts inside pipelined batch windows with a layer split (--batch-mtp across 4 GPUs, one draf (optimistic, 8–16 Tage)

Engine-only decode with 16 slots: somewhere between -10% and +15% against today's 603 tok/s. The sign depends on the unmeasured 8-row stage time. If T8 ≈ 8.2 ms and alpha ≈ 0.65: v1 ≈ 620-650, v2 ≈ 650-690. v3 (batched drafter plus one layer moved off CUDA3) ≈ 700-760. If T8 ≥ 10 ms, B costs throughput unless v3 is done. Through the HTTP server at 16+ users, the admission stall (≈0.5 s per short prompt, never interleaved in the pipelined path) caps the gain at about 0-8% for short prompts (≈360 → 370-390) and 0-5% for 1.2K prompts. At 4-12 users with variable S (2 slots x 2 rows = 4-row windows), the gain is larger and less risky: about +20-30% engine-level (≈345 → 420-450 at 8 users), +15-25% through the server. Alone, B does not get to vLLM's 710/400. It only matters after lever A (interleaved prefill / no pipe_drain stall) has moved the server toward the 603 engine figure. Single-user speed is unaffected: the solo path keeps spec-4 MTP.

**Nicht bestätigte Behauptungen:**
- Per-slot drafter VRAM is unknown (the #1413 warning says ~0.9 GB per slot). — It was measured on this model: /media/raid0/StrataQ38FN/logs/E-H10-batch2.txt and agent logs show 'strata mtp: shared draft weights, 164 MiB of private state and buffers' at 128K context, int8 K/V. At 32K that is about 60-70 MiB (the draft K/V is 2 heads x 256 x 2 bytes per cell), so 16 drafters take about 1-1.1 GiB on CUDA3, roughly 450 experts' worth. The 0.9 GB figure is too high.
- T_stage at 8 rows is ~8.2 ms, measurable without new code with tools/batch_test32.py (32 plain slots, 4 groups x 8 rows). — That test was already run. /media/raid0/StrataQ38FN/linux/results/batch_test_32.txt gives 547.1 tok/s from the first batch token, i.e. 4800 tokens in ~8.8 s, about 14.7 ms per 8-row group-step, against 6.7 ms at 4 rows (batch_test_16.txt, 602.6). The run is confounded: 32 slot sessions x 0.15 GiB evict experts, so solo-run hit rates fall to 94.6% (98057/103680, ~3000 RAM reads per request) from 99.4%, and the stages lose their zero-doorbell graphs. The only existing 8-row data point therefore contradicts the linear 0.4 ms/row extrapolation, and repeating the same test proves nothing. It must be re-run with a smaller --max-context (e.g. 16384), so that 32 sessions take the VRAM that 16 take at 32K.
- Acceptance alpha 0.65-0.8 ('BATCHING.md shows 140/172 = 81% greedy'). — docs/BATCHING.md:73: 140 of 172 is the acceptance of a request returned from a slot to the solo path, not a greedy first-draft figure for this model. Over the whole engine log on this box, acceptance is 35438/58996 = 60% (spec 4, min-p 0.5, mostly temperature 0.7); the analysis's own 295/453 = 65% is one request. For a single unconditional draft compared with a Philox-sampled pick, 0.6-0.72 is more realistic.
- Asynchronous drafting (v2, chain_launch/chain_poll) brings pure decode to ~720 tok/s. — The API exists (mtp.cpp:1404-1484; it needs force_on_ and prepare_chain). But with 16 slots x 2 rows at most 8 rows per group there are exactly 4 groups for 4 stages, so stage 0 cannot start group g until g's drafts exist; drafting latency stays on the ring. CUDA3 must also spend GPU time on 16 drafts per cycle (~0.45-0.6 ms each). Async drafting only removes the host stall, so expect ~650-690, not 720.
- Fixed S=2*GS per group (pad rows) avoids graph thrash, and the 8-user gain is +35-45%. — These two claims contradict each other. With a fixed S=8, a group with 2 active slots pays for an 8-row window (and pad rows clear sl.cached, generate.cpp:8612), which wipes out the 8-user gain. Keeping today's variable G.S (generate.cpp:8608-8611) means up to 4 layouts x 4 groups of window+commit graphs per stage. At 20-30 MiB per pair (#997, verify.cpp:2499) that does not fit in CUDA3's 267 MiB, so LRU eviction and recapture follow, and each capture synchronises the stream (verify.cpp:2371).
- Admission costs ~330 ms for a ~100-token prompt read and 48-86 ms for the slot copy. — Engine log: ~100-199-token prompts are read in a median 458 ms (n=2185, 314-892), and 'copied in' has a median of 40.5 ms (35-154). A ≈ 0.5 s+ per admission still holds.
- MTP in batch slots gives +31-39% (BATCHING.md). — That figure comes from one GPU on an RTX PRO 5000 / R9700 with another setup. On this model the only --batch-mtp runs (logs/E-H10-batch2.txt, E-H41, engine 0.1.40, RTX 5090) died at the first admission with 'mtp: unsupported native MMVQ GGML type'. Commit 569c094 in HEAD fixes that, but it has never been re-run here. Batch MTP has never produced a token on this model on this box.

**Übersehene Blocker:**
- The 8-row cost is unknown, and the gain changes sign around it. Break-even is T8 + 4*t_draft < (1+alpha)*T4 ≈ 1.6-1.7 x 6.6 ≈ 10.6-11.2 ms. The only existing 8-row measurement (linux/results/batch_test_32.txt: 547 tok/s, ~14.7 ms per group-step) is above that, though confounded by VRAM eviction. Measure first: batch_test32.py with --max-context 16384, no code, ~15 min.
- LRU eviction (evict_batch_graph, verify.cpp:2468-2496) destroys the window AND commit graph of a layout. With deferred commits it can delete the commit graph of a group whose commit is still pending. Pending keys must be pinned, or batch_commit_deferred must re-capture (capture_commit_batch) before launch.
- Pad rows versus variable S: fixed S kills the 4-12 user gain. Variable S (as today) multiplies graph pairs on CUDA3, which has 267 MiB free, leading to eviction and recapture stalls (a capture synchronises the stage stream, verify.cpp:2371). Moving a layer off CUDA3 or adding --vram-reserve-mib is likely required, not optional.
- Slot-drafter VRAM is a measured figure: ~164 MiB per slot at 128K, ~60-70 MiB at 32K, so ~1 GiB for 16 slots on CUDA3. Each drafter also captures its own round graphs (T=1,2). All of it is allocated before the expert cache is sized (generate.cpp:4006 vs the cache at ~4030+), so CUDA3 loses ~400-450 experts and its CPU-expert/doorbell work grows on the stage that also drafts.
- Draft latency stays on the critical path whenever groups == stages, which 16 slots x 2 rows at 8 rows per group forces. Hiding it needs more groups than stages (e.g. 8 groups of 2 slots = 4-row windows), which gives back part of the amortisation.
- --batch-mtp has never run successfully on this model here: the 0.1.40 runs (logs/E-H10, E-H41) failed at the first admission, fixed by 569c094 but never re-validated. Single-GPU batch MTP cannot be checked on one 3090 at full residency either. The groups-1 split prototype is effectively the first validation of the whole path on this model.
- The pipelined path ignores STRATA_BATCH_DECODE_SHARE and reads every prompt in one piece while all slots wait. read_part returns sp.run over the whole range when `piped` (generate.cpp:9722), and pipe_drain runs before every admission (8659). Measured share 1.0 vs 2.0: 246.3 vs 246.0 tok/s (linux/results/strata-E1/F2). The admission stall A that dilutes lever B cannot be tuned away, and the user's planned experiment 3 is a no-op in the current pipelined config.
- Upstream / ports: sycl/src/core/verify.cpp and sycl/src/program/generate.cpp keep their own copies of the guard and gate (maintained by 3-way merge, docs/INTEL.md). A PR must leave SYCL refusing and say so. The CUDA sources also build for HIP, so new stream priorities, events and graph code need the same STRATA_USE_HIP guards as mtp.cpp:419-436.
- The stash is ~6 MB per group per non-last stage (C=10240), ~18-24 MB per stage, not 'a few MB'. That is still small, but CUDA3 needs none, which the analysis did not exploit.

The code reading holds up: every cited file:line exists and does what the analysis says. Gate at generate.cpp:3871-3878. Slot drafters at 4006-4018 are bound to CUDA0 objects. Stage max_t at 6745/6757. stage_batch guard and slot-count check at verify.cpp:2527/2546. batch_launch commits right behind the window at :2790. Commit intermediates are indexed by window row at :2397-2447, and the mapped h_commitb_ race is real. The core design is the right one, and the cheaper alternatives (snapshotting GDN state: ~3 MB per layer per slot; speculative commit plus rollback) are worse: deferred per-group commits with a stash on stages 0..n-2, accept/draft at the last stage, OnDevice(last_st->dev) for the slot drafters.

The analysis is too optimistic about the numbers:
- The 8-row stage time is an extrapolation. The one real 8-row data point (linux/results/batch_test_32.txt, 547 tok/s ≈ 14.7 ms per group-step) is ~1.8x worse and is confounded by VRAM eviction, so 'just run batch_test32' cannot settle it.
- Alpha is overstated: the 81% figure is mislabelled, and this box shows 60% chain acceptance.
- The v2 async-drafting gain ignores that with 4 groups on 4 stages each group's draft is on its own critical path.
- Fixed-S padding contradicts the claimed 8-user gain.
- The measured drafter VRAM (164 MiB per slot at 128K) was not found.
- Batch MTP has never worked on this model here: the 0.1.40 runs crashed at admission, and the fix in HEAD has never been re-run.

Recommended order:
- Step 0 (no code, ~15 min, once the GPUs are free): run tools/batch_test32.py with --max-context 16384 and the same exactness flags, to get a clean T8 for 4 groups x 8 rows at full residency. Continue only if T8 ≲ 9 ms.
- Step 1 (the analysis's groups-1 prototype; correct as written, 1-2 days including build and exactness runs, an under-a-day coding estimate is plausible):
  - gate change;
  - stage max_t=8;
  - remove the guard at verify.cpp:2546 and replace the slots_.size() check at 2527 with a hand-off row capacity;
  - drafters, residual rows and bind on last_st;
  - run tools/batch_test.py --batch 4 with --batch-groups 1 --batch-mtp --pcie-frac 0 --adapt-every 1000000 and STRATA_IQ_MT_MIN=1.
  This gives exactness across 4 stages, alpha, per-slot draft ms and drafter MiB at 32K.
- Then decide on the pipelined version: v1+v2 ≈ 8-11 days, plus v3 batched drafter ≈ 4-6 more.

Relevant to the user's current plan: experiment 3 (raising STRATA_BATCH_DECODE_SHARE) does nothing in the pipelined layer-split config. read_part at /media/raid0/StrataQ38FN/Strata/src/program/generate.cpp:9722 reads the whole prompt with sp.run when `piped`. The share 1.0 vs 2.0 runs measured 246.3 vs 246.0 tok/s. It only applies with --batch-groups 1.

Key files: /media/raid0/StrataQ38FN/Strata/src/core/verify.cpp, /media/raid0/StrataQ38FN/Strata/src/program/generate.cpp, /media/raid0/StrataQ38FN/Strata/src/core/mtp.cpp, /media/raid0/StrataQ38FN/Strata/include/strata/core/verify.hpp, /media/raid0/StrataQ38FN/linux/results/batch_test_32.txt, /media/raid0/StrataQ38FN/logs/E-H10-batch2.txt, /media/raid0/StrataQ38FN/Strata/strata-unsloth-ud-iq4_xs.log.

## Gegenprüfung: A: prefill-interleave. The batched prompt read (sp.run) runs on a worker thread while the main threa (optimistic, 4–10 Tage)

The mechanism analysis is accurate, and the most valuable finding holds: decode share and BYIELD are inert in piped mode, so experiment 3 measures noise. Previous share 1.0 vs 2.0 runs with auto groups (the 0.1.41 default) were noise too.

Fixed-cost cuts:
- The device-to-device slot copy saves 35-85 ms per admission (copies measured 36-88 ms).
- Removing the loans saves a parallel refill of about 0.4 GB on CUDA0 and CUDA3 (both cards at 16 slots), roughly 30-60 ms.
- Together that is about +8-10% on short prompts (about 390 tok/s), as claimed. It is code-only work, but at 16 slots / 32K it needs VRAM that CUDA3 does not have.

Overlap, short prompts: 410-460 tok/s is realistic if f is about 0.4-0.6. The admission cap does not bind there (A_wall about 0.65 s gives a cap above 750 tok/s).

Overlap, long (1.2K) prompts: the 300-365 tok/s claim is optimistic. Low decode loss (f) and low prefill stretch (s) trade against each other on 512-token MMQ chunks, and serialized admissions cap throughput at 512/A_wall. Combining both gives about 275-330 tok/s (+10-30%), not +45%. The analysis is right that the ceiling for this lever is about 603 tok/s and that it cannot reach vLLM's 710.

Single-user speed is unaffected when the overlap is gated on piped && batch_on().

None of this applies to the 16-slot / 32K or the 262K agent configs until VRAM is rebalanced: CUDA0 and CUDA3 loans, CUDA3 at 6141/6144, and kv-resident.

**Nicht bestätigte Behauptungen:**
- copy_to_slot is per stage, with cudaDeviceSynchronize and a host round trip via conversation_checkpoint/kv save+restore. 'copied in 86.9-88.6 ms' for every admission. — The mechanism holds: generate.cpp:8312-8346; transfer() is a synchronous cudaMemcpy(Default) at conversation_snapshot.cpp:112-120; checkpoint save/restore go through host vectors at conversation_state.cpp:154-198. The 'every admission' figure is wrong: the same log run also shows 'copied in 48.1 ms' and '36.2 ms'. The saving from a device-to-device copy is about 35-85 ms, not a flat 80.
- CUDA0-2 keep their own prompt buffers, only CUDA3 borrows 166 slots, so each admission lends and refills on CUDA3 only. — That holds only for the 12-slot runs (log 32194-32197, 35429-35432). In both 16-slot server runs (log 34335, 35031-35032), CUDA0 ALSO borrows 170 slots and CUDA3 borrows 167 of only 6141 slots. CUDA3 holds 6141 of its 6144 experts ('expert cache 6141 slots', 'token graph hit path: 24573 resident experts'), so its verifier runs the doorbell graph (no '100% VRAM resident' suffix). CUDA3 had only 267 MiB free at capture time.
- Test at 12 slots, where CUDA3 has more headroom, to satisfy the no-loan gate. — Every 12-slot run in the log still borrows on CUDA3 (log 32197, 35432, 35996, 36553), with about 880 MiB free at capture. Giving CUDA3 its own 0.39 GiB buffers means shrinking the cache below 6144 (losing all-resident) or freeing VRAM elsewhere (smaller --max-context or fewer slots). --no-prefill-borrow books the buffers out of the cache, so the stage also stops being 100% resident.

**Übersehene Blocker:**
- The gate the analysis itself sets (no stage lends, every stage 100% resident) is NOT met by the 16-slot / 32K config the 360/250 tok/s numbers come from. CUDA0 lends 170 slots, and CUDA3 lends 167 and holds only 6141 of 6144 experts, so it runs the doorbell graph and serves the CPU pool on the main thread (log 34314/34335, 35010/35031-35032). The prototype as written would be gated off, or it would race the non-thread-safe ExpertSource. Fixing this costs VRAM on the tightest card (267 MiB free at capture), realistically a smaller --max-context, which changes the baseline and the product.
- Admissions stay serialized (ctl lock, one main session), so throughput is also capped at about 512 / A_wall, where A_wall includes the read STRETCHED by co-running decode. For 1.2K prompts (about 1.05 s read, chunk-pipelined and compute-dense), a 30-50% stretch caps throughput at about 300-340 tok/s, whatever f is. The model 603/(1+603R/512) ignores this coupling between f (decode loss) and s (prefill stretch).
- PleTable::gather_batch is not thread-safe (ngram.cpp:507ff: unguarded n_prefetch, prefetch slots, bytes_read). Today the stage-0 batch window (verify.cpp:2589, main thread) and the prefill's async PLE gather (prefill.cpp:2052, ple_next future) never run at the same time; with overlap they would. They also share one FIFO SSD reader queue (Direct mode is the default, generate.cpp:374). A 512-token chunk queues about 8K row reads, and the main thread blocks in stage_batch, so the whole pump stalls on stage 0. That needs a mutex, and it raises f.
- The stage-0 Verifier's batch graph writes the main session's ss.ple.scratch (verify.cpp:829/909/911). The prefill's per-token PLE fallback writes the same buffer (prefill.cpp:2449). The overlap must be gated on the batched PLE path (ple_batch true), or the prefill needs its own PLE scratch.
- A stale BYIELD in piped mode costs an extra pipe_drain (generate.cpp:8659 drains for any non-BSTOP line) before 8670 discards it. Dropping BYIELD on the server side for slot_gs is worth doing, not just cosmetic.
- --short-read 0 as a workaround makes the 2-5-token turn header a full batched sp.run, at about 250-300 ms of fixed cost across 4 serial stages (log: 87 tokens in 336 ms). That is more GPU time than the header windows (about 12 ms each); it only pays off if f is small.
- HIP/AMD: piped is the default on R9700 multi-GPU too (BATCHING.md:40). The overlap would change behaviour there unless it is gated to CUDA. The priority stream needs the same #if !HIP guard as mtp.cpp:421. f on RDNA is unmeasured. The sycl/ tree is a separately migrated copy of generate/verify (commit 23a38a0), so an upstream PR adds re-migration burden.
- The agent configs the user actually runs (X12/X16-agent-262k: --kv-resident 32768, conversation cache, STRATA_KV_PREFETCH=1) fall under the analysis's own 'gate off --kv-resident' rule. The 3-8 days cover only the 32K non-streamed config.
- Measuring f needs new instrumentation. pump prints its rows/s line only when all groups go idle (generate.cpp:8633-8644), so 'batch rows/s during reads' does not exist in the log today.

Nearly every file:line reference checks out against branch g3-batch-over-8 (3a69518): 9723, 8659/8670, 8545-8646, 8312-8346, 10962-11004, verify.cpp 600-606/2346-2376/2523-2549/2783-2876, server.py 741/1290-1330/1424-1455, prefill.hpp 150-156, kVerifyMaxT=8. The 'interleaving alone gives nothing' and 'stage-exclusive overlap gains about nothing' arguments are correct: with a serial single-chunk read, all decode groups pile up in front of the card that is reading. Rejecting real in-window prompt rows as the first step is also justified (8 rows per window, 4 spare rows per group-step, grouped rows refused on a split).

Main corrections:
1. The 16-slot config fails the no-loan / all-resident gate on CUDA0 and CUDA3 (log 35010, 35031-35032). The suggestion to 'test at 12 slots' does not help either, because CUDA3 lends there too.
2. Two host objects become concurrently shared that the analysis did not audit: PleTable::gather_batch (not thread-safe, one FIFO SSD queue) and the main session's ss.ple.scratch, which is written by the stage-0 batch graph.
3. Serialized admissions cap long-prompt throughput at 512/A_wall, and A_wall grows when the read shares GPUs.

Side finding for lever 6: at 16 slots, CUDA3's batch windows run the doorbell graph because 3 experts do not fit (verifier line without the '100% VRAM resident' suffix). Getting CUDA3 to 6144/6144 (a slightly smaller reserve or context) and re-measuring the 603 tok/s ceiling is a cheap no-code experiment.

Recommended minimal prototype (under 1 day, measurable):
(0) Drop experiment 3 or re-scope it.
(1) Pick a config where all four cards say 'keeps its own prompt buffers' and '100% VRAM resident' at 16 slots, e.g. --max-context 16384; confirm it in the log first and re-measure its baseline.
(2) Add a mutex in PleTable::gather_batch.
(3) Add the env-gated std::async around sp.run in read_part, with pump(true) on the main thread, apply_pending and the busy reset skipped, pipe_drain afterwards, and a counter for rows decoded during reads.
(4) Measure f (decode rows/s during reads vs pure decode) and s (read ms vs the drained baseline) on 150- and 1.2K-token prompts, plus the batch_test16 exactness check.

Decision rule: continue only if (1-f)/(1+s) gives at least about +15% on 1.2K prompts. Otherwise invest in the decode ceiling and prefill efficiency.

Effort is 4-10 focused days rather than 3-8, because of the VRAM/config precondition, the thread-safety audit, the device-to-device copy with exact checkpoint semantics (idx_pooled row rewrite, mapped-host pools fallback), piped exactness tests, HIP gating and the upstream PR. Making it work with --kv-resident or with an active loan is not included and would add more.

Key paths:
- /media/raid0/StrataQ38FN/Strata/src/program/generate.cpp
- /media/raid0/StrataQ38FN/Strata/src/core/verify.cpp
- /media/raid0/StrataQ38FN/Strata/src/prefill/prefill.cpp
- /media/raid0/StrataQ38FN/Strata/src/kernels/ngram.cpp
- /media/raid0/StrataQ38FN/Strata/serve/server.py
- /media/raid0/StrataQ38FN/Strata/strata-unsloth-ud-iq4_xs.log
- /media/raid0/StrataQ38FN/linux/configs/Q3-b12-share4.json

## Gegenprüfung: D: quick wins without deep engine work (trim plus an explicit split, slot count and groups, decode s (optimistic, 0.5–2.5 Tage)

The core lever is sound and well grounded: an explicit split of 12,24,36 plus --trim-stage-weights, which is upstream's own documented batching recipe. Runs with 1, 2, 3 and 4 lending cards cost ~60-70 ms of prompt read per lending card (380/457/522/579 ms for a ~160-token prompt). With all 4 cards on own buffers, admission cost (T_adm) for short prompts falls from ~0.57 s to ~0.43 s. Plugged into the analysis's validated model L/(L/R + T_adm): ~360 -> ~395-410 tok/s short (+10-14%), and long ~250 -> ~275-290 tok/s (+10-15%; long prompts read ~2 GB per admission today). CUDA3 at 100% residency may add 0-8% to decode. ckpt=0 adds +3-5%, but only for the benchmark. The D2D slot copy saves ~30 ms per admission, +2-3%. The 24-slot step is unproven, 0 to +10%: H24 lost because residency collapsed, and pure decode at 32 slots was below 16. pick_slot and BYIELD changes are ~0 in these benchmarks. Realistic ceiling for lever D: ~400-440 tok/s short and ~280-300 long, about 55-62% of vLLM's 710 and 70-75% of its 400. The analysis's 430-470 short is the optimistic end. Reaching vLLM needs admissions that overlap decode (lever 4). Decode share and DP2 are confirmed losers.

**Nicht bestätigte Behauptungen:**
- 24 slots in 4 groups of 6 lift pure decode from ~603 to ~700 and add ~8-10% through the server — Not supported by the data. H24 (4x6) measured 331 tok/s (334/330 rows/s), below 4x4 at 357-361. In that run CUDA0 held only 6033 slots and CUDA3 5612 (doorbell graph, CPU misses), and all 4 cards borrowed. Pure-decode batch_test with 32 slots (4x8) gave 547 tok/s, below 603 with 16 slots. The t = 2.8 + 0.95*S fit comes only from the 2-group Q2 run. The gain is possible only if trim restores 100% residency and own buffers at 24 slots, which is unmeasured.
- pick_slot ignores held prefixes in piped mode (stale comment), and fixing it is a big win for multi-turn traffic — The first half holds: server.py:1430-1440. The 'big win' does not: the engine already searches every idle cached slot for the longest prefix, whichever slot BGEN targets (generate.cpp:9157-9171, restored via copy_from_slot at 9215-9250), and it also has the parked-conversation cache. The server change only decides which idle slot gets overwritten. Pad rows also clear cached on idle slots inside active groups (8617).
- DP2: each 2-GPU instance holds ~49% of experts, CPU misses lower decode, single-user speed drops to 70-90 tok/s, so DP2 loses — The conclusion holds, the mechanism does not. strata-dp2-0.log: 15223 of 24576 pairs (62%, 98.6% of routed mass), decode hit rate 96-99%, so decode misses are not the problem. The ~164-token prompt read takes ~990 ms vs 457 ms on 4 GPUs. Measured: 1 user 74.4 tok/s aggregate (decode p50 89.5); 12 users 152.5 tok/s vs 361 for 4-GPU H12. Pinned RAM is 28.35 GiB per instance, matching the 26-30 GB claim.

**Übersehene Blocker:**
- Expert residency collapses as slots grow, because each slot session takes 0.15 GiB per stage. At 20 slots 3 cards lend. At 24 slots CUDA0 holds 6033 and CUDA3 5612 slots. At 32 slots all four cards drop below 100%, CUDA3 to 5080. That, not the per-row cost model, explains H24 and G32. Holding 24 slots, 100% residency and own buffers on CUDA3 at once is borderline even with trim (~3 GiB freed): CUDA3 also carries the head, the 839 MiB MTP draft layer and the 213 MiB draft head, and had 247-271 MiB free at graph capture. Dropping to --max-context 16384 to make room conflicts with the X12/X16-agent-262k use case.
- Batch CUDA graphs are keyed per layout (bkey(rows,S,hbase) in exec_bm_/commit_bm_): 16 per stage at 4x4, 24 at 4x6, 32 at 4x8. With 32 slots the LRU evicted and recaptured graphs 87 times (verify.cpp:2497-2515, 'no VRAM for a new window graph'). More rows per group needs VRAM headroom beyond the expert budget.
- No measurement shows that more rows per group raises pure decode. The 32-slot batch_test (547 tok/s) is below 16 slots (603). Before claiming the 24-slot gain, prove pure decode at 24 slots with 100% residency on all 4 cards.
- STRATA_SPLIT_OWN=1 or --no-prefill-borrow (generate.cpp:2079-2106) forces own buffers at ~884 MiB per stage for chunk 512, bypassing the buffers + 1.5 GiB rule. It is the fallback if trim leaves CUDA0 or CUDA3 just short, but it takes the room from the expert cache, which can break 100% residency.
- Exactness: own buffers and the 3 extra resident experts on CUDA3 change which experts run on GPU vs CPU. Upstream notes the output then differs from the borrowing default (generate.cpp:2084-2086). Re-run tools/batch_test16.py, batch_interleave_test.py and parking_test.py (layer-split stage images) with STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 1000000, and check single-user MTP stays at ~121 tok/s.
- The D2D slot copy is not a drop-in. conversation_kv_restore has kv_mode 1/2 paths (host authoritative pools, kv_stream_reset, kv_ring_restore: conversation_snapshot.cpp:203-225). checkpoint_restore also rewrites idx_pooled rows and ple_prev. copy_from_slot can restore from a checkpoint's stage_parts. The fast path must fall back for those cases and keep the HIP compat layer and Windows builds working.
- The engine already reuses prefixes from any idle cached slot and from parked conversations, so the pick_slot change is marginal. Measure it with bench_multiturn.py before counting it as a gain.
- The user's planned experiments are already settled. Experiment 2 (2 groups of 8) was measured in Q2 at -25%. Experiment 3 (higher decode share) cannot change anything while piped (code at 9721), and E1/F2 and Q3 confirm it. Experiment 1 (DP2) is losing so far: 152.5 tok/s at 12 users, against 361 for the 4-GPU H12 run.
- The throughput model assumes 512-token outputs. With shorter agent replies admissions dominate even more, so cutting admission cost (T_adm) gains relatively more there. With longer replies it gains less.

I checked every cited file:line against the code. The mechanisms hold: piped by default, 8 rows per window, a drain for every stdin line, a whole-prompt read when piped, decode share and BYIELD inert, loans plus pageable refills on CUDA0/CUDA3, the CUDA0 trim needing an explicit split, and CUDA3 on the doorbell graph.

Four places are overstated or wrong:
1. Data across runs contradicts the 24-slot gain. H24 lost because CUDA0 fell to 6033 and CUDA3 to 5612 resident experts with 24 slot sessions, and pure decode at 32 slots was 547 tok/s, below 603 at 16.
2. pick_slot is not a big multi-turn win. The engine already takes the longest prefix from any idle cached slot (generate.cpp:9157-9171).
3. The DP2 residency figure is wrong: 62% of pairs and 98.6% of the routed mass are resident, not 49%. DP2 still loses, because prompt reads take ~990 ms and the pipeline is half as long. It is already measured at 152.5 tok/s for 12 users.
4. Minor: syncs per stage are 2 + qsa_alloc, not 3, and the trim likely frees ~3 GiB per card, not 2.

Strong supporting evidence the analysis didn't cite: upstream comment generate.cpp:2067-2069 (a loan makes a 2K prompt read 35-37% slower), and docs/BATCHING.md:32/41/200 (the recommended batching setup is exactly 12,24,36 + --trim-stage-weights + --batch-groups 4).

Minimal first step, under a day, config only:
- Copy /media/raid0/StrataQ38FN/linux/configs/G16-pf512-share2.json, set "layer_split": "12,24,36", add "--trim-stage-weights", and drop the share env.
- Verify in the log: 4x 'loads the dense weights of layers ... only', 4x 'keeps its own prompt buffers', 4x 'zero-doorbell graph', 6144 slots on every card, and no new 'blob reads from the file' per request.
- Run G16-short and G16-long. Short-prompt reads should fall from ~457 ms to ~320 ms.
- If any card still borrows, retry with STRATA_SPLIT_OWN=1 and confirm residency stays at 100%.
- Only then try "parallel": 24, and keep it only if every card still holds 6144 slots and the batch_test16/32 pure-decode rate rises.

Code items: server.py BYIELD/pick_slot is about half a day; the D2D slot copy with kv_mode fallbacks and HIP/Windows builds is 1-1.5 days. Overall risk is low, but the exactness tests must be re-run because residency changes.