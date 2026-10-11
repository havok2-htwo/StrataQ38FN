# Design: overlapped batch admissions (final, after critique)

## Approach

REVISION SUMMARY. Five things changed from the first spec:
- The overlap never captures, instantiates or evicts a batch graph, and never grows a lazily sized buffer while a read is in flight. Missing layouts are captured before the read, with the pipeline drained.
- read_overlapped is now exception-safe and has a bounded abort.
- CUDA fault and ERR reporting are now correct across threads.
- Residency and ar state are checked live.
- The gain model now uses x = f*(1+s) and is recalibrated from the current log.
Everything else from the first spec stands: what is overlapped, the thread roles, and the gates.

GATING. The whole feature sits behind STRATA_PIPE_OVERLAP=1, which is off by default.
- Code: `#if defined(__linux__) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)` (kPipeOverlapBuild).
- It is reachable only when `piped && batch_on()`.
- The new PleTable lock, PLE urgent queueing, PLE in-flight reserve and stream priority are each switched on by a setter that only the env-on path calls. With the env off, every platform runs exactly the code path it runs today. src/sycl is untouched.

WHAT IS OVERLAPPED. Only the single `sp.run(ids+a0, b0-a0, a0, e)` that read_part makes in piped mode (generate.cpp:9766). That is the per-card Prefill chain:
- stage 0 runs on the calling thread;
- later stages run on std::async threads (prefill.cpp:3895-3912);
- run() joins the chain before returning (4074-4086).

Everything else stays serialized and drained, in today's order:
- the BGEN drain (8702);
- pre-read work (9159-9440);
- header read_windows;
- checkpoint_at;
- the first-token window;
- copy_to_slot and BADM.

THREADS.
- T_main (the request thread) runs read_overlapped. While the read runs it calls only pump(true), which uses batch_poll and batch_launch on already-captured graphs, plus the PLE gather and sample_rows. It also scans control lines.
- T_read (a new std::thread per segment) runs sp.run.
- T_stage1..3 already exist. They run on_chunk/on_stage_chunk, including mtp.prefill on CUDA3.
- T_ple and the PleReader worker already exist.
- T_stdin already exists.

T_main is the only thread that touches bs/pg/stage_group/bt_*, the Verifiers, in_lines and `err`. T_read writes only its own `we`, `wok` and `wcuda` (its cudaPeekAtLastError). T_main reads them after join.

NEW INVARIANT: NO CAPTURE AND NO GROWTH DURING THE READ.

Pre-read, in overlap_prepare(). T_main runs it with the pipeline drained, after the gate's checks pass:
- (a) Refresh ar on every stage (refresh_ar made public as `bool ar_live()`).
- (b) Each stage gets ensure_batch_layouts(GS, n_groups): it captures any missing full-width layout (base = g*GS, S = GS) for every group, under the current ar flag. Capture may evict LRU layouts as today; that is legal here because nothing else is running.
- (c) mtp.reserve_prefill_records(sp.chunk()) pre-grows pf_dev_ (mtp.cpp:1164-1172 is then a no-op for every chunk).
- (d) Set set_no_capture(true) on every stage verifier. stage_batch then fails with a clear error instead of capturing (defensive; it is not expected to fire).

During the read, pump starts a group at stage 0 only if every stage holds its layout:
- If the group's natural G.S layout is captured on all stages, it starts as today.
- Else, if the full GS layout is captured on all stages, the group runs PADDED UP to S = GS. The extra rows are pad rows for inactive slots, exactly like today's pad rows below G.S, with sl.cached = false.
- Else the group waits until after the final drain. That changes timing only.

Since (b) guarantees the full layouts, the wait case should never happen. If it does, it is only a slowdown.

grp_host is NOT a lazy per-read allocation: prefill.cpp:1068-1083 is inside Prefill::init (990-), so the first critic's concern does not apply. The remaining lazy allocations in run_impl are refused by the gate:
- cpu_x (3002): cpu pool, gated;
- the help buffers (1549): helper, gated;
- debug statics (2703, 3845: STRATA_IDX_FP16_CHECK / STRATA_CVEC_FUSE_CHECK): gated.

MTP prefill graphs are all captured in prepare_prefill (mtp.cpp:1132-1133, called at generate.cpp:6930). Since T_main never runs record_window/capture during the read, the per-device function statics in kernel wrappers (fused_gr.cu:748/766/1686, prefill/kernels.cu:699) are reached only by the prefill threads, one per device, which is the same as today. sampler.cu:1200 ok[] is reached only by T_main (sample_rows); the prefill and mtp.prefill never sample. split_scratch has its own mutex and never frees (sampler.cu:1135-1150).

ORDERING inside read_overlapped(a0, b0, e):
0. Precondition: no group in flight (the BGEN drain or a previous segment's drain).
1. overlap_prepare() (above). On failure, refuse with a reason and fall back to sp.run.
2. Construct an OvGuard (RAII) BEFORE starting the thread. Then set ov_active = true, take snapshots (t0, rows0) and the reference decode rate D_ref (see stats).
3. Start T_read: `try { wok = sp.run(...,we); } catch (const std::exception& x) { wok=false; we=...; } catch (...) { wok=false; we="prefill: unknown exception"; } wcuda = cudaPeekAtLastError(); done.store(true, release);`
4. While !done:
   - if pump_ok && !pump(true): pump_ok = false, ov_err_printed = true (pump already printed ERR with its own err text), stop_req = true, release_gpu_waits(5000) on every stage verifier so no stage-0 group spins on its PLE flag;
   - scan_ctl_lines(true): apply BSTOP, drop BYIELD;
   - sleep STRATA_PIPE_OVERLAP_SLEEP_US (default 20) or yield.
5. Joining: the guard waits on `done` with a bound (STRATA_PIPE_OVERLAP_JOIN_S, default 30 s, under the 60 s watchdog), then joins. On timeout: stderr line, fflush, _Exit(1), since a joinable thread cannot be destroyed safely.
6. If !pump_ok: batch_fatal = true, keep `err` (pump's text), log `we` to stderr only, return false. The caller (9960) skips its own ERR printf when ov_err_printed, then does the #224 fast exit when cudaPeekAtLastError() || ov_cuda_fault.
7. If !wok and (wcuda != cudaSuccess || sp.chain_cuda_error() != cudaSuccess): ov_cuda_fault = true, e = we, return false without draining (the context is poisoned, so the caller exits fast).
8. pipe_drain() with ov_active still set, then set_no_capture(false), ov_active = false, idle_report(/*reset_busy=*/false) if !batch_on(). This prints and resets bt_* without touching progress().busy.
9. Print the stats line.
10. If !wok: e = we and return false. "cancelled" behaves as today.

The OvGuard destructor runs on every exit, including exceptions thrown out of pump or scan_ctl_lines:
- if the thread is still joinable: stop_req = true, release_gpu_waits on every stage, then the bounded wait and join;
- clear no_capture and ov_active.

GATE (overlap_gate, on T_main, per segment). Every condition must hold, otherwise it falls back to today's sp.run with one stderr line per distinct reason:
- env on and build OK;
- piped, batch_on(), admit_slot >= 0;
- no pf_parts loan (p.first < 0 && p.lent.empty());
- LIVE residency: std::all_of over host_res for all layers (not the startup const at 7395), pending.empty(), and every stage_verifier(k).ar_live() (refresh then ar_on);
- o.kv_resident <= 0; no images; !peer.valid(); no remote experts; fs_swap == nullptr; !cvec().loaded();
- debug envs STRATA_IDX_FP16_CHECK / STRATA_CVEC_FUSE_CHECK unset;
- the PleTable is concurrency-capable: Mmap mode or a threaded reader (not --ple-sync-submit);
- sp.device() == stage_verifier(0).device();
- sp.overlap_safe(why) over the whole chain: no pp or help_pp, cpu_pool_ == nullptr, and every PLE stage uses the batched block (ple_batched_()).

PLE.
- PleTable gets set_concurrent(bool). When it is on, a mutex guards the bookkeeping and bytes_read is atomic. The reader I/O runs outside the table lock (the threaded reader locks itself).
- Verifier::set_ple_urgent(true), called only with the env on, makes stage_batch issue its gather as urgent. Urgent jobs are queued at the front.
- PleReader::set_urgent_reserve(R), only with the env on (default R = 16 of ple_inflight = 256), keeps R in-flight slots free for urgent jobs: the worker's pump() submits a non-urgent job only while free_slots.size() > R.
- Order-only changes, so the bytes are the same.
- The RowCache is shared, so the stats line reports the PLE hit rate and the stage-0 gather ms during reads.

PRIORITY (M4 A/B, not assumed to help).
- With env and STRATA_PIPE_OVERLAP_PRIO (default 1), each stage verifier's cs_ is created at the greatest priority. If that fails, it falls back to a normal NonBlocking stream, as mtp.cpp:427 does.
- Ampere does not preempt running blocks, and the persistent moe_fused grids hold SMs, so priority only orders new blocks. M4 also tries CUDA_DEVICE_MAX_CONNECTIONS=32 (launch env) and STRATA_PIPE_OVERLAP_MTP_PRIO=0 (the drafter's stream on CUDA3 at normal priority while the overlap is on). Every arm is judged on x, not f.

BYIELD. With the env on, BYIELD no longer triggers the drain at 8702. The line is dropped at 8713 anyway, and the overlap loop drops it as well. M6 may make this unconditional (critic 2: safe), as a separate commit.

ANSWERS TO THE CRITICS.
- Capture/VRAM race: no capture during the read (overlap_prepare, plus pad-up or wait, plus the no_capture assert).
- Exceptions: OvGuard, catch(...), bounded join.
- Per-thread CUDA error: wcuda plus the chain's recorded errors, and ov_cuda_fault feeds the #224 check.
- Double ERR: ov_err_printed, and pump's err text is kept.
- Priority fallback: normal stream.
- urgent/mutex on other platforms: setters, env-on only.
- Sync-submit: gated.
- Stale residency: live check.
- Lazy statics: T_main does no record_window during the read; see above.
- Busy spin: sleep 20 us by default, A/B in M4.
- Device assert: added.
- Hit rate: stats line.
- Gain model: x = f*(1+s), recalibrated.
- Test split: production 13,25,37 plus 12,24,36, with M0 confirming residency under the exactness flags.
- PLE flag on abort: release_gpu_waits.
- bt_* reset: idle_report(false).
- err aliasing: handled.
- TTFT: reported in M5.
- MTP pf_dev_: reserved.
- Audit additions: ss.block and ss.moe combine fallback.

## Changes

### src/program/generate.cpp — near 8563 (after `piped`): new state
static const bool pipe_overlap_env = kPipeOverlapBuild && env_int("STRATA_PIPE_OVERLAP",0)!=0;
static const bool pipe_overlap_prio = pipe_overlap_env && env_int("STRATA_PIPE_OVERLAP_PRIO",1)!=0;
static const int ov_sleep_us = env_int("STRATA_PIPE_OVERLAP_SLEEP_US",20);
static const int ov_join_s = env_int("STRATA_PIPE_OVERLAP_JOIN_S",30);
State: bool ov_active=false, ov_err_printed=false, ov_cuda_fault=false;
std::set<std::string> ov_refused_logged;
decode-rate tracker: dec_t0, dec_rows0, set at each BADM, sampled at each BGEN when no admission was in between, giving D_ref rows/s (an EMA);
stats sums.
Helper scan_ctl_lines(bool drop_yield): under in_mu, apply and erase BSTOP (copy of 9777-9781); with drop_yield, erase BYIELD and log once per read.

### src/program/generate.cpp — piped setup, where the stage verifiers and the PleTable are configured (around 6757/6820 and 2939)
Only when pipe_overlap_env && o.batch > 0 && the run will pipeline (o.batch_groups > 1 && !stages.empty()):
- gs.ver.set_stream_priority_high(pipe_overlap_prio) and ver.set_stream_priority_high(...) before init;
- every stage verifier: set_ple_urgent(true);
- ple_table.set_concurrent(true);
- ple_table.reader().set_urgent_reserve(env_int("STRATA_PIPE_OVERLAP_PLE_RESERVE",16)) when threaded;
- with STRATA_PIPE_OVERLAP_MTP_PRIO=0 (an A/B knob, default 1 = unchanged), mtp.set_stream_priority_normal() before its init.
None of this runs with the env off.

### src/program/generate.cpp — pump lambda 8588-8689
At the stage-0 start (8652-8666):
- in the round-robin pick, skip a group gi when ov_active && ov_layout(gi) == 0;
- after G.S is computed: `if (ov_active) G.S = ov_layout(pick);`, then the existing pad-row loop over t < G.S (so pad-up rows also get sl.cached = false and tok/pos 0);
- `if (!ov_active) apply_pending(false);`.
ov_layout(gi) computes S0 (as today):
- return S0 if every stage_verifier(k).has_batch_layout(gi*GS, S0);
- else return GS if every stage has (gi*GS, GS);
- else 0.
The all-idle block (8676) becomes `if (!ov_active && ...) idle_report(true);`. idle_report(bool reset_busy) is the existing body factored out; it stores busy=false only when reset_busy.
The ERR printf at 8604 and 8670 stays (it is the one ERR the server sees).

### src/program/generate.cpp — main loop 8702
`if (piped && line.rfind("BSTOP ",0)!=0 && !(pipe_overlap_env && line.rfind("BYIELD",0)==0) && !pipe_drain()) return 1;` M6 may drop the env condition (separate commit).

### src/program/generate.cpp — new lambdas overlap_gate(a0,b0,why) and overlap_prepare(why), before read_part
overlap_gate returns false with `why` on the first failing check:
- env/build; !piped; !batch_on(); admit_slot < 0;
- a pf_parts loan;
- !std::all_of(host_res...) (live, replacing the const at 7395); !pending.empty();
- any !stage_verifier(k).ar_live();
- o.kv_resident > 0; images (req_imgs or sp.embd_rows over [a0,b0)); peer.valid(); remote experts; fs_swap; cvec().loaded();
- getenv(STRATA_IDX_FP16_CHECK) or getenv(STRATA_CVEC_FUSE_CHECK);
- !ple_table.concurrent_capable() (Mmap mode or a threaded reader);
- sp.device() != stage_verifier(0).device();
- !sp.overlap_safe(why).
overlap_prepare runs only after the gate passes (T_main, drained):
- each stage: ensure_batch_layouts(GS, n_groups, err) (capture the missing full layouts; may evict as today);
- if MTP is on: mtp.reserve_prefill_records(sp.chunk(), err);
- each stage: set_no_capture(true).
A failure is a refusal ('overlapped read off: <why>'), not an ERR: set_no_capture(false), clear the CUDA error, fall back. Log the time it took when it captured anything.

### src/program/generate.cpp — new lambda read_overlapped(a0,b0,e) plus struct OvGuard
struct OvGuard {
  std::thread w; std::atomic<bool>* done; /*refs*/
  ~OvGuard() {
    if (w.joinable()) {
      if (!done->load()) {
        stop_req = true;
        for (k) stage_verifier(k).release_gpu_waits(5000);
      }
      if (!wait_done(*done, ov_join_s)) { fprintf(stderr, "strata batch: the overlapped prompt read did not end in %d s\n", ov_join_s); fflush(stdout); fflush(stderr); std::_Exit(1); }
      w.join();
    }
    for (k) stage_verifier(k).set_no_capture(false);
    ov_active = false;
  }
};
Body:
  ov_err_printed = ov_cuda_fault = false;
  bool wok = false; std::string we; cudaError_t wcuda = cudaSuccess; std::atomic<bool> done{false};
  OvGuard g{...};
  ov_active = true; t0 = now; rows0 = bt_rows;
  g.w = std::thread([&]{
    try { wok = sp.run(ids.data()+a0, b0-a0, a0, we); }
    catch (const std::exception& x) { wok=false; we=std::string("prefill: ")+x.what(); }
    catch (...) { wok=false; we="prefill: unknown exception"; }
    wcuda = cudaPeekAtLastError();
    done.store(true, std::memory_order_release);
  });
  bool pump_ok = true;
  while (!done.load(acquire)) {
    if (pump_ok && !pump(true)) { pump_ok=false; ov_err_printed=true; stop_req=true; for (k) stage_verifier(k).release_gpu_waits(5000); }
    scan_ctl_lines(true);
    sleep_or_relax(ov_sleep_us);
  }
  wait+join through g (factor a g.finish() that the destructor also calls).
  if (!pump_ok) { batch_fatal=true; if (!we.empty()) fprintf(stderr, "strata batch: overlapped read: %s\n", we.c_str()); return false; }   // err keeps pump's text
  if (!wok && (wcuda != cudaSuccess || sp.chain_cuda_error() != cudaSuccess)) { ov_cuda_fault=true; e=we; return false; }
  t1 = now;
  const bool drained = pipe_drain();   // ov_active still set
  for (k) set_no_capture(false); ov_active=false;
  if (!drained) { batch_fatal=true; ov_err_printed=true; return false; }
  if (!batch_on() && bt_windows > 0) idle_report(false);
  stats line: "strata batch: overlapped read of %lld tokens in %.0f ms (drain %.1f ms); slots decoded %lld rows meanwhile, D_ref %.1f rows/s, lost %.0f ms of decode; stage-0 PLE gather %.2f ms avg, ple hit %.1f%%"
  where lost_ms = read_ms - rows_during*1000/D_ref.
  if (!wok) { e = we; return false; }
  return true;

### src/program/generate.cpp — read_part 9764-9767 and read-failure path 9959-9973
read_part first line: `if (piped && batch_on()) { std::string why; if (overlap_gate(a0,b0,why) && overlap_prepare(why)) return read_overlapped(a0,b0,e); log_refusal(why); }`. The existing piped early return remains the fallback.
Failure path: `if (!ov_err_printed) { fprintf(stderr...); printf("ERR %s\n", err.c_str()); }`, and the #224 check becomes `if (cudaPeekAtLastError() != cudaSuccess || ov_cuda_fault)`. Reset ov_err_printed and ov_cuda_fault at request start.

### src/core/verify.cpp + include/strata/core/verify.hpp — Verifier::init 600-606; new members and methods; stage_batch 2556-2589
- set_stream_priority_high(bool) plus cs_prio_high_. In init (HIP-guarded): create with the greatest priority; on failure call cudaGetLastError() and fall back to cudaStreamCreateWithFlags(NonBlocking), as mtp.cpp:427 does.
- `bool ar_live() { refresh_ar(); return ar_on(); }`.
- `bool has_batch_layout(int base, int S) const`: build rows base..base+S-1 and check that exec_bm_ and commit_bm_ both hold non-null entries for bkey(rows,S,base) (the current ar_off_).
- `bool ensure_batch_layouts(int GS, int groups, std::string& err)`: OnDevice; refresh_ar(); for g < groups, capture_batch + capture_commit_batch with rows g*GS.. and S = GS, hbase = g*GS; bm_used_[key] = ++bm_tick_.
- `void set_no_capture(bool)`. stage_batch, before capture_batch: `if (no_capture_ && (exec_bm_.find(key)==end || commit_bm_.find(key)==end)) { err = "verify: a batch layout was not captured before an overlapped read"; return false; }`. Also skip the batch_graph_limit_ eviction when no_capture_ (it only runs for a missing key anyway).
- `void set_ple_urgent(bool)`. stage_batch passes ple_urgent_ to gather_batch (default false).
- `int device() const`.

### src/kernels/ngram.cpp + include/strata/kernels/ngram.hpp — PleTable::Impl 202-226, wait_prefetches 390-396, prefetch_rows 398-406, issue/collect, gather_batch 507-548
- Impl: `bool concurrent=false; std::mutex mu; mutable std::atomic<uint64_t> bytes_read`.
- set_concurrent(bool), plus `bool concurrent_capable() const` (Mmap, or the reader threaded).
- Every public entry takes `std::unique_lock lk(mu, std::defer_lock); if (concurrent) lk.lock();`.
- gather_batch(rows, n, out, err, bool urgent=false): under the lock it checks pending, does the Direct-mode prefetch fast path and wait_prefetches_locked(). It then unlocks if the reader is threaded and does issue(urgent)+collect+decode outside the lock. Mmap mode: wait_prefetches_locked, unlock, then the read_row loop.
- With concurrent=false the behaviour is byte- and order-identical to today.

### src/ngram/ple_reader.cpp + include/strata/ngram/ple_reader.hpp — Job struct; Impl::pump 148-175; issue 421-482; new accessors
- Job gets `bool urgent=false`.
- issue(rows, n, out_raw, bool urgent=false): marks its jobs urgent and inserts them at the front of m.queue (in sorted order) instead of push_back.
- Impl::pump: `if (!queue.front().urgent && free_slots.size() <= urgent_reserve) break;` (urgent_reserve defaults to 0, so behaviour is unchanged).
- set_urgent_reserve(uint32_t) is clamped to max_inflight/2.
- threaded() accessor.
- stats: an urgent_wait_us sum, for the stats line.
Read order changes only; each job writes only its own ticket's destinations.

### src/core/mtp.cpp + include/strata/core/mtp.hpp — new reserve_prefill_records near prepare_prefill 1119; stream creation 419-427
- `bool reserve_prefill_records(int64_t n, std::string& err)`: OnDevice; per_row = 1+4+n_head; if pf_cap_ < n*per_row, cudaFree + cudaMalloc (same code as 1164-1172). Called only from overlap_prepare (drained, T_main).
- set_stream_priority_normal(bool) (A/B knob): when set before init, create cs_ at the default priority instead of force_on_'s greatest one.

### src/prefill/prefill.cpp + include/strata/prefill/prefill.hpp — run_impl 2030-2040; stage thread lambda 3905-3907; new const methods
- Factor `bool ple_batched_() const` out of the ple_batch predicate and use it in run_impl.
- `bool overlap_safe(std::string& why) const` walks this stage and next_: refuses pp/help_pp, cpu_pool_ != nullptr, and (ple_on && !ple_batched_()).
- `int device() const { return impl_->device; }`.
- The stage thread lambda: `bool r = next_->run_impl(...); next_->cuda_err_.store(cudaPeekAtLastError()); return r;`.
- `cudaError_t chain_cuda_error() const` returns the first non-success over the chain.

### tools/batch_overlap_test.py (new) — new test
Runs both splits: '--layer-split 13,25,37' (production) and '12,24,36'. Common flags: --trim-stage-weights --batch 16 --prefill 512 --pcie-frac 0 --adapt-every 1000000, plus --no-prefill-borrow unless M0 shows it costs all-resident (the gate already forbids loans). STRATA_IQ_MT_MIN=1, with STRATA_PIPE_OVERLAP=0 and =1.
Cases:
- solo references (120/170, 1.2K, 5K prompts, turn and root checkpoints);
- staggered admissions with 12-15 slots decoding;
- BSTOP mid-read;
- STOP of the admitted request mid-read;
- slot reuse (copy_from_slot);
- BYIELD mid-read;
- a 64-request soak;
- a forced pad-up arm (STRATA_PIPE_OVERLAP_FORCE_PADUP=1, debug: always run S=GS during reads);
- a forced-wait arm (STRATA_PIPE_OVERLAP_FORCE_WAIT=1, debug: never start a group whose natural layout is missing).
Every slot must equal solo, and the env 0 and 1 runs must be token-identical. The log must show 'overlapped read of' for the eligible admissions, no 'not captured before an overlapped read', and no 'captured the batch window' line between a read's start and its stats line.

### src/ngram/ple_reader_test.cpp — new cases
- Concurrent PleTable::gather_batch with set_concurrent(true): one thread does 512-token batches, another a loop of 4-token urgent batches. Results are byte-identical to single-threaded ones, bytes_read is consistent, no unknown ticket.
- The urgent ticket completes before the bulk tail with reserve=16.
- concurrent=false and reserve=0 give the same order as today.
- The test is run under TSAN.

### serve/server.py (optional, M6) — 1234, 1314
Do not send BYIELD when getattr(self,'slot_gs',0). Cosmetic.

### docs/BATCHING.md — Limits (~79-84)
Document STRATA_PIPE_OVERLAP: its gates, layout pre-capture and pad-up, the knobs (SLEEP_US, PRIO, PLE_RESERVE, JOIN_S, MTP_PRIO), the stats line, and that BYIELD has no effect when piped.


## Shared-state audit

- **Main sessions ss / stages[k]->ss: GDN state, QSA pools and indexer, ple.hist, ple_prev** (The prefill chain (T_read, T_stage*) writes them. Batch windows use slot sessions (verify.cpp:751, 993-1030, 1113-1226, 905; ple_post excluded when batch_rec_, 878; batch_launch updates the slots' ple_prev 2793-2798).): Main-session reads on T_main during the read. → Only sp.run is overlapped. checkpoint_at, read_windows, the first window and copy_to_slot run after join plus drain, in today's order.
- **ss.ple.scratch (CUDA0)** (The stage-0 batch graph (verify.cpp:829, 909, 911). The prefill's per-token PLE fallback (prefill.cpp:2449).): Device data race, so outputs are not exact. → Gate: overlap_safe requires ple_batched_() (the predicate is shared with run_impl). The batched path uses m.region and ss.ple.hist only.
- **ss.moe (router fallback verify.cpp:1310-1318, per-token combine fallback 1501) and ss.block (last stage per-token head fallback 1591)** (Batch graph fallbacks only. The prefill uses neither (critic 2 confirmed).): none → none; listed for completeness
- **ss.ple.w / consts, weights, head, embeddings** (Both sides, read-only): none → none
- **Batch graph maps exec_bm_/commit_bm_/bm_used_, capture state, instantiate_evicting LRU (verify.cpp:2346-2521, 2556-2566)** (T_main only (stage_batch from pump).): A lazy capture during the read: VRAM contention with the stage threads, eviction churn, a synchronizing upload stalling the card, and capture overlapping other threads' implicitly synchronizing calls. → overlap_prepare captures the full layouts with the pipeline drained. During the read pump runs only groups whose layout exists on every stage (natural S, else padded up to GS, else the group waits). set_no_capture(true) makes stage_batch fail loudly instead of capturing, and also blocks the batch_graph_limit_ eviction. Tests check that no capture line appears inside a read.
- **Free VRAM on each card (CUDA3 tightest)** (T_main (graph capture, now only with the pipeline drained). The stage threads: mtp pf_dev_ growth (mtp.cpp:1164-1172), the MMQ pool, the cuBLAS workspace. split_scratch on T_main (sample_rows).): An OOM in the middle of a read. → No captures during the read. pf_dev_ is pre-grown in overlap_prepare (reserve_prefill_records(sp.chunk())). grp_host is allocated in Prefill::init (prefill.cpp:990/1068), not per read. cpu_x and help buffers are gated off (cpu pool, helper). The MMQ pool and cuBLAS sizes are reached by the first admission, which is never overlapped (no slots decode at startup, so batch_on() is false). split_scratch is sized per stream on the first non-greedy row, never freed, under its own mutex. M5 watches CUDA3's VRAM.
- **Device capture state vs implicit syncs** (MTP graphs (captured at startup by prepare_prefill, generate.cpp:6930) and verifier graphs (captured in overlap_prepare, drained).): cudaFree/cudaFreeHost on one thread while another thread captures on the same device. → No thread captures during an overlapped read: T_main is blocked by no_capture, the prefill never captures, and MTP is pre-captured. pf_dev_ cudaFree is removed by the reservation.
- **CUDA per-thread last-error state (#224 fast exit, generate.cpp:9966)** (T_read, T_stage*, T_main): A prefill fault is recorded on T_read or a stage thread, so T_main skips the fast exit and hangs while unwinding. → T_read records wcuda. Each stage thread records cuda_err_ (Prefill::chain_cuda_error). ov_cuda_fault feeds the exit check, and the read returns without pipe_drain on a fault. A fault seen first by pump's cudaStreamQuery sets T_main's own last error.
- **Function-static per-device caches (fused_gr.cu:748-768/1686, prefill/kernels.cu:699, sampler.cu:1200)** (The prefill threads (one per device, as today). T_main reaches the fused_gr ones only through record_window, which cannot run during a read (no_capture), and sampler ok[] only through sample_rows; the prefill and mtp.prefill never sample.): Two threads first-touching the same device element. → Excluded structurally. Optional hardening: std::call_once or atomics in M6, not needed for correctness.
- **PleTable (generate.cpp:2939) and PleReader queue, in-flight slots, RowCache** (T_main (stage_batch) and the prefill's gather threads (prefill.cpp:2052, 2124, 2149).): Unguarded bookkeeping. FIFO/in-flight starvation of decode rows (256 in flight). Cache eviction. In sync-submit mode the lock would be held across a bulk read. → set_concurrent(true) (env only): mutex plus atomic bytes_read. urgent front-queue plus an in-flight reserve (16). Gate: Mmap or a threaded reader. The stats line reports the stage-0 gather ms and the hit rate. Order-only, so the same bytes.
- **Stage Verifier objects (arena, cs_, copy_, h_* flags, head_logits_, slot_sp_, b_running_, released_)** (T_main only.): A solo window during the read. A stage-0 group spinning on the PLE flag if T_main stops polling (abort). → Solo windows run only after the drain. On abort, T_main or OvGuard calls release_gpu_waits(5000) on every stage before the bounded join. The engine then exits (released_ blocks further windows).
- **Expert cache, host_res, d_res, res_upload, pending swaps** (Prefill (reads). Batch windows (resident_plan). Writers: lend/refill, apply_pending, adapt_tick, VRAM command.): A table changes mid-read. A stale ar_on or a stale all_experts_resident const (7395). → Live gate: all_of(host_res), pending.empty(), ar_live() per stage (refresh_ar). No loans. pump skips apply_pending while ov_active. adapt_tick is not called from pump. VRAM/SAVE lines stay queued until after the read, and a VRAM change is refused while slots decode (8716).
- **CPU pool / ExpertSource / peer / remote / fs_swap** (Prefill streaming, batch doorbell service (!ar_on)): Not thread-safe. → Gated: share_pool is false with --batch (6226), cpu_pool_ is null in overlap_safe, every stage is ar_live, and peer/remote/fs_swap are off.
- **Prefill hand-off and helper buffers, next_run_ futures** (The prefill chain only): none with the helper off → overlap_safe refuses pp/help_pp
- **MTP drafter (CUDA3): its top-priority cs_, pf_dev_, draft K/V** (on_chunk on the last stage's thread (mtp.prefill)): Competes with decode at equal top priority; pf_dev_ regrowth. → pf_dev_ is reserved. The priority interplay is measured in M4 (STRATA_PIPE_OVERLAP_MTP_PRIO=0 arm). Piped batch windows build no drafts (8564-8569).
- **checks / part_at / pp_* / ckpt_why / cur / ids** (on_chunk / on_stage_chunk (stage threads, part_mu)): T_main reading them during the read → pump and scan_ctl_lines touch none of them; they are read after the join.
- **`err` string (it is read_part's `e`, generate.cpp:9952)** (pump writes it on failure (T_main). The worker writes only `we`.): Lost reason text; double ERR. → On a pump failure err keeps pump's text, `we` goes to stderr only, and ov_err_printed suppresses the caller's second ERR.
- **bs[], pg[], stage_group, rr, bt_* counters, progress().busy** (T_main only): Pad rows (now including pad-up rows) write inactive slots. The all-idle block resets busy inside BusyScope. bt_* are never reset if every slot ends mid-read. → copy_from_slot runs before the read, copy_to_slot after the drain, and pad-up rows get sl.cached=false. The all-idle block is skipped while ov_active, and idle_report(false) runs after the drain.
- **stdin queue (in_lines, in_mu, stop_req)** (T_stdin, T_main, T_read (should_stop)): BSTOP delayed; BYIELD extra drain → scan_ctl_lines applies BSTOP and drops BYIELD. stop_req also serves as the abort signal.
- **Hardware work queues per device (CUDA_DEVICE_MAX_CONNECTIONS) and SM residency of persistent prefill grids** (All streams on a card): False dependencies and no preemption, which limit how much priority helps (performance only). → M4 A/B with CUDA_DEVICE_MAX_CONNECTIONS=32 and PRIO 0/1, judged on x. It never affects results.
- **stdout/stderr** (T_main (BT/BDONE/ERR), the last stage's thread (PP), all threads (stderr)): Interleaving → Line-atomic printf. server.py routes BT/BDONE by prefix (766-770). Exactly one ERR per failure.

## Exactness

Exactness settings: --pcie-frac 0, --adapt-every 1000000, no loans (--no-prefill-borrow, or the gate's own no-loan rule if M0 shows that flag costs residency), STRATA_IQ_MT_MIN=1, plus the gates. Under these, the overlap changes only WHEN things run, and which pad rows a window carries.

1) The admitted request. It is the same single sp.run over [a0, b0) with the same chunking as today's piped path and the solo reference. Its inputs are untouched by the concurrent windows:
- main-session state, which only the prefill writes;
- its own buffers (no loan);
- PLE rows: the same file bytes whatever the queue order or reserve;
- the expert tables, frozen: no apply_pending, adapt or lend during the read, checked live by the gate;
- read-only weights.
The header windows, the first window and copy_to_slot run drained, in today's order.

2) The decoding slots. Each window is a captured graph for its row layout, run on slot sessions. Every main-session buffer the batch graph can touch was audited:
- scratch: gated to the batched PLE path;
- ss.moe and ss.block: fallbacks the prefill never touches;
- consts and weights: read-only.

3) Pad-up and waiting are the only scheduling differences.
- Waiting a group only delays it, and its per-slot sequence of windows is unchanged.
- Pad-up runs a slot's row in an S = GS layout instead of S0 < GS. A row's result does not depend on S. Today's exactness already relies on that: G.S changes over a slot's lifetime whenever the highest active slot of its group finishes or is admitted, and batch_test16/32 and batch_interleave_test still match solo, which itself runs at another width.
- The extra pad rows write only inactive slots, as today's pad rows below G.S do. Those slots get sl.cached = false, so no stale state is reused. admit_slot is overwritten by copy_to_slot after the drain, and the source slot was already copied before the read.
- The forced pad-up arm in batch_overlap_test (STRATA_PIPE_OVERLAP_FORCE_PADUP=1) tests this claim directly.

4) Concurrency cannot change bits. Kernels are deterministic per launch. No kernel choice depends on load: gdn_rec uses static SM counts, cuBLAS is prefill-only, and per-device statics are fixed once set. Stream priority and hardware queue mapping change only scheduling. No graph is captured during a read, so every window replays a graph recorded with the pipeline drained, from the same code as today's lazy capture.

5) What is gated off because it would break exactness:
- loans or non-resident experts (CPU rounding);
- the per-token PLE fallback;
- the stage helper;
- --kv-resident;
- images;
- a control vector;
- peer, remote or fs_swap;
- debug-check envs;
- a sync-submit PLE reader (performance, not exactness).

6) Empirical check, with STRATA_PIPE_OVERLAP=1 on both splits:
- every slot and every admission equals solo;
- the env 0 and 1 runs are token-identical;
- the forced pad-up and forced wait arms are identical too;
- the existing batch tests are unchanged with the env off.

With production adaptive swaps (--adapt-async), outputs were never bit-exact run to run (generate.cpp:7596), and that stays so.

## Test plan

- M0 (no code). Run the production split 13,25,37 and also 12,24,36, with STRATA_TRACE=1, under both the production flags and the exactness flags. Confirm 'keeps its own prompt buffers' x4, no 'borrows', and '100% VRAM resident' on every stage verifier, also with --no-prefill-borrow (if that flag costs residency, run the overlap exactness tests without it; the gate already forbids loans). Record the PLE mode (Mmap or threaded reader). Per admission, split the R stall into: BGEN drain, the sp.run 'read N tokens (batched) in X ms' segment (the overlappable part), header windows, checkpoint, first window (11 ms), copy (8 ms steady state, 60 ms on first use), and refill. Record D_ref: rows/s of decode-only stretches with 15 and 16 active slots.
- Unit (host-only, TSAN): ple_reader_test concurrency. With set_concurrent(true), bulk 512-token and urgent 4-token gathers on two threads give byte-identical results and consistent bytes_read. The urgent ticket completes before the bulk tail with reserve=16. With concurrent=false and reserve=0 the order is unchanged.
- Env off (regression): batch_test16.py, batch_test32.py and batch_interleave_test.py on both splits stay identical to solo. No 'overlapped read' or 'ensure' lines appear in the log. Solo prompt and decode speeds are within noise.
- M2 gate dry run (env on, read_part still falls back after the gate plus prepare): the base config logs 'eligible', and overlap_prepare logs its one-time layout captures. Each of these logs its refusal reason and gives today's results: STRATA_PLE_BATCH=0, STRATA_PREFILL_HELP=1, --kv-resident, BGENI, STRATA_SPLIT_OWN_BUFFERS=0, --ple-sync-submit, STRATA_IDX_FP16_CHECK=1, and a forced VRAM shrink (a live residency failure). A HIP build compiles with the feature compiled out.
- Exactness with the env on: batch_overlap_test.py on both splits. Prompts of 170, 1.2K and 5K tokens (with turn and root checkpoints) are admitted while 12-15 slots decode. Every slot and admission must equal solo, and the env 0 and 1 runs must be token-identical. Repeat with STRATA_PIPE_OVERLAP_FORCE_PADUP=1 and with STRATA_PIPE_OVERLAP_FORCE_WAIT=1 (both identical). The log must show no 'captured the batch window' between a read's start and its stats line, and no 'not captured before an overlapped read'.
- Control lines during an overlapped read: a BSTOP ends its slot at the next window (BDONE cancel); a STOP of the admitted request gives DONE cancel with the other slots exact; a BYIELD is dropped with no extra drain. Slot reuse (copy_from_slot before, copy_to_slot after) equals solo. A 64-request soak through 16 slots.
- Fault paths. (a) A debug env makes batch_poll return -1 mid-read: exactly one ERR, release_gpu_waits runs, the worker ends via stop_req, clean exit, no watchdog. (b) A debug env throws std::runtime_error from scan_ctl_lines mid-read: OvGuard joins, no std::terminate. (c) A debug env makes Prefill::run fail on stage 2: ERR with the prefill's text, and the engine stays usable (if no CUDA fault). (d) An injected CUDA fault on the prefill side (debug kernel trap): ov_cuda_fault triggers the #224 _Exit within seconds. (e) A debug env makes the worker sleep past STRATA_PIPE_OVERLAP_JOIN_S=2: bounded _Exit with a stderr line.
- Performance, M4. From the stats line: x = lost_ms / read_M0_ms per prompt class (read_M0 from M0). Report the stage-0 PLE gather ms and hit rate during reads. Arms: PRIO 0/1; CUDA_DEVICE_MAX_CONNECTIONS 8/32; SLEEP_US 0/20/50; PLE_RESERVE 0/16; MTP_PRIO 1/0. Take an nsys trace of one 1.2K admission per winning arm, to check that decode graph kernels interleave with MMQ and how long persistent moe_fused grids block them.
- End to end, M5: bench_concurrency.py through server.py with 16 and 24 users, 170-token and 1.2K prompts, max-tok 512, env off against on, 3 runs each, on the production split. Report throughput, TTFT p50/p95, ERR/watchdog/OOM counts and CUDA3 VRAM.

## Expected gain

MODEL. Throughput = D / (1 + D*R/L), with:
- D ≈ 620 rows/s of pure decode;
- L ≈ 511-526 rows per request;
- R = the all-slot stall per admission.

Calibrated on the current log (strata-unsloth-ud-iq4_xs.log):
- short prompts: 403.5 rows/s gives R ≈ 0.44 s, with ~170-token prompts read in ~345 ms;
- 1.2K prompts: 294.6 rows/s gives R ≈ 0.94 s, with 'read in ~812 ms' including the header windows and 1 checkpoint;
- copy_to_slot: 8 ms in steady state;
- first window: 11 ms.

Corrected form: R_new = fixed + x * read_ov, with x = f*(1+s). Here f is the share of the decode rate lost during the read and s is the read's stretch. Decode is throttled for the whole stretched read, so x is what matters. If the two workloads simply time-slice the cards, then x = 1 and there is no gain.

Working split, which M0 refines:
- short: fixed ≈ 0.14 s, read_ov ≈ 0.30 s;
- 1.2K: fixed ≈ 0.22 s, read_ov ≈ 0.72 s.

Results:
| x | 170-token prompts (today 405) | 1.2K prompts (today 294) |
|---|---|---|
| 0.5 | ~458 (+13%) | ~365 (+24%) |
| 0.6 | ~445 (+10%) | ~350 (+19%) |
| 0.75 | ~430 (+6%) | ~325 (+10%) |
| 1.0 | 405 (0) | 294 (0) |

Both workloads are memory-bound, and Ampere cannot preempt running blocks (persistent moe_fused grids), so x ≈ 0.6-0.9 is plausible. The realistic outcome is +0-12% on short prompts and +0-20% on 1.2K. The earlier 'f = 0.2 gives 500 tok/s' figure is withdrawn.

Costs:
- The admitted request's TTFT grows by s*read_ov.
- Admissions stay serialized, so queueing delay for later requests also grows. M5 reports TTFT p50/p95.

The ceiling for this lever stays at pure decode (603-640), below vLLM's ~710 on short prompts.

GO/NO-GO (M4): continue only if measured x ≤ ~0.6 on short prompts AND ≤ ~0.75 on 1.2K (each about +10%). Otherwise keep only the independent wins: the BYIELD no-drain, and optionally the PLE reserve, which is useful on its own if stage-0 gathers queue behind prefetches. Solo and single-user speed are unchanged.

## Milestones

1. M0: baseline (no code). Split R into its phases on 13,25,37 and 12,24,36 under the production and exactness flags; confirm own buffers, no loans and 100% resident under --no-prefill-borrow; record the PLE mode and D_ref.
   - verify: A table of drain, sp.run read, header windows, checkpoint, first window, copy and refill per prompt class, consistent with R ≈ 0.44 s / 0.94 s. The resident and own-buffer log lines are present. A decision on whether the exactness test keeps --no-prefill-borrow.
2. M1: PLE concurrency (PleTable set_concurrent, mutex, atomic bytes_read; PleReader urgent front-queue, in-flight reserve, threaded(); Verifier::set_ple_urgent). All of it only switched on by setters, so it is inert with the env off.
   - verify: ple_reader_test passes, also under TSAN. Env-off batch tests are identical. No change in solo or batch speed with the setters off.
3. M2: gate and prepare plumbing. Prefill::ple_batched_/overlap_safe/device/chain_cuda_error; Verifier::ar_live/has_batch_layout/ensure_batch_layouts/set_no_capture/device/stream priority with fallback; MtpDrafter::reserve_prefill_records and the MTP_PRIO knob; live residency in overlap_gate; overlap_prepare. read_part still falls back after the gate and prepare (dry run).
   - verify: Builds on CUDA and HIP. Env on: 'eligible' for the base config, one-time layout captures logged, every refusal reason reproduced. Outputs are identical with the env on and off, and no 'not captured before an overlapped read' errors appear.
4. M3: the overlap. read_overlapped with OvGuard, catch(...), wcuda/chain errors, bounded join, release_gpu_waits on abort; pump's ov_active guards plus pad-up/wait at stage-0 start; idle_report(false); ERR de-duplication and ov_cuda_fault in the #224 check; scan_ctl_lines; BYIELD no-drain behind the env; the stats line with lost_ms, PLE gather and hit rate.
   - verify: batch_overlap_test passes on both splits, including the forced pad-up and forced wait arms. Fault cases (a)-(e) behave as specified. No capture lines inside reads. The watchdog stays quiet. Every eligible admission prints its stats line.
5. M4: measure x per prompt class across the arms (PRIO, CUDA_DEVICE_MAX_CONNECTIONS, SLEEP_US, PLE_RESERVE, MTP_PRIO), plus an nsys trace of one 1.2K admission.
   - verify: x ≤ ~0.6 on short prompts and ≤ ~0.75 on 1.2K for the best arm means go to M5. Otherwise stop, and keep only M1 (if the stage-0 gather latency justified it) and the BYIELD fix.
6. M5: end-to-end server benchmark (16/24 users, 170 and 1.2K prompts, max-tok 512, 3 runs each, env off against on, production split), with TTFT p50/p95 and CUDA3 VRAM.
   - verify: About +10% throughput on both classes, an acceptable TTFT p95, no ERR, watchdog or OOM, and the exactness suite green on the final build.
7. M6: polish. server.py skips BYIELD when slot_gs; BYIELD no-drain unconditional (separate commit); docs/BATCHING.md; optional call_once hardening of the per-device statics; consider default-on for CUDA/Linux piped mode (separate commit).
   - verify: Server tests are unchanged, a manual run sends no BYIELD in piped mode, and the results are identical.

## Open questions

- What x do we actually get on the 3090s? It decides the lever. Priority cannot preempt the persistent moe_fused / gdn_rec_kh grids, so the PRIO arm may lower f by raising s and leave x unchanged.
- Is pad-up (running a group at S = GS while its natural layout is missing) ever exercised in practice once overlap_prepare has captured the full layouts? Or is the natural layout almost always present, because steady-state layouts are captured before the first overlapped read? If pad-up is common, its extra rows cost decode time, which raises x.
- Does ensure_batch_layouts' recapture after LRU eviction (outside reads) cause visible churn on CUDA3 at 16 slots? A larger --vram-reserve-mib or pinning the 4 full layouts per stage would avoid it, at the risk of new OOMs for lazily captured smaller layouts.
- Does the PLE in-flight reserve of 16 hurt the prefill's SSD throughput (a longer read, higher s), or is Mmap/RAM PLE mode (--ple-io ram) the better fix when RAM allows?
- A 1.2K read stays one serialized admission with a stretch s. Is a second admission path (a second main session plus a prefill chain, about 0.4 GiB per card) worth it later?
- Should checkpoint_at between cuts run without the final drain of each segment (lazy drain only before read_windows, the first window and copy_to_slot)? It would save 10-30 ms per admission with a root checkpoint.
- Deferred first token (admit with x = ids[n-1], so the first batch row produces token 1) would remove the first window and a drain, but changes the BGEN/BADM protocol and penalty handling (penalty_last_n = 0 in batch sampling, verify.cpp:2774).
- Production configs with --kv-resident and --adapt-async remain gated off. They need a separate audit of K/V streaming and swap timing.