# Upstream submission + benchmark plan (research workflow wf_91db16d8-cd1, 2026-11-08)

{
  "summary": "Research how to upstream our v0.2.12 port changes to Strata and prove the speed gain with upstream-comparable benchmarks",
  "agentCount": 9,
  "logs": [],
  "result": {
    "plan": "**Plan: submitting our v0.2.12 series upstream and showing "we are faster" (Niko1221/Strata)**

Sources: the four scout+verify results, `/media/raid0/StrataQ38FN/linux/window-212.sh` header lines 1-78, `/media/raid0/StrataQ38FN/Strata-212/docs/BENCHMARK_PROTOCOL.md` lines 1-60 and `/media/raid0/StrataQ38FN/Strata-212/tools/strata_bench.py` CLI lines 1031-1037. Claims the verify step marked wrong are left out or corrected.

### 0. Bottom line
- **Yes, submitting is worth it.** Upstream has ported our work three times with credit:
  - the #2364 series via merge df44bcb7, with 29 commits still authored by havok2-htwo;
  - the HC_Q8 fix as abf2953b, co-authored, credited in the v0.2.12 notes;
  - db568409, which folds in our batch_test16/32.
- **Upstream already depends on us.** Since v0.2.9 (dd471d31) six of the nine default "exact parallel stack" switches are ours: CKPT_POOL, SLOT_CKPT_MOVE, SLOT_COPY_DIRECT, SLOT_COPY_BATCHSYNC, BATCH_SELF_COMMIT and QSA_SLOTS. The v0.2.9 headline "1.45x at 8 requests" does not name us.
  - So stock v0.2.12 already contains much of our work.
  - The honest pitch is the part that is not upstream yet: P9 pipe overlap, P10 solo pipeline + spec lanes, P11 read yield, the RAM/cache series and the vision per-slot fix.
- **We cannot claim "faster than stock" in public yet.** The only valid number is PT against our own old production A (w212c, TESTLOG #99):
  - agg16: +3.8 % short, +3.2 % long
  - decode at 1/2/4 users: +5.0 / +5.2 / +4.7 %
  - 62/62 replies identical
- **Why the stock numbers are invalid.** Every stock arm (B, C, E-I in w212, TESTLOG #98) ran with presence_penalty 1.5. That makes "PT 529 / C 529 / B 486", the 5.8 s join gap and the stock decode/TTFT unusable.
- **Most of PT's lead over P comes from upstream's own sm_86 table.** PT vs P is +4.7 % short and +3.3 % long. No single feature of ours has a gain measured on v0.2.12.
- **PT is not ahead on every metric**, even against the penalised stock runs:

| Metric | B / C (stock, penalised) | PT |
|---|---|---|
| Short TTFT p95 / max | 3.27/3.66 s and 3.19/3.57 s | 4.28/4.81 s |
| Follow-up TTFT p50 | 1.87 / 2.07 s | 2.43-2.57 s |
| TTFT of the cold 32K joiner itself | C 5.83 s | 8.15 s |

  The joiner is slower because the overlap shifts the cost onto it (TESTLOG #81: 7.7 → 10.7 s). Any report must state these tails.
- **Deploy check for the main agent.** Commit 8e703929 says production on a v0.2.12 engine must set `STRATA_SERVE_STOP_BOUND=1` to keep today's behaviour. Arm I's env adds it, and P/PT inherit I's env (window-212.sh header). Confirm it is in the deployed PT config (a713392 / deploy-212pt).

### 1. Should we submit, and in what order?
Small PRs, one problem each (CONTRIBUTING.md:17).
- Focused fixes and benchmark reports can go straight to a PR (CONTRIBUTING.md:16).
- Features and default changes need an [Engine] issue to agree scope first (CONTRIBUTING.md:14-16).

**Wave 0: no code (lowest risk, highest visibility)**
- **(a) Evidence comment on open #3354, optionally #3295.**
  - Content: a8091f4d/#2685 brings back #2940 (closed, ~25 % slower decode). The default presence_penalty 1.5 silently turns off the solo pipeline, lanes, ADMIT_MERGE_FIRST and EARLY_SAMPLE.
  - It also penalises upstream's own tools: strata_bench.py:141-145, ab_engine.py:45 and piped_read_join_bench.py:68-70.
  - Only post this after the penalty-free stock rerun (section 3, R1), so we have stock with and without the penalty on the same binary.
- **(b) A [Benchmark] results-only PR**: 4x RTX 3090, stock v0.2.12 vs PT (section 2). Upstream asks for multi-GPU NVIDIA reports (TEST_REQUESTS.md:14, #880) and an hour-plus soak (TEST_REQUESTS.md:17, #848).
- **(c) Optional follow-ups we promised and never did:**
  - answer Vop3p's 4x3090 test request on #2326 (issuecomment-6255353503, about #3103 STRATA_PIPED_CONCURRENT_SMB);
  - the small test-fix PR offered on #2364 (tools/batch_groups_cache_test.py:83-84 is unchanged). This now matters more, because SLOT_CKPT_MOVE is default-on since v0.2.9.

**Wave 1: direct fix PRs, [Engine], each small**
- **PR1 abbaf026**: NVFP4 guard for MMQ_DIRECT (prefill.cpp +3).
- **PR2 8e703929**: STOP_BOUND opt-in on CUDA + the `produced_n < max_new` guard, which also fixes `outp[-1]` on HIP's default path. Goes in the open #4677 thread.
  - Blocker: it touches the HIP path, and CONTRIBUTING requires building every backend a change touches (CUDA, HIP, SYCL). We only build CUDA. We need a HIP build check, or must say so under "Not run".
- **PR3 17b1102b**: ProgressWriter. It refers to #4271, which is closed. Reword the subject, which does not mention ProgressWriter.
- **PR4 6151bde2 + d3bc1da0**: test fixes. The snapshot-test flake went from 4/10 to 0/30 (PLAN-0212-port.md:259).

**Wave 2: vision correctness (high value)**
- #1242 is a PR (ilumn's, not an issue). It was closed by the tracker reset on 2026-10-19 and brocktice reproduced the bug on v0.2.2. No M-RoPE batch fix exists in the 146df3c4 log.
- Open an [Engine] issue first, because 1661c314 changes default behaviour (it refuses mixed image/text rows). Then:
  - **PR5**: guard + server image-alone path (1661c314, c0cdd140, 69193088);
  - **PR6**: STRATA_MROPE_SLOTS=1 per-slot regions (da4b2ee3) + vision_batch_check (3591ee27). Scrub the private paths in vision_check_smoke.py:43 and :49.
- Before citing, run the vision check on stock to confirm #1242 is still present (R5).

**Wave 3: RAM / conversation cache, opt-in, issue first**
- **PR7**: CONV_MMAP + GATE_EVICT park half (bd2fb145, ac6f8e66; #3115). Peak RSS 148.0 → ~90 GiB (#95).
- **PR8**: PARK_CAPTURE (6eb77581 + e50204d2), framed as the zero-VRAM sibling of upstream's STRATA_PARK_ASYNC (f73b3c86, linxuhao). Fix the 6eb77581 subject, which still says PARK_ASYNC. Figures (#68, 0.1.44.1): +7.2/+6.1 %, park stall 70 → 20 ms.
- **PR9**: SLOT_PARK + 0873de12 + 3261acbe, in the open #2526 thread.
- **PR10**: borrow on a split (3daa2f24, cc84bdfb). Upstream gates borrow on `stages.empty()` at Strata-212/src/program/generate.cpp:16205.
  - The evidence is production data (#94: 73.3 → 91.7 % of prompts ≥100K reused, 118 borrows), not an A/B.
- **PR11**: CONV_TIER. Low priority: it is large and gives +25 % only under forced overflow.
- Hold SLOT_CHAIN_REUSE, GIVEBACK and MSG_CKPT_TAIL; there is no single-feature figure for them.

**Wave 4: speed (the core of "faster"), only after the v0.2.12 ablations (R3)**
- **PR12 PIPE_OVERLAP(+KVS)**, pitched as an extension of PIPED_CONCURRENT.
  - PIPED_CONCURRENT is off beside in-place GGUF experts (include/strata/program/batch_read.hpp:44-55), and arm E did not engage.
  - Split it into PLE setters, hooks, generate and KVS.
  - First explain the batch_overlap FAIL on PT: `ERR BGEN: no such free slot` in results/w212b-penalised/overlap-PT.txt.
- **PR13 SOLO_PIPELINE, then SPEC_LANES**, in the open #3794 thread ("MTP drafts in the pipelined batch-slot path").
  - #1253 was declined as serial-split batch MTP, and the maintainer wrote: "A pipelined batch-MTP variant with a measured gain would be welcome." Ours is pipelined, so it is not a resubmission under CONTRIBUTING.md:19. That this matches the invitation is our reading, not the maintainer's.
  - fed5a555 (+1529 lines) must be split.
  - Needs the lane VRAM calibration; PT's lowest free VRAM is 814 MiB.
- **PR14 + PR15**: READ_YIELD (engine) together with ADMIT_BY_NEW/YIELD_SHORT (server, 74711179 + a88014b7). Precedent: our ADMIT_SHORT_FIRST, ed4fa42a, was accepted.

**Wave 5: optional**
- QSA_SLOTS_STREAM is roughly matched by upstream's ATTN_DAILY (#98/#99).
- GR_DOWN_PF (+1.4/+1.8 %) and EARLY_SAMPLE (never measured alone) fall below the ~3 % bar.
- SPLIT_OWN_RESERVE_MIB: the +4.9/+3.6 % in #79 is the whole s28r config, not this switch alone.

**Never submit**
- the Genesis replay fixture (419a4dc9, serve/fixtures/frontend_90f73293.py, private);
- GET /v1/load as it is: it clashes with POST /v1/load at server.py:8262; propose /v1/load_status instead;
- PREFIX_DIAG, PIPE_OVERLAP_MAX_QUEUE and the other items on the PLAN-0212-port.md:289 list.

**Commit-message cleanup**
- Remove "port <sha>", "our ...", "TESTLOG #" and similar.
- Write problem-first (CONTRIBUTING.md:56) with an area prefix.

### 2. Upstream's official benchmark formats, and how to show "faster" honestly
**Format A: standard benchmark, `strata bench run`** (BENCHMARK_PROTOCOL.md)
- Still a draft: protocol id strata-qwen38-thinking-v1-draft.1, "pilot validation pending".
- Workload: concurrency 1, thinking on, T 1.0, top_p 0.95, top_k 20, presence_penalty 0 sent explicitly, seeds 1-3, 512-token cap.
- Requests: 1 warm-up, then short / 4K / 16K x3, then 12 sanity checks at 2048. `--optional-concurrency` adds 4 clients x 3 rounds.
- Metrics:
  - decode = predicted_n / predicted_ms, rejected drafts excluded;
  - prompt tok/s = fresh tokens only;
  - each case reported as median / min / max / n, never pooled;
  - a cache-hit limit applies, otherwise the run is nonconforming;
  - restart the engine between runs.
- Takes 2-4 min, plus 1.5-2 min for the optional stage, plus 1-5 min of model hashing.
- A modified build goes under `--track custom` and needs a same-machine baseline.
- Submission: results-only PR adding `bench/results/<UTC-date>-standard-<hw>-<run-id>/` (README, results.json, sanitized config, engine logs, outputs.jsonl).
  - Title must be `[Benchmark] <GPU> / <model> / <quant> / <protocol-id>`. The doc's `benchmark:` form fails .github/workflows/title-area.yml.
- This format is immune to the presence_penalty trap. It barely shows our multi-user gains, though it does show the 1-user decode lead.

**Format B: quick community check, `./bench.sh` / tools/strata_bench.py** (COMMUNITY_BENCHMARKS.md)
- Greedy, thinking off, `--runs 3`, chat / 4K / 16K / 5-turn agent, plus `--clients N` and `--long` (32K).
- The README template is only suggested; our own harness is allowed if fields and units are documented. Publish at least 3 runs with median and range (COMMUNITY_BENCHMARKS.md:175-176).
- On a v0.2.12 server this tool is penalised (server.py:6220-6223 merge order). Both arms need `"sampling": {"presence_penalty": 0}` in the config, and the report must say so.

**Board reality**
- No Discussions, no wiki, no pinned issues.
- Every [Benchmark] PR since the 0.2.0-beta reset is still open (#5428, #5463, #5396, #5352, #5556 and others). The newest bench/results folders in v0.2.12 are from 2026-10-20.
- Closest public comparison point: #2718, closed unmerged (filmo). Same setup as ours (4x 3090, UD-IQ4_XS, 262K), measured with strata_bench `--clients 1 --long`: 113.3 tok/s decode.
- Content that lands keeps the contributor as git author (2d49e617, 6024082a and our #2364 commits). The verify step corrected the scout's "his own commits".

**Honest framing rules for our report**
1. Same machine, same day, same window, arms interleaved. A drifts about 1.1 % between windows (514.8 vs 509.1), so do not mix w212 and w212c numbers.
2. Two stock arms:
   - **S (new)**: stock v0.2.12 with the config its own setup generates for 4x3090. This is "as shipped", and reviewers will expect it.
   - **C**: stock + our production args + presence_penalty 0, which isolates engine differences.
   - Show both. Say that S as shipped is also hit by #2685 when clients send no sampling.
3. Say plainly that stock already contains six of our switches (v0.2.9 defaults), so the delta shown is the not-yet-upstream part.
4. Exactness:
   - batch_test engine identity, 16/16 solo + batch (bt B=A, H=B, I=H, P=A, PT);
   - HTTP replies identical, 62/62 short (PT vs A). A penalty-free C should now match PT over HTTP too; check it. The old B-vs-A HTTP diff is explained by the penalty, not "other server code".
   - PTX is not exact and is shown as INFO only, or left out.
5. Report the tails where PT loses (section 0) and all failures: batch_overlap FAIL on PT, bgc FAIL (counting artifact), stock serve unittests rc 124, E not engaged.
6. List every changed setting per arm; the T212 manifest already has them.

### 3. Benchmarks still to run
All runs on 127.0.0.1:8091, never 8090. Every GPU window stops strata.service; coordinate with the docs/bench session first. A window costs about 5 min prep/stop + about 2 min restore, plus about 6.7 min per arm run (w212c measured: 76.7 min for 11 runs).

| # | What | How | Arms | Est. downtime |
|---|---|---|---|---|
| R1 | Penalty-free stock rerun (must-have) | `RUN=w212d NOW=1 ARMS="B C PT A" ./window-212.sh` (T212-B/C.json now carry presence_penalty 0, window-212.sh:351, 211827a). Run `--dry-run` first. Add a new arm S (setup-defaults stock): window-212.sh has none, so it must be added, which is a file change for the main agent. Keep IDENT_GATE=strict so C's HTTP replies are checked against PT and A. | S, B, C, PT, (A), x2 each with interleaving (R2_FIRST) | ~60-75 min |
| R2 | Upstream standard protocol | Per arm, start the server on 8091, then `strata bench run --url http://127.0.0.1:8091 --track default` (S) or `--track custom --track-notes "..."` (PT), with `--optional-concurrency`. Restart between runs. Check PT's INFO lines: do the solo pipeline and lanes engage at T 1.0? Unmeasured so far. | S, PT, x2 (3 if time) | ~45 min |
| R3 | Community quick check with 16 clients | `tools/strata_bench.py --url http://127.0.0.1:8091 --runs 3 --clients 16 --long`, both configs with presence_penalty 0 | C (or S + pp0), PT | ~25-30 min (est.) |
| R4 | Error bars / speed-pass grade | 5 interleaved rounds of S/C/PT; for wave 4, single-feature ablations on v0.2.12: PT minus PIPE_OVERLAP, minus SPEC_LANES/SOLO_PIPELINE, minus READ_YIELD | | ~2.3 h base (night) + ~7 min per ablation run |
| R5 | Stock vision check (#1242 still present?) | vision_batch_check against stock | S or C | ~5-10 min |
| R6 | Soak 1-2 h on PT (TEST_REQUESTS asks for it; also covers the VRAM risk) | after the deploy, under production traffic + monitoring, or on 8091 | PT | 1-2 h, or 0 if done live in production |
| R7 | Not GPU: batch_overlap FAIL triage, lane VRAM calibration, HIP build check for PR2 | | | — |

- **Suggested windows:**
  - one night: R1 + R2 + R5, about 2 h;
  - R3 and the R4 base can join the same night, about 4 h;
  - ablations in a second night.
- **Reusable as-is:**
  - all bt identity results (not affected by the penalty);
  - PT vs A (w212c, valid; useful as "our old fork vs new");
  - PT's vision GO 8/8 and 4/4 (results/w212c/PT-vision.txt). The "0 GO, 2 NO-GO" line in the summary is a parser artifact.
  - older per-feature figures, labelled as 0.1.43/0.1.44.1: #68, #70, #72, #77, #80-#84, #88/#89, #94, #95;
  - tools: window-212.sh, compare_replies.py, bench_multiturn.py, upstream's piped_read_join_bench.py (credible because it is upstream's own; set presence_penalty 0).
- **Not reusable:**
  - any B, C, E-I throughput, decode, TTFT or join numbers from w212;
  - community_bench_144.sh (0.1.44-era, port 8095, hardcoded Strata-143/144, stops production). Use it only as a template.

### 4. Risks
- **Process and tone**
  - The maintainer never replies to us directly. Items move through ports and release notes.
  - Bar for speed work: at least +3 % end to end over interleaved pairs (#1187, #1535, #1511), plus an identity gate with the switch unset (#2326).
  - Required for every PR:
    - area prefix and template sections (Problem and result / Related discussion / Approach / Validation as Passed / Failed / Not run / Limits);
    - item-check.js: a "Problem and result" of at least 20 characters, a Validation section and a tested revision;
    - intake.yml closes needs-format items after 5 days;
    - stale issues get swept (2026-11-05);
    - only the author or a maintainer can /reopen.
  - Expect PRs to be closed as "ported" rather than merged (#3559), which is fine.
- **Already upstream or reimplemented**
  - PIPED_CONCURRENT (#2326) was taken instead of our overlap.
  - PARK_ASYNC (f73b3c86) overlaps by name with PARK_CAPTURE.
  - ATTN_DAILY roughly equals QSA_SLOTS_STREAM.
  - PW_WITH_BATCH (d17c52e7) and PIPED_MTP (17878ac5, #4121) are alternatives to our lanes. Their #98 numbers are penalised, so "ours is better" is not proven yet; R1/R4 must include F and G if we claim it.
  - Our "do not use" memory for SLOT_CKPT_MOVE is stale: it is default-on since v0.2.9 (memory upstream-0145-port.md needs updating).
- **Declined**
  - #1253 serial-split batch MTP. Do not resubmit anything resembling it (CONTRIBUTING.md:19).
- **AI policy**
  - "AI assistance is welcome. The contributor is responsible for understanding the change and supplying real evidence." (CONTRIBUTING.md:3)
  - Never submit unproduced measurements (CONTRIBUTING.md:106).
  - No CLA or DCO found. The user must be able to defend each PR.
- **Identity**
  - Commits and PRs only as havok2-htwo <havok2-htwo@users.noreply.github.com>, never matthias@sons-of.de.
  - Every outward text (issue, PR, comment, report) needs the user's text approval before posting.
  - The paper author line (Matthias Weiner, Sons of Motion Pictures GmbH) is for the paper only.
  - Public-launch gate: no LinkedIn or announcement without the user's go. The paper v1.3, one-pager and LinkedIn post (2026-10-17/18) carry 0.1.44.1-era numbers and are stale. The LaTeX source was not found.
- **Technical**
  - Missing HIP/SYCL builds.
  - The batch_overlap FAIL.
  - VRAM headroom (814 MiB).
  - Unknown behaviour under T 1.0 sampling.
  - The maintainer re-measures on 4x R9700 HIP. Our CUDA-only gains may not reproduce there, so state the hardware scope.

### 5a. First PR outline (recommended first code PR: PR2, or PR1 if the HIP build is not ready)
**Title:** `[Engine] serve: STOP_BOUND on CUDA (opt-in) and a produced_n guard for the stop bound`
- **Problem and result:** the stop bound is HIP-only, and the default HIP path can read `outp[-1]` when produced_n reaches max_new. The fix adds the guard and makes STRATA_SERVE_STOP_BOUND=1 usable on CUDA. Default behaviour is unchanged on CUDA; HIP keeps its default plus the guard.
- **Related discussion:** #4677.
- **Approach:** the diff of 8e703929 (generate.cpp +11/-12), described in 2-4 lines.
- **Validation:**
  - Passed: CUDA build sm_86; the relevant tests; identity with the switch unset (bt 16/16); tested revision 146df3c4 + this commit.
  - Failed: list any.
  - Not run: HIP/SYCL build, unless done.
- **Limits:** measured only on 4x RTX 3090. No speed claim.

**Alternative first post (Wave 0a, comment on #3354):** stock v0.2.12 on the same binary with and without `presence_penalty 0`. Give the decode delta and which paths turn off. Note that strata_bench.py, ab_engine.py and piped_read_join_bench.py are affected, and point to #2940. Needs R1 data.

### 5b. Benchmark report outline
**Title:** `[Benchmark] 4x RTX 3090 / Qwen3.8-Flash-Next / UD-IQ4_XS / strata-qwen38-thinking-v1-draft.1 + multi-user (custom)`
1. **Summary:** 2-3 lines. Stock v0.2.12 vs v0.2.12 + opt-in patch series, same day and machine, identical greedy replies.
2. **Hardware and software:**
   - 4x RTX 3090 (layer split; PCIe/NVLink topology), power limit (320 W), CPU/RAM, driver, CUDA 13.0;
   - Strata 146df3c4, our branch head a099d295, binary sha256 values;
   - model files with SHA-256; full configs for every arm (sanitized), 16 slots, 262K context, --kv int8, --spec 4.
3. **Arms:** S (setup defaults), C (stock + matched args + presence_penalty 0), PT (ours). For each, the exact switch list and a note that stock already contains our v0.2.9 default switches.
4. **Standard protocol results (R2):** the per-case median/min/max/n table, including the optional 4-client stage; results.json and outputs.jsonl attached.
5. **Multi-user (custom track, R1/R3):**
   - agg tok/s at 16 users short and long;
   - per-user decode at 1/2/4;
   - TTFT p50/p95/max;
   - multi-turn follow-up TTFT;
   - cold-32K join (piped_read_join_bench): stream gap and the joiner's own TTFT;
   - peak RSS and VRAM per card;
   - every run listed, medians with ranges, interleaving order stated.
6. **Exactness:** batch_test 16/16 solo and batch; HTTP reply comparison counts.
7. **Sampling note:** the #2685 default and how both arms set presence_penalty 0, with stock-as-shipped numbers shown separately.
8. **Where PT is slower or equal:** TTFT tails, the joiner's TTFT, the VRAM headroom cost.
9. **Failures and not run:** batch_overlap, bgc, serve unittests timeout, HIP not built, soak status.
10. **Reproduce:** the commands, a link to the patch branch, and the scripts with documented fields and units.

**Open / unverifiable:**
- Whether the RTX PRO 6000 or 5090 hosts are still available for single-GPU runs: not found.
- Upstream commits after 146df3c4: none as of the check (`gh api commits`, main = 146df3c4).
- Comments on #5428 beyond the bot notes: not checked.
- Whether PT keeps its fast paths under T 1.0 sampling: unmeasured.",
    "results": [
      {
        "key": "bench-infra",
        "scout": {
          "summary": "Upstream Strata (origin https://github.com/Niko1221/Strata.git, v0.2.12 = 146df3c4) has two official ways to benchmark, plus a set of A/B tools for engine PRs. No CI workflow runs benchmarks.

1. **Standard benchmark** (`strata bench run`). Spec: docs/BENCHMARK_PROTOCOL.md, still a draft, protocol id strata-qwen38-thinking-v1-draft.1. It sends frozen fixtures from bench/prompts/standard-v1 (corpus SHA e75eff8b...): concurrency 1, thinking on, the model card's sampling (T 1.0, top_p 0.95, top_k 20, presence_penalty 0), seeds 1-3, 512-token cap. Cases are 1 warm-up plus short, 4K and 16K prompts x3, plus 12 sanity checks. There is an optional stage of 4 clients x 3 rounds. Output is results.json and outputs.jsonl, submitted as a results-only PR titled "[Benchmark] <GPU> / <model> / <quant> / <protocol-id>".
2. **Quick community check** (`./bench.sh`, BENCH.bat, tools/strata_bench.py). Spec: docs/COMMUNITY_BENCHMARKS.md. Greedy (temperature 0), thinking off, 3 runs; tests are chat, 4K/16K prompts, an agent session, and optional `--clients N` and `--long`. Output is a README in a template plus results.json under bench/results/<date>-community-<gpu>-<model>/.

For engine A/B work there are tools/ab_engine.py (restarts per arm, alternates arms, greedy), tools/batch_test.py (each slot must decode the same tokens as alone), tools/piped_read_join_bench.py and the serving suites of `strata bench run --suite serving-smoke / parallel-mixed-v1`. The serving suites are explicitly not standard submissions.

What a perf PR must show (CONTRIBUTING.md and AGENTS.md):
- before and after on the same hardware, same day, same workload, with every changed setting listed
- each individual run kept, including failures
- Passed / Failed / Not run kept apart
- new features opt-in, with the default path byte-identical to the last release
- every touched backend built (CUDA, HIP, SYCL)
- a title area prefix such as [Benchmark] or [Engine]; a bot enforces it

**Hardware and model coverage:**
- Upstream's own headline tables are single-GPU RTX 5070 (DETAILS.md "Speed (measured)") and 4x R9700 HIP (BATCHING.md 16-slot table, ported from our PR #2364).
- No upstream-measured 4x RTX 3090 multi-user numbers exist in v0.2.12. The only 4x RTX 3090 line is our STRATA_MMQ_DIRECT row (DETAILS.md:1534).
- docs/TEST_REQUESTS.md lists "Three or four NVIDIA GPUs" as wanted because they have no such rig.
- The repo has no Discussions and no wiki, and no pinned issues were returned. The de-facto "post your numbers" board is the stream of [Benchmark] issues and PRs. There are dozens, including a 4x RTX 3090 PR #5428 (filmo, single-user, UD-Q4_K_XL, still open). Search returned no merged [Benchmark] PRs, and #2718 is closed unmerged.
- Reports land when the maintainer ports them in his own commits (e.g. our #2364 became df44bcb7/79e5cd18).

**Our community_bench_144.sh does not target either official format.** It is a 0.1.44-era wrapper around our own run_strata_variant.sh, bench_multiturn.py and sweep.jsonl. It hardcodes Strata-143-prod and Strata-144, stops strata.service, and uses port 8095.

**Two caveats for the showcase:**
- Our own TESTLOG #99 says every v0.2.12-server arm of #98 was measured with presence_penalty 1.5, so its decode and aggregate numbers are invalid. That includes the stock baselines B and C quoted in the task (results/w212/summary.txt). A clean stock comparison at presence_penalty 0 has not been run.
- Both official harnesses set sampling explicitly (temperature 0, or the card's thinking sampling with presence_penalty 0), so the #2685 default does not apply to them.",
          "findings": [
            {
              "claim": "Upstream's standard benchmark is `strata bench run`, defined in docs/BENCHMARK_PROTOCOL.md and still marked draft (protocol id strata-qwen38-thinking-v1-draft.1, result format strata-standard-benchmark v1).",
              "evidence": "/media/raid0/StrataQ38FN/Strata-212/docs/BENCHMARK_PROTOCOL.md lines 10-25. The runner is strata_cli/bench_standard.py; fixtures are bench/prompts/standard-v1, corpus SHA-256 e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407 (commit e1ef9df2). It shipped in 0.2.1 (docs/release-notes-0.2.1.md:25).",
              "importance": "high"
            },
            {
              "claim": "How the standard protocol works: concurrency 1, thinking on, model-card sampling, seeds 1-3, 512-token output cap, 1 warm-up + 9 performance requests + 12 sanity checks. The optional stage is 4 clients x 3 rounds. The server's slot count is only recorded, and testing another capacity counts as another run.",
              "evidence": "Lines 68-78: enable_thinking true, reasoning_effort xhigh, T 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.0, repetition_penalty 1.0. Lines 106-121: the stage table (warmup 128 cap; performance short/prompt-4k/prompt-16k x3 at 512; sanity I1-I4 R1-R4 T1-T4 at 2048; optional 4x short x3 rounds). The prompts are 4,093 and 16,386 tokens (line 140). The runner rejects --concurrency, --repeat and --request-timeout (line 50).",
              "importance": "high"
            },
            {
              "claim": "How the standard protocol defines its metrics. Prompt tok/s = engine read_tokens/prompt_s, fresh tokens only. Decode tok/s = engine predicted_n/predicted_ms, counting accepted tokens and excluding rejected drafts. TTFT and total time come from the client's monotonic clock. Concurrent aggregate = the round's output tokens / its wall time. Results are medians, min, max and n per case, never pooled.",
              "evidence": "BENCHMARK_PROTOCOL.md lines 254-298: the metric table and the field mapping to usage/timings. The user payload may not come from the cache (cached_tokens <= max_shared_prefix_tokens + 8); otherwise the run is nonconforming (line 214).",
              "importance": "high"
            },
            {
              "claim": "How to submit a standard result: a results-only PR adding bench/results/<UTC-date>-standard-<hardware>-<run-id>/ with README.md, results.json, sanitized config, engine logs and outputs.jsonl, titled `benchmark: <GPU> / <model> / <quant> / <protocol-id>`. Engine changes go in separate PRs. A future bench/approved.json manifest is planned but does not exist yet.",
              "evidence": "BENCHMARK_PROTOCOL.md lines 448-505 (Submit and review, Future approved manifest).",
              "importance": "medium"
            },
            {
              "claim": "The quick community check is a separate format: greedy (temperature 0), thinking off, 3 runs, tests for chat, 4K/16K prompts, a 5-turn agent session, and optional --clients N and --long (32K). Its output follows a fixed README template.",
              "evidence": "/media/raid0/StrataQ38FN/Strata-212/docs/COMMUNITY_BENCHMARKS.md ('Run the benchmark' table, 'Report template'). tools/strata_bench.py:141 sends temperature 0, :146 sets enable_thinking False, :1035-1036 define --runs and --clients. bench.sh only execs tools/strata_bench.py.",
              "importance": "high"
            },
            {
              "claim": "What upstream expects from a perf PR: same-hardware, same-workload before/after runs with every changed setting named; all individual runs and failures kept; Passed / Failed / Not run kept apart; defaults changed only where measured faster; new features opt-in with the default path byte-identical to the last release; every touched backend built.",
              "evidence": "/media/raid0/StrataQ38FN/Strata-212/CONTRIBUTING.md lines 70-97 (PR body template) and 115-121. AGENTS.md lines 87-100 ('compare against a same-day run of the build you are testing'). The area prefix is enforced by .github/workflows/title-area.yml.",
              "importance": "high"
            },
            {
              "claim": "No CI workflow runs benchmarks or uses GPUs.",
              "evidence": ".github/workflows/: docker-images, intake, metal-build (macos build), reopen, scripts-test (node), terminal-minimal-server, terminal-portable-monitor, title-area. All use GitHub-hosted runners (ubuntu/macos); none is self-hosted and none has a bench step.",
              "importance": "medium"
            },
            {
              "claim": "The engine A/B tools upstream uses itself: tools/ab_engine.py (restarts the server per arm, alternates arm order, greedy fixed prompts, medians over rounds), tools/batch_test.py (bit-exactness of each slot vs solo, --n 16), tools/piped_read_join_bench.py (cold join beside streams), and `strata bench run --suite serving-smoke/parallel-mixed-v1/...` (explicitly not standard submissions).",
              "evidence": "tools/ab_engine.py docstring lines 1-8. docs/BATCHING.md:108 cites `tools/batch_test.py --n 16`; BATCHING.md:262 cites tools/piped_read_join_bench.py. docs/TERMINAL.md:134-154 covers the serving suites.",
              "importance": "medium"
            },
            {
              "claim": "Upstream has no measured 4x RTX 3090 multi-user numbers of its own and asks for multi-GPU NVIDIA reports. Its headline speeds are single RTX 5070 and 4x R9700 HIP.",
              "evidence": "docs/TEST_REQUESTS.md row 'Three or four NVIDIA GPUs | No multi-GPU NVIDIA rig of that size here | #880'. docs/DETAILS.md:43-48 is RTX 5070, engine 0.1.26. docs/BATCHING.md:99-140 is the 16-slot table on 4x R9700, credited 'From havok2-htwo's patch series, PR #2364'. The only 4x RTX 3090 line in docs is DETAILS.md:1534 (our STRATA_MMQ_DIRECT row). The SM86_PERF table was measured on RTX 3060, A4000 and P100 (docs/release-notes-0.2.0.md:55).",
              "importance": "high"
            },
            {
              "claim": "There is no GitHub Discussions, wiki or pinned benchmark thread. The community board is the stream of [Benchmark] issues and PRs, and maintainers port reports in their own commits rather than merging the PRs.",
              "evidence": "`gh api repos/Niko1221/Strata`: has_discussions false, has_wiki false. The pinnedIssues GraphQL query returned none. Benchmark-titled issues and PRs include #5428 (4x RTX 3090 UD-Q4_K_XL, filmo, open), #5463 (2x RTX 3090), #5355, #5396 and #5482 (standard-v1 draft.1 runs). A search for merged '[Benchmark]' PRs returned none; #2718 is closed unmerged. Our #2364 was closed by the 0.2.0-beta tracker reset (Niko1221 comment, 'Closed here does not mean rejected', /reopen procedure) yet ported as df44bcb7 / 79e5cd18. Our earlier report commit 42de2938 is not an ancestor of v0.2.12 146df3c4.",
              "importance": "high"
            },
            {
              "claim": "/media/raid0/StrataQ38FN/linux/community_bench_144.sh targets neither official format. It is our own 0.1.44-era harness and cannot be reused as-is for a v0.2.12 submission.",
              "evidence": "community_bench_144.sh lines 26-43: it stops strata.service, uses PORT 8095, and calls our run_strata_variant.sh (1,8,16 users, 100-token prompts, --temperature 0, 2 rounds) and bench_multiturn.py, writing results/sweep.jsonl and multiturn.jsonl. Server dirs are hardcoded to Strata-143-prod and Strata-144, configs to CB144-*.json. It never calls tools/strata_bench.py or `strata bench run`. It was used for TESTLOG #76 and #2364 (42de2938 added bench/2026-10-14-4x-rtx-3090/ with scripts/).",
              "importance": "high"
            },
            {
              "claim": "The stock-v0.2.12 baselines quoted for the showcase (B, C) are invalid by our own TESTLOG. A like-for-like stock run at presence_penalty 0 is still missing, and CONTRIBUTING's 'every changed setting identified' rule needs one.",
              "evidence": "linux/TESTLOG.md #99: 'All 212-server arms of #98 ... were measured penalised; their decode/agg numbers are invalid'. results/w212/summary.txt lines 5-6 and 20 are the B, C and C-r2 rows from #98. w212c compares PT only against A (our old production) and P.",
              "importance": "high"
            },
            {
              "claim": "Both official harnesses send sampling explicitly, so the #2685 model-card default (presence_penalty 1.5 when thinking is off) does not hit them. The standard protocol is sampled (T 1.0), not greedy.",
              "evidence": "a8091f4d says request fields win (serve/test_model_card_sampling.py:4). tools/strata_bench.py:141 sends temperature 0. BENCHMARK_PROTOCOL.md:206 says every request carries all sampling values, with presence_penalty 0.0. TESTLOG #59/#77 show our sampled-batch paths equal solo at T 0.7 with a seed.",
              "importance": "medium"
            },
            {
              "claim": "The standard protocol (concurrency 1, optional 4 clients) would barely show our main multi-user gains (16 slots, cold-join stall, admission). A 16-user comparison has to go in as a community or custom-track report or as engine-PR evidence, with a same-day stock arm.",
              "evidence": "BENCHMARK_PROTOCOL.md lines 106-116 (concurrency 1; optional 4; 'Testing another capacity is another run') and lines 93-97 (track 'custom' for modified builds; 'A claimed improvement requires a separate, same-machine baseline').",
              "importance": "medium"
            }
          ],
          "open_questions": [
            "Should the rerun of stock v0.2.12 (+SM86 table) vs PT use our sweep harness, tools/strata_bench.py --clients 16, or `strata bench run` (standard-v1, optional concurrency)? Only standard-v1 gives the board's comparable format, and it does not exercise 16 slots.",
            "Does PT stay bit-identical and keep its fast paths under the standard protocol's thinking-on sampling (T 1.0, top_k 20, seeded)? Not measured; the solo-pipeline and lanes gate is documented only for presence_penalty (TESTLOG #99).",
            "Is #2364's ported content (df44bcb7) the precedent the maintainer wants for multi-user reports, or should the new numbers be posted on an existing thread such as #5428 or #880? Posting needs the user's text approval (upstream-contrib-identity memory).",
            "Not checked: upstream main commits after 146df3c4 (pushed 2026-11-07) for new bench docs, or the comments on #5428 beyond the two bot notes."
          ]
        },
        "verify": {
          "checked": [
            {
              "claim": "Standard benchmark is `strata bench run`, defined in docs/BENCHMARK_PROTOCOL.md, still draft (strata-qwen38-thinking-v1-draft.1), fixtures bench/prompts/standard-v1, corpus SHA e75eff8b..., from 0.2.1.",
              "verdict": "confirmed",
              "correction": "Confirmed in BENCHMARK_PROTOCOL.md lines 14-19 (it also says BENCH.bat/bench.sh/strata_bench.py do NOT implement it and that the approved manifest does not exist yet). Origin is https://github.com/Niko1221/Strata.git and Strata-212 HEAD is 146df3c468739147f520ac3816de806f71232571."
            },
            {
              "claim": "Protocol mechanics: concurrency 1, thinking on, model-card thinking sampling with presence_penalty 0, seeds 1-3, 512 cap, 1 warm-up + 9 perf + 12 sanity, optional 4 clients x 3 rounds, capacity only recorded ('another capacity is another run'), runner rejects --concurrency/--repeat/--request-timeout.",
              "verdict": "confirmed",
              "correction": "Line numbers are a little off. The sampling table is at lines 66-76, the stage table at 103-113 and 'Testing another capacity is another run' at line 117. The rejected flags are at line 43, not 50; line 50 is the --track/--custom-pack options."
            },
            {
              "claim": "Metric definitions (prompt tok/s = read_tokens/prompt_s; decode = predicted_n/predicted_ms excluding rejected drafts; client-clock TTFT; round aggregate; per-case median/min/max/n, never pooled; cached_tokens <= max_shared_prefix+8 else nonconforming).",
              "verdict": "confirmed",
              "correction": "The cache rule is at lines 207-210, not 214. When the cache is exceeded on a performance request, that request is dropped from the speed summaries and the run is marked nonconforming. The protocol also tells you to restart the engine between two runs on the same server."
            },
            {
              "claim": "Submission: results-only PR with bench/results/<UTC-date>-standard-<hardware>-<run-id>/ (README, results.json, sanitized config, engine logs, outputs.jsonl), title `benchmark: <GPU> / <model> / <quant> / <protocol-id>`; approved.json planned but absent.",
              "verdict": "confirmed",
              "correction": "The doc text is quoted correctly (lines 448-468). Two problems in practice. First, the title the doc suggests ('benchmark: ...') would fail .github/workflows/title-area.yml, which only allows '[Engine] ', '[GUI] ', '[TUI] ', '[Benchmark] ', '[Docs] ' or '[Other] '. Real submissions therefore use '[Benchmark] <GPU> / <model> / <quant> / <protocol-id>' (e.g. #5396, #5556, #5567). Second, the findings' own summary uses the [Benchmark] form, which contradicts this finding."
            },
            {
              "claim": "Quick community check (bench.sh/BENCH.bat/tools/strata_bench.py): greedy temperature 0, thinking off, 3 runs, chat/4K/16K/5-turn agent, optional --clients N and --long 32K, fixed README template.",
              "verdict": "confirmed",
              "correction": "Confirmed: tools/strata_bench.py:141 sets temperature 0, :145-146 turn thinking off, :1035-1037 define --runs/--clients/--long, and the test table is at COMMUNITY_BENCHMARKS.md:108-114. One correction: the template is only suggested. COMMUNITY_BENCHMARKS.md 'Submit a report' step 3 says 'Use your own data format and document its fields and units'."
            },
            {
              "claim": "Perf PR expectations: same-hardware/workload before/after with every changed setting, keep runs and failures, Passed/Failed/Not run apart, defaults changed only where measured faster, opt-in with default byte-identical, build every touched backend, area prefix enforced by bot.",
              "verdict": "confirmed",
              "correction": "CONTRIBUTING.md lines 70-97 and AGENTS.md lines 87-99 (same-day run, opt-in, byte-identical default). title-area.yml enforces the prefix on PRs and issues. CONTRIBUTING also requires that agent-assisted claims are verified against the source, and that no unproduced measurements are submitted."
            },
            {
              "claim": "No CI workflow runs benchmarks or uses GPUs.",
              "verdict": "confirmed",
              "correction": "The 8 workflows run on ubuntu-latest, ubuntu-24.04, macos-26 or a matrix OS. None is self-hosted and none has a bench step."
            },
            {
              "claim": "Upstream A/B tools: ab_engine.py (restart per arm, alternate, greedy), batch_test.py --n 16, piped_read_join_bench.py, serving suites (not standard submissions).",
              "verdict": "confirmed",
              "correction": "Confirmed: ab_engine.py docstring lines 1-9, BATCHING.md:108 and :263, and TERMINAL.md:152 ('their output is not a standard submission'). Missing caveat: ab_engine.py:45 and piped_read_join_bench.py:68-70 send temperature 0 with thinking off and no presence_penalty. On a v0.2.12 server they therefore get presence_penalty 1.5 (see the #2685 item)."
            },
            {
              "claim": "Upstream has no own 4x RTX 3090 multi-user numbers. Headline speeds are a single RTX 5070 (DETAILS.md:43-48) and 4x R9700 HIP (BATCHING.md 16-slot table from PR #2364). TEST_REQUESTS lists 'Three or four NVIDIA GPUs' (#880). The only 4x 3090 line is DETAILS.md:1534. The SM86 table was measured on 3060/A4000/P100.",
              "verdict": "confirmed",
              "correction": "Mostly right. Fix: the sm_86 table was measured on the RTX 3060 and RTX A4000 only. The P100 figure in release-notes-0.2.0.md:55 belongs to the separate sm_60 table. DETAILS.md:1534 was added in Niko1221's doc commit ed4b4db4; it describes our STRATA_MMQ_DIRECT measurement. bench/results also has 2x RTX 3090 and single-3090 community reports, but no 4x 3090 multi-user report."
            },
            {
              "claim": "No Discussions/wiki/pinned thread. The board is the stream of [Benchmark] issues/PRs. No merged [Benchmark] PRs; #2718 closed unmerged. Reports land when the maintainer ports them in his own commits (#2364 -> df44bcb7/79e5cd18). 42de2938 is not an ancestor of 146df3c4.",
              "verdict": "wrong",
              "correction": "Several parts are confirmed: has_discussions false and has_wiki false (gh api). #2718 was closed unmerged on 2026-10-19 by the 0.2.0-beta tracker reset. 42de2938 is not an ancestor of 146df3c4. GitHub shows mergedAt null even for PRs whose content landed, e.g. #1655 (Cass) and #1192 (apollo-mg). The 'his own commits' part is wrong. Content is committed with the contributor kept as git author: 2d49e617 by Cass '(PR #1655)', 6024082a by apollo-mg '(PR #1192)'. Our #2364 engine commits also keep havok2-htwo as author (ee2a907e, dd544364, 0cf1a9b7, ca239cda, 62fc7c8a, a71ee46c, 69a05bfb). Only the merge (df44bcb7) and the docs port (79e5cd18) are Niko1221's. Missed fact: in v0.2.12 the newest bench/results folders are 2026-10-20 (two 0.2.1 maintainer folders). No community report has landed since the tracker reset, and every [Benchmark] PR found is still OPEN (#5428, #5463, #5396, #5352 and others)."
            },
            {
              "claim": "community_bench_144.sh targets neither official format; it is a 0.1.44-era wrapper (stops strata.service, port 8095, run_strata_variant.sh, bench_multiturn.py, hardcoded Strata-143-prod/Strata-144, CB144 configs) and cannot be reused as-is.",
              "verdict": "wrong",
              "correction": "The technical details are correct (lines 10-43), and it cannot be reused as-is for v0.2.12. But 'targets neither official format' is wrong for the community track. Its header (line 2) says it is 'for an upstream community report (docs/COMMUNITY_BENCHMARKS.md)', and that guide explicitly allows your own harness and data format if fields and units are documented. So it targets the community-report track, just not strata_bench.py or the standard protocol. Note also that it is not 'read-only' safe: it stops production."
            },
            {
              "claim": "The stock v0.2.12 baselines B and C are invalid per TESTLOG #99, and a like-for-like stock run at presence_penalty 0 is still missing.",
              "verdict": "confirmed",
              "correction": "TESTLOG.md row 99 (line 131): 'All 212-server arms of #98 ... were measured penalised; their decode/agg numbers are invalid'. results/w212/summary.txt rows B (line 5), C (line 6) and C-r2 (line 20) come from #98. w212c compares only A/P/PT/PTD/PT4/PTX. The task's figures (C 529/397, dec 131/86/74, join gap 5.8 s) are therefore penalised stock numbers. They also differ from the w212c PT numbers, which are a 2-run mean (529.1/406.6) and not the 526.1/404.3 run-1 row."
            },
            {
              "claim": "Both official harnesses send sampling explicitly, so the #2685 default (presence_penalty 1.5 when thinking off) does not hit them; standard protocol is sampled T 1.0.",
              "verdict": "wrong",
              "correction": "This holds only for `strata bench run` (thinking on, so the card's thinking set has no presence_penalty, and BENCHMARK_PROTOCOL.md:200 has the runner send every value). It does not hold for the quick community check. serve/server.py:6220-6223 merges defaults per field: {**model_card_sampling(arch, thinking), **config sampling, **shared}, then the request's non-None fields win. tools/strata_bench.py:141 sends only temperature 0 and thinking off, with no presence_penalty, top_p or top_k. On a v0.2.12 server it therefore gets top_k 20, top_p 0.8 and presence_penalty 1.5 with greedy temperature. serve/test_model_card_sampling.py:82-85 asserts that temperature=0 still yields top_k=20. This is the same mechanism that invalidated our greedy #98 arms. The same applies to tools/ab_engine.py:45 and tools/piped_read_join_bench.py:68-70. A stock v0.2.12 community-check run is penalised unless the config carries "sampling": {"presence_penalty": 0}."
            },
            {
              "claim": "The standard protocol (concurrency 1, optional 4) would barely show our multi-user gains; a 16-user comparison has to go as a community/custom report or engine-PR evidence with a same-day stock arm.",
              "verdict": "confirmed",
              "correction": "BENCHMARK_PROTOCOL.md:103-117 and :88-90 (track 'custom', 'A claimed improvement requires a separate, same-machine baseline'). AGENTS.md:96-98 also asks for a same-day run."
            }
          ],
          "missed": [
            "#5428 (filmo, 4x RTX 3090, open) says it 'extends the merged 2026-10-24 report from the same machine (PR #2718)'. GitHub shows #2718 CLOSED with mergedAt null, and v0.2.12 has no bench/results folder dated after 2026-10-20 and no 4x-3090 report. filmo's 'merged' claim is wrong. #2718 measured exactly our setup: 4x RTX 3090, UD-IQ4_XS, 262K, with tools/strata_bench.py --clients 1 --long, 113.3 tok/s decode, single user. It is the closest public comparison point.",
            "Since the 0.2.0-beta tracker reset, no community [Benchmark] PR has landed in v0.2.12. All the ones listed by `gh pr list --search 'Benchmark in:title'` are OPEN (#5428, #5463, #5461, #5396, #5352, #5556 and more). A new report would probably sit in that queue.",
            "The v0.2.12 upstream server penalises upstream's own A/B and quick-bench tools. ab_engine.py, piped_read_join_bench.py and strata_bench.py send greedy, thinking off and no presence_penalty, so per server.py:6220-6223 they get presence_penalty 1.5. A fair stock arm with any of them needs the config sampling block (presence_penalty 0). If we present 'stock as shipped' instead, we must say so, because #2685 by design turns off the solo pipeline, lanes and merge-first."
          ]
        }
      },
      {
        "key": "github-discussion",
        "scout": {
          "summary": "Read-only survey of what Niko1221/Strata upstream (https://github.com/Niko1221/Strata, remote from /media/raid0/StrataQ38FN/Strata-212) has said about havok2-htwo's work. Our public events (gh api users/havok2-htwo/events/public, the complete list for this repo) are: the fork on 2026-10-14, 3 PRs (#2364 bench report, #3022 HC_Q8 fix, #3559 HC_Q8 fix resubmitted with the [Engine] prefix), and 8 comments on #1242, #2364, #2275, #2326 (2x), #2597 and #2235. Search-API results miss some of these: `--commenter` did not return #1242. We have no issues of our own, and nothing of ours is open now.

What upstream adopted, with credit:
(1) The v0.1.45 series: merge df44bcb7 "Merge f-145-havok (#2364 multi-user series, opt-in)". 29 commits in v0.2.12 keep havok2-htwo as author (git log --author=havok2 146df3c4). The release notes have a section "Multi-user servers: opt-in switches (#2364, thanks havok2-htwo)". docs/BATCHING.md:101 says "(From havok2-htwo's patch series, PR #2364.)". The maintainer re-measured on 4x R9700 rather than citing our 3090 numbers.
(2) The HC_Q8 fix as abf2953b, authored by Niko1221 with "Co-authored-by: havok2-htwo". The v0.2.12 notes say "Hyper-connection rows copied per stage: 414 MiB less per card on two A6000 (#3559)" and "havok2-htwo: #3559" under Thanks.
(3) db568409 "folds in havok2-htwo's batch_test16.py and batch_test32.py".

What was flagged or not taken: STRATA_SLOT_CKPT_MOVE is marked "do not use" in the v0.1.45 notes. Our counting-artifact analysis and test fix on #2364 got no reply, and the test is unchanged at tools/batch_groups_cache_test.py:83-84. Our STRATA_PIPE_OVERLAP and spec-lanes work was never submitted as PRs; upstream took #2326 PIPED_CONCURRENT and #2050 instead.

The maintainer has never replied directly to any of our comments. Their only posts on our items are the tracker-reset template (2026-10-19) and the "ported" close on #3559. The one community reply addressed to us is Vop3p on #2326, asking for a 4x3090 CUDA test of #3103 (STRATA_PIPED_CONCURRENT_SMB). It is still unanswered, and #3103 was closed by the reset.

Contribution rules (CONTRIBUTING.md 122 lines, AGENTS.md, .github/pull_request_template.md, workflows intake.yml/reopen.yml/title-area.yml):
- Area prefix in the title.
- Template sections: Problem and result / Related discussion / Approach / Validation (Passed / Failed / Not run) / Limits.
- Agree scope in an issue first for features and default changes.
- One problem per PR. New features opt-in, default path byte-identical.
- Build every backend touched.
- Perf claims need same-hardware before/after runs, individual runs preserved.
- AI assistance is welcome; the contributor is responsible.
- No CLA or DCO found.
- In practice the maintainer requires a +3% bar with interleaved rounds.

The presence_penalty finding (#2685 → a8091f4d) is already reported upstream: #2940 (closed, /reopen by a non-author did not reopen it), plus open #3295 and #3354.",
          "findings": [
            {
              "claim": "Complete inventory of our upstream activity: 3 PRs (all closed, none merged on GitHub), 0 issues, 8 comments on 6 threads, nothing open.",
              "evidence": "gh api users/havok2-htwo/events/public: ForkEvent 2026-10-14T03:08; PR #2364 opened 2026-10-14T03:18; comment on #1242 2026-10-14T03:18 (https://github.com/Niko1221/Strata/pull/1242#issuecomment-6160104471); PR #3022 opened 2026-10-18T06:43; comments 2026-10-18 06:43-06:44 on #2364 (issuecomment-6246688929), #2275 (issuecomment-6246689644), #2326 (issuecomment-6246690323), #2597 (issuecomment-6246691977), #2235 (issuecomment-6246694385); #2326 again 2026-10-18T19:38 (issuecomment-6254259757); PR #3559 opened 2026-10-21T07:22. search/issues involves:havok2-htwo returns 7 items, all closed. Note: gh search --commenter missed #1242, so the search index is incomplete.",
              "importance": "high"
            },
            {
              "claim": "v0.1.45 ported most of our 0.1.43 multi-user series with authorship and release-note credit. The maintainer re-measured on their own 4x R9700 (HIP) and did not use our 3090 numbers.",
              "evidence": "Merge df44bcb7 'Merge f-145-havok (#2364 multi-user series, opt-in)'. 29 commits authored havok2-htwo reachable from v0.2.12 146df3c4 (e.g. ee2a907e '>8 slots with a layer split', dd544364 D2D slot copy, 86f677b4 pointer-table MMQ, 0cf1a9b7 ADMIT_MERGE_FIRST, a71ee46c QSA_SLOTS, 1d9758f8 HANDOFF_PIPE, ed4fa42a ADMIT_SHORT_FIRST). Release v0.1.45 (https://github.com/Niko1221/Strata/releases/tag/v0.1.45): heading 'Multi-user servers: opt-in switches (#2364, thanks havok2-htwo)'; 'Measured on 4x R9700 ... 16 users 207 to 324 tok/s (1.56x ...)'. docs/BATCHING.md:101 '(From havok2-htwo's patch series, PR #2364.)', with the R9700 table at lines 129-139. Also 79e5cd18 'docs/BATCHING.md: ... the 4x R9700 numbers (PR #2364 port)' and db568409 'tools: batch_test.py takes up to 32 prompts (folds in havok2-htwo's batch_test16.py and batch_test32.py)'.",
              "importance": "high"
            },
            {
              "claim": "STRATA_SLOT_CKPT_MOVE is flagged 'do not use'. Our analysis showing the test failure is a counting artifact, and our suggested test fix, got no maintainer reply and were not adopted.",
              "evidence": "v0.1.45 notes: '**`STRATA_SLOT_CKPT_MOVE=1` is experimental: do not use it.** With it, `tools/batch_groups_cache_test.py` fails'. Our comment https://github.com/Niko1221/Strata/pull/2364#issuecomment-6246688929 proposes `reqs = [(p, r) for p, r in reqs if r != p - 1]` and asks 'Should I close it?'. The next and only maintainer post on #2364 is the reset template (issuecomment-6272998304). The test is unchanged in v0.2.12 (tools/batch_groups_cache_test.py:83 `ok = len(reqs) >= 6`, :84 `for p, r in reqs[2:]`; last change 399a75c8 on 2026-10-09).",
              "importance": "medium"
            },
            {
              "claim": "The HC_Q8 trimmed-split fix was adopted in v0.2.12 as abf2953b, with havok2-htwo as co-author and a release-note credit. The maintainer re-measured it on 2x A6000.",
              "evidence": "abf2953be0cdb3c72c91f711d952c1411b12de22, author Niko1221, trailer 'Co-authored-by: havok2-htwo <65780686+havok2-htwo@users.noreply.github.com>'. It also adds STRATA_HC_Q8_ALL_STAGES=1 and a manifest record, and is labelled 'Class E'. Maintainer on #3559 (https://github.com/Niko1221/Strata/pull/3559#issuecomment-6704759191): 'Thanks, havok2-htwo. This is in Strata v0.2.12 ... It was ported as abf2953be, with you as a co-author. The change reached main as its own commit, so GitHub does not mark this PR as merged; closing it by hand.' v0.2.12 notes (https://github.com/Niko1221/Strata/releases/tag/v0.2.12): 'Hyper-connection rows copied per stage: 414 MiB less per card on two A6000 (#3559).' and under Thanks '**havok2-htwo:** #3559'. #3022, the first version, was closed by the reset (https://github.com/Niko1221/Strata/pull/3022#issuecomment-6274293247); it lacked the [Engine] prefix and the template, and #3559 is the resubmission.",
              "importance": "high"
            },
            {
              "claim": "The 0.2.0-beta tracker reset (2026-10-19) closed every open item. Its stated meaning is 'not rejected', and items come back with /reopen once the template is filled in. A second sweep (2026-11-05, 0.2.11) closes issues older than a week with no new reports.",
              "evidence": "Verbatim on PR #2364 (https://github.com/Niko1221/Strata/pull/2364#issuecomment-6272998304) and #3022: 'We're resetting the tracker for 0.2.0-beta. There are more than 1,300 open issues and pull requests, too many for us to review properly, so we're closing everything that isn't being worked on right now. Closed here does not mean rejected.' Steps: '1. Rebase on `main` ... re-run your checks on 0.2.0-beta. 2. Start the title with its area: [Engine], [GUI], [TUI], [Benchmark], [Docs] or [Other]. 3. Fill in the pull request template ... 4. Comment `/reopen`. A check runs and either reopens the pull request or lists what is still missing.' Issue version on #2235/#2275 (issuecomment-6270884483, -6270920646). Second sweep on #2275 (https://github.com/Niko1221/Strata/issues/2275#issuecomment-6659275272): 'Strata has changed a lot since this was opened, and 0.2.11 is out now ... We're closing issues older than a week that have had no new reports in the last week. Closed here does not mean rejected.' Mechanics: .github/workflows/reopen.yml ('Only the author of the item or a maintainer can trigger it'), .github/scripts/item-check.js checkPull (title prefix, 'Problem and result' of at least 20 chars, a 'Validation' section, a tested revision), and intake.yml (needs-format label; closed after 5 days). Limitation: #3119's /reopen 'passed, but GitHub would not reopen it' (https://github.com/Niko1221/Strata/pull/3119), and a non-author's /reopen on #2940 (Vop3p) left it closed.",
              "importance": "high"
            },
            {
              "claim": "The maintainer's acceptance process for speed work: a 'speed pass' with a +3% end-to-end bar, interleaved pairs, and an identity gate with the switch unset. Fixes without a speed claim go to a separate 'fix review'.",
              "evidence": "Niko1221 on 2026-10-17: #1187 'The measured +1.4% is below our +3% bar for taking an extra kernel path' (https://github.com/Niko1221/Strata/pull/1187#issuecomment-6231016634); #1535 'If you get a >=3% gain over 5+ interleaved pairs on a current build we would gladly look again' (issuecomment-6231053038); #1511 'below the 3% we need for a change of this size' (issuecomment-6231048029); #1100 'your numbers show no end-to-end gain, so we are not porting it' (issuecomment-6231004909). #1242, where we commented: 'This is a correctness fix without a speed claim, so it is outside the speed pass; it stays on the list for the fix review.' (https://github.com/Niko1221/Strata/pull/1242#issuecomment-6231019026). #2326 port: '3 interleaved rounds ... With the variable unset everything is identical to main (identity gate on sm_86, sm_60 and gfx1201). It stays opt-in and experimental in 0.1.45; we would like more reports from 3-4 stage splits before we change that.' (https://github.com/Niko1221/Strata/pull/2326#issuecomment-6231132844).",
              "importance": "high"
            },
            {
              "claim": "Declined related work: batch MTP on a serial split (#1253), because it was not faster than the pipelined groups. The maintainer explicitly invited a pipelined batch-MTP variant with a measured gain, which matches our spec lanes / solo pipeline.",
              "evidence": "Niko1221 on #1253 (https://github.com/Niko1221/Strata/pull/1253#issuecomment-6231020182): 'batch MTP over a serial split is no faster than the pipelined groups, which are the default on a split, and we already carry a batch-MTP path for two stages from #1636. So we are not taking it. A pipelined batch-MTP variant with a measured gain would be welcome.' Our spec-lanes data (+26..+45 % per user at 2-4 users) is only a comment on closed #2235 (issuecomment-6246694385) and was never submitted as a PR.",
              "importance": "high"
            },
            {
              "claim": "Community replies to our comments: Vop3p built #3103 (PIPED_CONCURRENT beside SPLIT_MTP_BATCH) in response to our #2326 comment and asked us for a 4x3090 CUDA test. We never answered, and #3103 was closed by the reset.",
              "evidence": "https://github.com/Niko1221/Strata/pull/2326#issuecomment-6255353503: '@havok2-htwo I tried it: #3103 (opt-in `STRATA_PIPED_CONCURRENT_SMB=1`) ... What I cannot test, and where your rig would help: the stream-k path that the reservation is actually for. If you can, a run on your 4x 3090 with `STRATA_PIPED_CONCURRENT=1 STRATA_SPLIT_MTP_BATCH=1 STRATA_PIPED_CONCURRENT_SMB=1` would tell whether it holds on CUDA'. #3103 was closed 2026-10-19T18:32 with the reset template (https://github.com/Niko1221/Strata/pull/3103#issuecomment-6274389202). Other threads: on #2235, superniker posted after us (2026-10-19) but did not address us; on #2597 and #2275 nobody replied to our comments. #2597 was closed by its author on 2026-10-18T07:09. #2275 was reopened via /reopen by superniker and closed again 2026-11-05. On #1242, brocktice reports the image-position bug still on v0.2.2 (issuecomment-6317383073); we commented there that we have a per-slot M-RoPE fix.",
              "importance": "medium"
            },
            {
              "claim": "Our #2364 benchmark report was never merged into the repo's benchmark folders. bench/results has no 4x 3090 multi-user folder and COMMUNITY.md does not mention #2364. The repo explicitly asks for 3-4 NVIDIA GPU reports and for an hour-plus soak on a split.",
              "evidence": "ls /media/raid0/StrataQ38FN/Strata-212/bench/results | grep 3090 lists only single-3090 reports (2026-09-29, 10-03, 10-06, 10-11). grep '2364|havok' in bench/results/COMMUNITY.md and docs/COMMUNITY_BENCHMARKS.md finds nothing. docs/TEST_REQUESTS.md: 'Three or four NVIDIA GPUs | No multi-GPU NVIDIA rig of that size here | #880' and 'A long soak (an hour or more of back-to-back requests) on a split or resident setup'.",
              "importance": "high"
            },
            {
              "claim": "Contribution rules: AI assistance is allowed; title area prefix and template are required; scope must be agreed in an issue first for features and default changes; one problem per PR; opt-in features with a byte-identical default path; every touched backend built; same-hardware before/after perf evidence; results-only PRs kept separate. No CLA or DCO.",
              "evidence": "CONTRIBUTING.md:3 '**AI assistance is welcome. The contributor is responsible for understanding the change and supplying real evidence.**'; :14-16 'Get maintainer agreement on scope before substantial features, refactors, architecture changes, or changes to defaults ... Focused fixes and benchmark reports can go straight to a PR.'; :17 'Keep one problem per PR'; :19 'Do not resubmit a declined proposal under a different title or through another agent'; :50-52 title prefix; :76 'Build every backend a code change touches, including CUDA, HIP, and SYCL'; :82-86 'Performance claims need before/after runs on the same hardware and workload, with every changed setting identified. Preserve the individual runs, failures, and variability ... One fast run does not justify changing a default'; :106 'Never submit fabricated measurements'; :117-118 results-only PR = 'human report, raw measurements, sanitized configuration, and supporting logs. Keep it separate from engine changes and include poor results and failed requests.' AGENTS.md:93 'A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.'; :98-100 hardware reports 'compare against a same-day run of the build you are testing'. Template: .github/pull_request_template.md (Problem and result / Related discussion / Approach / Validation Passed-Failed-Not run / Limits). docs/BENCHMARK_PROTOCOL.md:3 'Status: draft, implemented in `strata bench run`; pilot validation pending.' docs/COMMUNITY_BENCHMARKS.md:181-182 'try at least three measured runs and publish all of them with a median and range'. grep for CLA/DCO/Signed-off-by in CONTRIBUTING.md, AGENTS.md, .github/workflows and .github/scripts: not found. CONTRIBUTING.md was added 2026-10-19 (e9e08adb, dac63544).",
              "importance": "high"
            },
            {
              "claim": "The presence_penalty trap (#2685 → a8091f4d) is already known upstream and has open fix PRs, so it would be a duplicate to file. Evidence from us would go into the existing threads.",
              "evidence": "a8091f4d (2026-10-15, Niko1221) 'serve: a request with no sampling gets the model card's (... not thinking 0.7/0.8/20 + presence 1.5 ...) (#2685)'. #2940 (Cass67, closed by the reset, https://github.com/Niko1221/Strata/issues/2940) reports '~25% slower decode'. Vop3p there: 'Any penalty turns `--pipeline-windows` off for the request'. Open: #3295 '[Engine] serve: temperature 0 stays greedy - no model-card penalties on a greedy request (#2940)' and #3354 'the model-card sampling is applied silently and disables MTP drafting'; latest comment on #3354 2026-11-07 (schmittcn). CONTRIBUTING.md:10-12 says to add evidence to an existing thread.",
              "importance": "medium"
            },
            {
              "claim": "Things we said we would do and did not, and items still pending from our side.",
              "evidence": "#2364 comment: 'I can send this as a small PR if you want it' (the test fix), never sent. #2326: Vop3p's #3103 test request was not answered. #1242 comment (issuecomment-6160104471) offered a port of our per-slot fix; #1242 was closed by the reset and no follow-up from us was found. Our 0.1.45 PIPED_CONCURRENT report says the switch is gated off with in-place GGUF experts (issuecomment-6246690323: 'Is the gate needed in tha[t case]'); no maintainer answer was found.",
              "importance": "low"
            }
          ],
          "open_questions": [
            "Did the maintainer comment anywhere else (release discussions, Discord, other PRs) on our 0.1.45 series? GitHub Discussions are disabled (has_discussions=false). Apart from the reset template there is no maintainer comment on #2364, so credit appears only in the release notes, merge df44bcb7 and docs/BATCHING.md.",
            "Is CKPT_MOVE still flagged 'do not use' in v0.2.12 docs or notes? docs/BATCHING.md:118 lists it without that warning. I did not check DETAILS.md or the later release notes for the warning.",
            "What does the user mean by 'numerous comments'? I found only the reset templates, the #3559 'ported' close, and Vop3p's direct reply. Comments on the reports by other community members (blange48, superniker, oscarlius and others on #2275/#2326) are not about our work as such.",
            "Before any new submission: for a pipelined batch-MTP / spec-lanes PR or a solo pipeline, should we first open an [Engine] issue for scope agreement (CONTRIBUTING.md:14-16 requires it for substantial features)? And should the 4x3090 comparison go in as a separate [Benchmark] results-only PR that follows COMMUNITY_BENCHMARKS.md (3+ runs per arm, raw logs, same-day stock v0.2.12 arm)?"
          ]
        },
        "verify": {
          "checked": [
            {
              "claim": "Complete inventory: fork 2026-10-14, 3 PRs (#2364, #3022, #3559, all closed, none merged on GitHub), 0 own issues, 8 comments on 6 threads (#1242, #2364, #2275, #2326 x2, #2597, #2235), nothing open.",
              "verdict": "confirmed",
              "correction": "gh api users/havok2-htwo/events/public lists exactly these 11 events. search/issues involves:havok2-htwo returns 7 items, all closed. Per-thread comment ids match (e.g. #2364 6246688929, #2326 6246690323 and 6254259757, #1242 6160104471). I did not re-test the claim that '--commenter misses #1242'."
            },
            {
              "claim": "v0.1.45 ported our series with credit: merge df44bcb7, 29 havok2-htwo commits reachable from 146df3c4, the release-note heading 'Multi-user servers: opt-in switches (#2364, thanks havok2-htwo)', docs/BATCHING.md:101 credit, db568409 and 79e5cd18. The maintainer re-measured on 4x R9700 (207 to 324 tok/s, 1.56x).",
              "verdict": "confirmed",
              "correction": "All verified: git log --author=havok2 146df3c4 | wc -l = 29; v0.1.45 notes lines 12 and 19; BATCHING.md R9700 'Measured' block around line 129."
            },
            {
              "claim": "STRATA_SLOT_CKPT_MOVE is flagged 'do not use'. Our counting-artifact analysis and test fix got no reply, and tools/batch_groups_cache_test.py:83-84 is unchanged (last change 399a75c8, 2026-10-09).",
              "verdict": "wrong",
              "correction": "Two parts hold. Nobody replied on #2364 except the reset template (6272998304), and the test is unchanged at lines 83-84 (399a75c8). But 'do not use' is stale. dd471d31 (Niko1221, 2026-10-30, 'serve: the exact parallel stack is the default with batch slots') makes the server set nine switches by default with --parallel/--batch >= 2 on CUDA/HIP Qwen engines. STRATA_SLOT_CKPT_MOVE is one of them. The v0.2.9 release notes call these 'the nine qualified settings', with 'Parallel requests on Linux with NVIDIA cards: 1.45 times the 0.2.8 speed at 8 requests'. So CKPT_MOVE went from 'do not use' (v0.1.45 only) to default-on in v0.2.9."
            },
            {
              "claim": "The HC_Q8 fix was adopted as abf2953b (author Niko1221, Co-authored-by havok2-htwo), with STRATA_HC_Q8_ALL_STAGES=1, labelled Class E. The #3559 close comment says 'ported as abf2953be, with you as a co-author'. The v0.2.12 notes say '414 MiB less per card on two A6000 (#3559)' and credit '**havok2-htwo:** #3559'. #3022 was closed by the reset.",
              "verdict": "confirmed",
              "correction": "Verified: git show abf2953b, #3559 comment 6704759191 (2026-11-07), v0.2.12 notes lines 60 and 107, #3022 comment 6274293247 (2026-10-19)."
            },
            {
              "claim": "The 0.2.0-beta tracker reset (2026-10-19) says 'Closed here does not mean rejected' and gives /reopen steps. A second sweep on 2026-11-05 (0.2.11) closes stale issues. reopen.yml allows only the author or a maintainer to trigger /reopen. intake.yml closes items after 5 days with needs-format. item-check.js needs a 20-char 'Problem and result', a Validation section and a revision. On #3119 the check passed but GitHub would not reopen it. Vop3p's non-author /reopen on #2940 left it closed.",
              "verdict": "confirmed",
              "correction": "Verified: #2364 6272998304; #2275 6659275272 (2026-11-05); reopen.yml:9; intake.yml:3-4; item-check.js:15,149-151; the #3119 bot reply 2026-10-20T00:25:43Z ('The check passed, but GitHub would not reopen it'); #2940 has Vop3p's /reopen at 2026-10-23T01:15:23Z, no bot reply, and it is still closed."
            },
            {
              "claim": "Acceptance bar for speed work: +3% end to end, interleaved pairs (#1187, #1535, #1511, #1100), an identity gate with the switch unset (#2326 port). Correctness fixes go to a separate fix review (#1242).",
              "verdict": "confirmed",
              "correction": "All quoted comments verified (6231016634, 6231053038, 6231048029, 6231004909, 6231019026, 6231132844), all Niko1221 on 2026-10-17. On #1242 and #2326 the maintainer was answering the PR authors (ilumn and SuperDistance), not us."
            },
            {
              "claim": "#1253: batch MTP over a serial split was declined, and the maintainer wrote 'A pipelined batch-MTP variant with a measured gain would be welcome.' Our spec-lanes data exists only as a comment on closed #2235.",
              "verdict": "confirmed",
              "correction": "Verified at https://github.com/Niko1221/Strata/pull/1253#issuecomment-6231020182 and #2235 6246694385. That our spec lanes / solo pipeline 'match' the invitation is the other agent's interpretation, not something the maintainer said."
            },
            {
              "claim": "Vop3p built #3103 in reply to our #2326 comment and asked for a 4x3090 CUDA test. We never answered, and #3103 was closed by the reset on 2026-10-19T18:32. On #2597 and #2275 nobody replied to us. #2597 was closed by its author on 2026-10-18T07:09. #2275 was reopened by superniker and closed again on 2026-11-05. brocktice reproduced the #1242 problem on v0.2.2.",
              "verdict": "confirmed",
              "correction": "Verified: 6255353503; #3103 closed_at 2026-10-19T18:32:01Z; #2597 closed by shahrokhzargarpour at 07:09:45; #2275 bot reopen 6281850660 and close 2026-11-05T13:31:40Z; 6317383073. Our only later comment on #2326 (19:38) came before Vop3p's reply (21:27), so the request is unanswered."
            },
            {
              "claim": "Our #2364 report was never added to bench/results. bench/results has only single-3090 reports, and COMMUNITY.md does not mention #2364. TEST_REQUESTS.md asks for 3-4 NVIDIA GPUs (#880) and an hour-plus soak (#848).",
              "verdict": "wrong",
              "correction": "Mostly right, with one error. bench/results/2026-09-29-rtx3090-epyc-milan is 2x RTX 3090 (README.md:13), not a single card. A 4-GPU NVIDIA community folder already exists (2026-10-06-community-4x-tesla-p100), though none is 4x 3090 or multi-user. No 2364/havok mention in COMMUNITY.md or COMMUNITY_BENCHMARKS.md is confirmed, as are TEST_REQUESTS.md:14 and :17."
            },
            {
              "claim": "Contribution rules from CONTRIBUTING.md (122 lines) and AGENTS.md: AI assistance welcome, scope agreed in an issue first, one problem per PR, area prefix, build every backend, same-hardware perf evidence, results-only PRs kept separate, opt-in with a byte-identical default, no CLA or DCO. CONTRIBUTING.md was added 2026-10-19 (e9e08adb, dac63544).",
              "verdict": "confirmed",
              "correction": "Quotes and line numbers check out (CONTRIBUTING.md:3, 14-17, 19, 50-52, 76, 82-86, 106, 117-118; AGENTS.md:93, 98-100), and there is no CLA/DCO/Signed-off-by anywhere. One cite is off: the 'at least three measured runs ... median and range' text is docs/COMMUNITY_BENCHMARKS.md:175-176, not 181-182. Also note CONTRIBUTING.md:19: 'Do not resubmit a declined proposal under a different title or through another agent'."
            },
            {
              "claim": "The presence_penalty trap (a8091f4d / #2685) is already reported: #2940 (closed by the reset, title '~25% slower decode'), plus open #3295 and #3354 (last comment schmittcn 2026-11-07). Our evidence should go into those threads.",
              "verdict": "confirmed",
              "correction": "Verified: a8091f4d is dated 2026-10-15; #2940 was closed by Niko1221 on 2026-10-19; #3295 and #3354 are open. Vop3p's '--pipeline-windows off for the request' comment on #2940 matches our finding (solo pipeline / lanes off)."
            },
            {
              "claim": "Things we offered and did not do: the test-fix PR on #2364, a reply to Vop3p's #3103 request, a port of our per-slot M-RoPE fix on #1242, and the PIPED_CONCURRENT gate question (no answer).",
              "verdict": "confirmed",
              "correction": "The #2364 comment says 'I can send this as a small PR if you want it' and 'Should I close it?'. The #1242 comment says 'If the maintainers would rather have the region approach, I can port it to main'. In 6246690323, 'Is the gate needed in that case' got no maintainer reply."
            },
            {
              "claim": "The maintainer has never replied directly to any of our comments. Their only posts on our items are the reset templates and the #3559 'ported' close.",
              "verdict": "confirmed",
              "correction": "On #2364, #3022, #3559, #2235 and #2275 the only Niko1221 posts are templates or the port close. Their substantive #2326 and #1242 comments answer the PR authors and predate or ignore ours."
            }
          ],
          "missed": [
            "Most important miss: v0.2.9 (dd471d31, Niko1221, 2026-10-30) made a 9-switch 'exact parallel stack' the server default whenever --parallel/--batch >= 2 on CUDA/HIP Qwen engines. Six of the nine come from our series, going by commit messages: STRATA_CKPT_POOL (bebeb791), STRATA_SLOT_CKPT_MOVE (139d702e, 2f4fe20a), STRATA_SLOT_COPY_DIRECT (dd544364, ca239cda), STRATA_SLOT_COPY_BATCHSYNC (ca239cda), STRATA_BATCH_SELF_COMMIT (69a05bfb), and STRATA_QSA_SLOTS (a71ee46c), all authored by havok2-htwo. The v0.2.9 notes headline 'Parallel requests on Linux with NVIDIA cards: 1.45 times the 0.2.8 speed at 8 requests' but do not name havok2-htwo; only v0.1.45 and v0.2.12 do.",
            "Consequence for the showcase: arm B 'stock v0.2.12' with 16 slots already runs those six of our switches by default (unless the bench config set them to 0). B vs PT therefore understates our total contribution. A public comparison should say that stock already contains our series, and frame PT's gain as the not-yet-upstream part (P9-P11: pipe overlap, solo pipeline plus spec lanes, read yield). The same caveat applies to the planned comparison against arm C.",
            "The 'do not use' warning on CKPT_MOVE appears only in the v0.1.45 notes. It is gone from v0.2.12 docs/BATCHING.md:118, and v0.2.9 lists the switch as 'qualified'. Our unanswered test-fix offer on #2364 is moot for adoption, but tools/batch_groups_cache_test.py:83-84 still uses the old count.",
            "Spec lanes plus solo pipeline are a substantial feature. CONTRIBUTING.md:14-16 calls for an [Engine] issue to agree scope first. CONTRIBUTING.md:19 forbids resubmitting a declined proposal under another title, but #1253 was declined only as serial-split batch MTP and the maintainer invited a pipelined variant, so a pipelined one is not a resubmission. A 4x3090 results PR should be a separate [Benchmark] results-only PR (CONTRIBUTING.md:117-118). It needs raw runs, at least 3 per arm with median and range (docs/COMMUNITY_BENCHMARKS.md:175-176), and a same-day stock arm (AGENTS.md:98-100).",
            "GitHub Discussions are disabled (gh api repos/Niko1221/Strata -q .has_discussions = false). A showcase therefore has to be an issue or PR, or a comment in an existing thread such as #2275 (closed 2026-11-05) or the open #3354 for the presence_penalty evidence."
          ]
        }
      },
      {
        "key": "our-delta",
        "scout": {
          "summary": "I inventoried our series on top of v0.2.12 (146df3c4). It is 56 commits on g3-212-pipe (HEAD a099d295), all by havok2-htwo noreply, touching 86 files with +20,951/-464 lines. src/program/generate.cpp alone has +4,585/-226, and serve/server.py has +920. The commits are in 5 patch dirs under /media/raid0/StrataQ38FN/linux/patches/: v0.2.12-p1p2, -convmem, -vision, -serve and -pipe. Those dirs have no per-dir README. The only one is /media/raid0/StrataQ38FN/linux/patches/v0.2.12-README.txt. Every feature is behind an env switch and off by default, except a few fixes: the NVFP4 guard, ProgressWriter, the test fixes, and the vision mixed-row refusal plus the server's image-alone path. Upstream main is still exactly v0.2.12 (`gh api commits`, newest 146df3c4).

The biggest caveat for a "we are faster" submission: we have no gain for any single feature measured on v0.2.12. On v0.2.12, P (v0.2.12 + our series with production's switches) is about equal to old production A: -0.9 % short, -0.1 % long (results/w212c/summary.txt). PT's whole lead over P (+4.7 % short, +3.3 % long, +7 % decode at 1 user) comes from upstream's own sm_86 table. TESTLOG #99 also says every v0.2.12-server arm of #98 was measured with the model-card presence_penalty 1.5. That makes the B/C/F/G throughput and decode numbers invalid, and it covers both the "PT vs C" stock comparison in the task context and the claim that PW_WITH_BATCH and PIPED_MTP do not replace our lanes. Per-feature figures exist only from older bases (0.1.43 and 0.1.44.1, TESTLOG #68-#95).

Proposed order, smallest and safest first:
1. **Fixes, can go straight to a PR** (CONTRIBUTING.md:15-17): the NVFP4 MMQ_DIRECT guard, STOP_BOUND for CUDA with the produced_n guard (attach to open #4677), ProgressWriter, two test fixes, and an evidence comment on open #3354 about the presence_penalty finding.
2. **Vision correctness:** the guard plus the image gate, then per-slot M-RoPE, citing #1242.
3. **RAM and cache series, opt-in, issue first:** CONV_MMAP + GATE_EVICT, then PARK_CAPTURE, SLOT_PARK, borrow on a split, CONV_TIER.
4. **Speed:** PIPE_OVERLAP as an extension of PIPED_CONCURRENT, then SOLO_PIPELINE + SPEC_LANES (answers open #3794), then READ_YIELD together with the server admission changes. Each needs a clean A/B on v0.2.12 with presence_penalty 0 first.
5. **Small kernel levers** (QSA_SLOTS_STREAM, GR_DOWN_PF, EARLY_SAMPLE, SPLIT_OWN_RESERVE_MIB): offer last or not at all, since each sits below the ~3 % speed-pass bar or has no figure.

Do not submit: the Genesis replay (a private client parser fixture), GET /v1/load as it is (it clashes with upstream's POST /v1/load), or vision_check_smoke.py's private paths without cleaning them.",
          "findings": [
            {
              "claim": "Series size and shape: 56 commits on top of v0.2.12 in 5 patch groups. Only one README exists, not one per dir.",
              "evidence": "`git -C Strata-212p log --oneline 146df3c4..HEAD` = 56 lines. `git diff --shortstat 146df3c4..HEAD` = 86 files, +20951/-464; generate.cpp +4585/-226; server.py +920. Patch dirs: /media/raid0/StrataQ38FN/linux/patches/v0.2.12-{p1p2 (8 patches), convmem (17), vision (3), serve (8), pipe (17)}. The only README is /media/raid0/StrataQ38FN/linux/patches/v0.2.12-README.txt; there are no README files inside the dirs. Every commit's author is havok2-htwo <havok2-htwo@users.noreply.github.com>. Upstream main HEAD = 146df3c4 = v0.2.12 (gh api commits?sha=main).",
              "importance": "medium"
            },
            {
              "claim": "No gain for any single feature of ours has been measured on v0.2.12. On v0.2.12 our series as a whole is about equal to old production. PT's +3.8 % over production comes from upstream's own sm_86 table.",
              "evidence": "/media/raid0/StrataQ38FN/linux/results/w212c/summary.txt: P vs A short -0.9 %, long -0.1 %, decode at 1/2/4 users -1.9/-2.4/+0.2 %. PT vs P (the sm_86 table plus the stack): +4.7 % short, +3.3 % long, decode +7.0/+7.8/+4.5 %. PT vs A: +3.8/+3.2 %. P, PT, PTD and PT4 each match A on 62/62 short replies (TESTLOG.md L131, #99).",
              "importance": "high"
            },
            {
              "claim": "The stock comparison (PT vs C/B) and the 'upstream alternatives do not replace our lanes/overlap' conclusion are invalid as measured. All v0.2.12-server arms of #98 ran with presence_penalty 1.5. Stock B/C/F/G must be re-measured with presence_penalty 0 before any public claim.",
              "evidence": "TESTLOG.md L131 (#99): 'All 212-server arms of #98 (and w212b P/PT before 14:02) were measured penalised; their decode/agg numbers are invalid (bt identity runs without the server stay valid)'. The penalty disables the solo pipeline, the lanes, ADMIT_MERGE_FIRST and EARLY_SAMPLE. #98 (L130) holds the penalised C 528.5, F 131/87/74, G 435.2 (-15.5 %), and the join gap of about 5.8 s for every 212 arm. Upstream a8091f4d (#2685). Related open upstream issue #3354, whose last comment says Flash-Next temperature-0 requests run with presence_penalty 1.5, and #2940 (closed).",
              "importance": "high"
            },
            {
              "claim": "P1 'small fixes' group: low risk, can go straight to a PR, no measured gain.",
              "evidence": "abbaf026 (prefill.cpp +3: `if (f.tail_off) continue;` = NVFP4 guard for MMQ_DIRECT, correctness). 8e703929 (generate.cpp 11+/12-: serve_stop_bound un-gated from HIP. It stays opt-in on CUDA via STRATA_SERVE_STOP_BOUND=1 and keeps the HIP default. It adds `produced_n < max_new`, which fixes `a=-1`/`outp[-1]` on HIP's default path too). Relevant open upstream issue #4677. 17b1102b (server.py +30, test +100: ProgressWriter for #4271 part 2; it changes default behaviour, but as a bug fix). 6151bde2 (conversation_snapshot_test sync, +4; flake 4/10 → 0/30 per PLAN-0212-port.md §5). d3bc1da0 (gr_parity 1..7 tokens, +22). 25818b9f STRATA_SPLIT_OWN_RESERVE_MIB (generate.cpp +19/-4, opt-in) enables the loan-free 11,24,37 rebalance: +4.9/+3.6 % (TESTLOG #79, 0.1.44.1). That figure is hardware-specific and has not been re-measured on v0.2.12.",
              "importance": "high"
            },
            {
              "claim": "Vision batch fix (P8) is the highest-value correctness submission. Upstream v0.2.12 still has one M-RoPE table per device, and #1242 was closed only by the tracker reset.",
              "evidence": "1661c314: generate.cpp +26, refuses to mix image and text rows with ERR. That changes default behaviour with --vision --batch and needs maintainer agreement. da4b2ee3: 13 files +1091/-26, STRATA_MROPE_SLOTS=1 opt-in, mrope_regions.hpp plus a 196-line test. 3591ee27: vision_batch_check plus a fake engine, +2751. c0cdd140 + 69193088: server.py +299, image request runs alone unless the engine reports mrope_slots=1; dead encoder fails only image requests; test_image_alone.py +547. Evidence: TESTLOG #75 (15/15 image checks on 0.1.43), #98 'vision on I GO (8/8, 4/4)'. gh issue view 1242: CLOSED, brocktice reproduced it on v0.2.2. PLAN-0212-upstream-behaviour.md cites generate.cpp:5256-5271 etc. Caveat: w212c window.log '[vision PT] rc 0 GO (0 GO, 2 NO-GO, 3 INCONCLUSIVE) INFO' needs reading before citing it. tools/vision_check_smoke.py contains private paths (/media/raid0/StrataQ38FN/... TEMPLATE, tokenizer) and must be scrubbed.",
              "importance": "high"
            },
            {
              "claim": "RAM and conversation-cache group (P3-P7): opt-in features in a fixed order. Gains are measured on 0.1.43/0.1.44.1.",
              "evidence": "In dependency order: (a) bd2fb145 STRATA_CONV_MMAP (12 files +406/-30, conversation_alloc.hpp +160, test +191) plus ac6f8e66 STRATA_GATE_EVICT park half (+252); #95 peak RSS P 148.0 → 89.8-90.4 GiB; upstream #3115 closed unfixed. (b) 6eb77581 + e50204d2 PARK_CAPTURE (8 files +1016, generate.cpp +588): #68 +7.2/+6.1 % one-shot, park stall 70 → 20 ms, 0 VRAM, captures byte-identical 96/96. This overlaps by name with upstream f73b3c86 STRATA_PARK_ASYNC (#3779, VRAM stage, about 1.5 GB). Ours is renamed STRATA_PARK_CAPTURE (conversation_snapshot.hpp:45, generate.cpp:529), but the commit subject still says 'STRATA_PARK_ASYNC=1' and needs rewording. (c) 0541a7e5 SLOT_PARK (+916, generate.cpp +717) + 0873de12 + 3261acbe GATE_EVICT slot half: #70 multi-turn +2.6/+2.9 %, follow-up TTFT 22.9 → 20.6 s; it fills the gap where upstream #2526 park_slot skips with --mtp but no --batch-mtp. (d) 3daa2f24 + cc84bdfb PARK_BORROW on a split: #94 prompts ≥100K reused 73.3 → 91.7 %, 118 borrows, 0 failed. That is live production data, not an A/B; borrow did not engage in the #98/#95 bench arms. It extends upstream borrow, which is gated on `stages.empty()` (generate.cpp:8166 in v0.2.12). (e) 94069033 + a9ab3f9c + b2d5db6d + a5f20f3a + e46a087f CONV_TIER_DIR (+2099, conversation_tier.cpp 1072): #72 +25 % only under forced overflow, neutral otherwise. (f) 025bd80b + a0b8eb5e SLOT_CHAIN_REUSE/GIVEBACK_CHECKS (+655) and c91ec085 MSG_CKPT_TAIL (+342): no isolated figure, not found in TESTLOG.",
              "importance": "high"
            },
            {
              "claim": "P9 PIPE_OVERLAP(+KVS) carries the biggest latency win, a near-zero stall when a cold prompt joins. Upstream's PIPED_CONCURRENT does not cover our config.",
              "evidence": "Commits cb28df87 (PLE setters, 11 files +437), 8dbfa139 (hooks), 2042f8fa (generate.cpp +602, batch_overlap_test.py +408), 9a40582a (KVS), 4f129610 (parity test), 8d74eec9, 8a21a129. P9 total +2241/-114 lines. Opt-in STRATA_PIPE_OVERLAP / _KVS (generate.cpp:360/370). Exact: #81 16/16 token-identical, kernels SASS-identical. Gain: #80/#81 (0.1.44.1) worst stream pause 7.7 → 0.35 s at a 32K join, 19.3 → 0.29 s at 100K; join TTFT about ×1.38 (worse for the joiner). On v0.2.12 PT keeps a 0.13 s gap (w212c). Upstream overlap: bc90e1ac STRATA_PIPED_CONCURRENT, off beside SPLIT_MTP_BATCH/staged sources (batch_read.hpp:44-55); #98 arm E did not start ('refuses an in-place GGUF expert source'), E2 3.1 s (penalised). Open: batch_overlap PT FAIL 'ERR BGEN no such free slot' (#99), not yet investigated. Maintainer history: upstream took #2326 PIPED_CONCURRENT instead of our overlap in 0.1.45 (memory upstream-0145-port.md), so pitch this as an extension of PIPED_CONCURRENT.",
              "importance": "high"
            },
            {
              "claim": "P10 SOLO_PIPELINE + SPEC_LANES (LC_SHARE) has the biggest decode win at 2-4 users. It depends on P9 and is too large for one PR as it stands.",
              "evidence": "f6a9a10d (verify, arena_carve.hpp, +274), 02a71b4a (rules + host tests, +586), fed5a555 (generate.cpp +1529/-25 in ONE commit), fixups 8df1c227, 01c44bee, e33f8112. P10 total +2418/-58. Opt-in STRATA_SOLO_PIPELINE / STRATA_SPEC_LANES=2..4 / STRATA_LC_SHARE (generate.cpp:474-495). Exact: #82/#84 lane slots token-identical to serial solo spec (HOLD 10/10, 12/12, 12/12; poisoned arenas identical). Gains on 0.1.44.1: #82 1 user +12.8/+10.4 %, 2 users +45 %/+38 %; #84 3 users +38/+30 %, 4 users +26/+20 %. Cost: +104..396 MiB VRAM per card, about 0.5 GiB host RSS. On v0.2.12 lane VRAM calibration is still open (v0.2.12-README.txt), and PT's lowest free VRAM is 814 MiB (w212c). Upstream overlap: d17c52e7 PW_WITH_BATCH and 17878ac5 PIPED_MTP (#4121, needs --batch-mtp). Their #98 numbers are penalised, so not yet disproven. Open upstream feature request #3794 ('MTP drafts in the pipelined batch-slot path', 11 comments) is the natural thread to post in.",
              "importance": "high"
            },
            {
              "claim": "P11 READ_YIELD (engine) needs P9, P4 and the server admission rework (74711179). It carries the best fairness numbers but has the most coupling.",
              "evidence": "d77cb139 (prefill should_yield hook, +20), 044326f8 (read_yield.hpp + generate.cpp +403, test +190), fc538bde, a099d295. P11 total +713/-15. Opt-in STRATA_READ_YIELD=1 (generate.cpp:421). Off beside upstream PIPED_MTP/KVA/elastic K/V (fc538bde). Server side 74711179 (+1180: admit_est.py, STRATA_ADMIT_BY_NEW, STRATA_YIELD_SHORT, STRATA_YIELDS_MAX; upstream hard-codes 2) + a88014b7. Gains on 0.1.44.1: #88/#89 short TTFT max 20.20 → 1.19 s, 16 shorts' last first token 17.08 → 4.93 s, follow-up TTFT p50 11.79 → 3.09 s with ADMIT_BY_NEW; long read +3.2/+4.3 s; exact 48/48, 32/32. Overlaps upstream's #656 BYIELD/give_way and ADMIT_SHORT_FIRST (ed4fa42a) and the COALESCE entry layout (PLAN-0212-port.md risk 9).",
              "importance": "medium"
            },
            {
              "claim": "Our kernel levers alone fall below upstream's speed bar on v0.2.12, or have no figure. QSA_SLOTS_STREAM is roughly replaced by upstream's ATTN_DAILY.",
              "evidence": "0d8a7b43 QSA_SLOTS_STREAM (verify.cpp +47, parity test +425): #77 +4.3/+2.8 % on 0.1.44.1. On v0.2.12, #98 'ATTN_DAILY = our slots-stream (H2 vs H, within noise)'; #99 PTD (DAILY) vs PT -1.9 % short, -0.6 % long, but PTD decode at 1 user is 149 vs 145. 7b68023c GR_DOWN_PF=2|3 (fused_gr.cu +125): #77 'gr' arm +1.4/+1.8 % (with GR_NORM_FAST/GR_FAST). d5ad28dc EARLY_SAMPLE (7 files +145, sampler_same_test): never measured alone (PLAN-0212-port.md F14d; not found in TESTLOG), and disabled under the penalty (#99). Speed-pass bar: about 3 % end to end, 3 interleaved rounds × 8 repeats, outputs identical with the switch unset (memory upstream-0145-port.md). AGENTS.md:93: 'A new feature is opt-in, and the default path stays byte-identical to the last release.'",
              "importance": "medium"
            },
            {
              "claim": "Do not submit some server and tooling parts as they are.",
              "evidence": "419a4dc9 Genesis replay vendors serve/fixtures/frontend_90f73293.py (1435 lines, our private client parser). 6aaa20cf GET /v1/load (+493) clashes with upstream's POST /v1/load = model load (PLAN-0212-upstream-behaviour.md §4); propose /v1/load_status. acb7328a PREFIX_DIAG (+576) is diagnostics only. PLAN-0212-port.md §5 'Not to submit' also lists the stage-2 READS series (not on this branch), GR splits, ROUTE_PLAN_FUSE, IMAGE_HISTORY. Commit messages throughout say 'port <our sha>', 'our ...', 'TESTLOG #'; they need rewriting into upstream's problem-first form (CONTRIBUTING.md:54-58, title prefix [Engine]/[Other] per CONTRIBUTING.md:50).",
              "importance": "medium"
            },
            {
              "claim": "Proposed PR split, ordered by value and risk.",
              "evidence": "Wave 1, direct PRs per CONTRIBUTING.md:15-17 ('Focused fixes ... can go straight to a PR'): PR1 abbaf026 NVFP4 guard. PR2 8e703929 STOP_BOUND CUDA opt-in + produced_n guard, linked to #4677 with our CUDA data. PR3 17b1102b ProgressWriter (#4271). PR4 6151bde2 + d3bc1da0 test fixes. Plus an evidence comment on #3354 (penalty silently disables ADMIT_MERGE_FIRST and other paths). Wave 2: PR5 vision guard + server image-alone (1661c314, c0cdd140, 69193088). PR6 MROPE_SLOTS per-slot regions (da4b2ee3) + vision_batch_check (3591ee27, paths scrubbed); open an issue first since #1242 is closed. Wave 3, each opt-in, an issue first per CONTRIBUTING.md:14-16: PR7 CONV_MMAP + GATE_EVICT park half (bd2fb145, ac6f8e66; #3115). PR8 PARK_CAPTURE (6eb77581 + e50204d2, framed as the zero-VRAM sibling of upstream's PARK_ASYNC). PR9 SLOT_PARK + 0873de12 + 3261acbe. PR10 borrow on a split (3daa2f24, cc84bdfb). PR11 CONV_TIER (94069033 + 4 fixups + e46a087f); low priority, large. Hold SLOT_CHAIN_REUSE/GIVEBACK/MSG_CKPT_TAIL until there is a single-feature figure. Wave 4, after a fresh unpenalised v0.2.12 ablation: PR12 PIPE_OVERLAP(+KVS) as a PIPED_CONCURRENT extension (cb28df87 → 8a21a129), split into PLE setters / hooks / generate / KVS. PR13 SOLO_PIPELINE then SPEC_LANES (f6a9a10d → e33f8112), with fed5a555 split up, in the #3794 thread. PR14 READ_YIELD engine + PR15 server ADMIT_BY_NEW/YIELD_SHORT (74711179, a88014b7), which must land together. Wave 5, optional: QSA_SLOTS_STREAM / GR_DOWN_PF / EARLY_SAMPLE / SPLIT_OWN_RESERVE_MIB, only with a speed-pass-grade A/B.",
              "importance": "high"
            }
          ],
          "open_questions": [
            "How fast is stock v0.2.12 (B and C) really with presence_penalty 0? Until we re-measure it, the claim 'PT 529 vs C 529 vs B 486' has no valid stock baseline (TESTLOG #99 says the #98 server arms were penalised).",
            "Do upstream's PW_WITH_BATCH and PIPED_MTP (arms F and G) still fall short of our SOLO_PIPELINE+SPEC_LANES once both run without the penalty? The 'keep ours' decision in #98 rests on penalised numbers.",
            "We have no ablation on v0.2.12 of a single feature of ours, such as P minus PIPE_OVERLAP or P minus SPEC_LANES. Should the next window run them at speed-pass grade (3 interleaved rounds × 8 repeats)?",
            "Why does batch_overlap fail on PT ('ERR BGEN no such free slot', #99)? This should be understood before submitting P9.",
            "The w212c vision check says 'rc 0 GO (0 GO, 2 NO-GO, 3 INCONCLUSIVE)'. Is that a checker artifact, or a real regression on 212p?",
            "Should the vision mixed-row refusal (1661c314), which changes default behaviour, be offered as a default-on fix or kept opt-in? AGENTS.md:93 requires the default path to stay byte-identical.",
            "Who reviews: does the maintainer still port PRs into f-* branches instead of merging (memory upstream-0145-port.md)? The answer affects how finely to split the large commits fed5a555 (+1529 lines) and 2042f8fa (+602 lines in generate.cpp)."
          ]
        },
        "verify": {
          "checked": [
            {
              "claim": "Series size and shape: 56 commits by havok2-htwo noreply on 146df3c4..a099d295, 86 files +20951/-464, generate.cpp +4585/-226, server.py +920, 5 patch dirs (8/17/3/8/17), only one README (patches/v0.2.12-README.txt), upstream main = 146df3c4",
              "verdict": "wrong",
              "correction": "Almost all of it holds. I checked with git log/diff in /media/raid0/StrataQ38FN/Strata-212p, ls on linux/patches, and gh api commits?sha=main (newest is 146df3c4, 2026-11-07, tag v0.2.12). The one wrong number is serve/server.py: git diff --numstat gives +853/-67, not +920."
            },
            {
              "claim": "No single feature of ours has a gain measured on v0.2.12. P is about equal to A (-0.9/-0.1 %). PT's lead over P (+4.7/+3.3 %, decode +7.0/+7.8/+4.5 %) comes from upstream's sm_86 table. P/PT/PTD/PT4 give 62/62 identical replies.",
              "verdict": "confirmed",
              "correction": "Matches results/w212c/summary.txt (decisions section) and TESTLOG.md L131 (#99) exactly. One nuance: P still has our full series, and its join gap (0.13 s) equals A's, so the overlap does work on v0.2.12. What is missing is a throughput ablation."
            },
            {
              "claim": "The stock comparison PT vs C/B and the conclusion that 'upstream alternatives do not replace our lanes' are invalid, because the #98 212-server arms ran with presence_penalty 1.5",
              "verdict": "confirmed",
              "correction": "TESTLOG.md L131 says it word for word. a8091f4d (Niko1221) is 'serve: a request with no sampling gets the model card's ... presence 1.5 (#2685)'. #3354 is OPEN with 6 comments. Its last comment (schmittcn, 2026-11-07) says a temperature-0 Flash-Next request still runs with presence_penalty 1.5. #2940 is CLOSED and reported the same thing on 0.1.45 (~25 % slower decode), so the default has returned. The context's 'C long 397' does not match: results/w212/summary.txt has C long 395.9."
            },
            {
              "claim": "P1 fixes: abbaf026 NVFP4 tail_off guard (+3), 8e703929 STOP_BOUND un-gated from HIP with the produced_n guard (#4677), 17b1102b ProgressWriter (#4271 part 2), 6151bde2 (flake 4/10 -> 0/30), d3bc1da0, 25818b9f SPLIT_OWN_RESERVE +4.9/+3.6 % (#79)",
              "verdict": "confirmed",
              "correction": "The diffs match (git show --stat). PLAN-0212-port.md:259 has the flake figure. #4677 is OPEN with 3 comments. #4271 is CLOSED. Four details: (1) 8e703929's message says production on a v0.2.12 engine must now set STRATA_SERVE_STOP_BOUND=1 to keep the behaviour. (2) 17b1102b's subject does not mention ProgressWriter, but its body does. (3) #4271 is closed, so the ProgressWriter PR has to reference a closed issue. (4) In TESTLOG #79, +4.9/+3.6 % is the s28r config (--kv-resident 28672 plus the rebalance, 536->562 / 425->440). It is not SPLIT_OWN_RESERVE_MIB alone."
            },
            {
              "claim": "Vision: 1661c314 (+26, refuses mixed rows), da4b2ee3 (13 files +1091/-26, MROPE_SLOTS opt-in, 196-line test), 3591ee27, c0cdd140+69193088. #1242 was closed only by the tracker reset and brocktice reproduced it on v0.2.2. w212c vision 'GO (0 GO, 2 NO-GO, 3 INCONCLUSIVE)'. vision_check_smoke.py has private paths",
              "verdict": "confirmed",
              "correction": "#1242 is a PULL REQUEST, not an issue. Niko1221 closed it on 2026-10-19 with the tracker-reset message, after writing on 10-17 that it 'stays on the list for the fix review'. brocktice reproduced it on 2026-10-21. We commented on it on 2026-10-14. The vision line is at w212c/window.log:80 and :273. vision_check_smoke.py:43 and :49 hold the /media/raid0 paths. PLAN-0212-upstream-behaviour.md:13 cites generate.cpp:5256-5271."
            },
            {
              "claim": "RAM/cache group figures and the claim that 'PARK_BORROW extends upstream borrow, which is gated on stages.empty() (generate.cpp:8166 in v0.2.12)'",
              "verdict": "wrong",
              "correction": "The line citation is wrong. Strata-212/src/program/generate.cpp:8166 is the prefill expert-loan auto-noborrow condition, which has nothing to do with conversation borrow. Upstream's conversation borrow gate is at generate.cpp:16205: `bool borrow = stages.empty() && parked.tokens > ... && conversations.borrows(parked);` (from c9e571e0, #1164). The 3daa2f24 commit message confirms the 'gated on stages.empty()' fact. The figures I checked match TESTLOG: #68 +7.2/+6.1 % with the park stall 68-71 -> 20-21 ms; #70 +2.6/+2.9 % with 22.9 -> 20.6 s; #72 +25 %; #94 73.3 -> 91.7 % and 118 borrows. f73b3c86 is by linxuhao and is tagged '#3779, #3712'. #3779 is an OPEN sm_80 tuning PR, not a dedicated PARK_ASYNC issue. Commit 6eb77581's subject does still say STRATA_PARK_ASYNC=1, as the finding says."
            },
            {
              "claim": "P9 PIPE_OVERLAP: join pause 7.7 -> 0.35 s (32K) and 19.3 -> 0.29 s (100K), joiner TTFT about x1.38, PIPED_CONCURRENT off beside staged sources, arm E did not start, batch_overlap PT FAIL",
              "verdict": "confirmed",
              "correction": "TESTLOG #81 has these exact figures, and the joiner cost is 7.7->10.7 s and 19.3->26.7 s. The refusal function is concurrent_read_off at include/strata/program/batch_read.hpp:44-55, not under src/. Env reads are at generate.cpp:360 and :370. #2326 (by SuperDistance, closed) is upstream's PIPED_CONCURRENT origin. That fits the 'upstream took theirs' history, but I did not check the memory file's wording."
            },
            {
              "claim": "P10 SOLO_PIPELINE+SPEC_LANES: fed5a555 +1529/-25 in one commit, #82 +12.8/+10.4 %, 2 users +45/+38 %, #84 +38/+30 and +26/+20 %, VRAM calibration open, #3794 open with 11 comments",
              "verdict": "confirmed",
              "correction": "Checked against git show --stat, TESTLOG #82/#84, v0.2.12-README.txt:15 and gh issue view 3794 (OPEN, 11 comments). #4121 (OPEN, 7 comments) is upstream's PIPED_MTP thread, by wannagoxuf (17878ac5)."
            },
            {
              "claim": "P11 READ_YIELD: #88/#89 20.20 -> 1.19 s, 17.08 -> 4.93 s, 11.79 -> 3.09 s; overlaps upstream ADMIT_SHORT_FIRST (ed4fa42a)",
              "verdict": "confirmed",
              "correction": "The figures match TESTLOG #88/#89. #89 also gives 20.20 -> 1.17 s. But ed4fa42a ADMIT_SHORT_FIRST was authored by havok2-htwo. It is OUR earlier upstreamed contribution, not an upstream feature, so it shows upstream has accepted our serve-side admission work before."
            },
            {
              "claim": "Kernel levers: QSA_SLOTS_STREAM #77 +4.3/+2.8 %, GR_DOWN_PF +1.4/+1.8 %, EARLY_SAMPLE never measured (F14d), AGENTS.md:93 rule",
              "verdict": "confirmed",
              "correction": "Checked in TESTLOG #77, PLAN-0212-port.md:46 and AGENTS.md:93. CONTRIBUTING.md:15-17 holds: line 16 is 'Focused fixes and benchmark reports can go straight to a PR'. The titles rule is at line 48ff, not 50. 'Lead with the problem' is at line 56."
            },
            {
              "claim": "Do not submit the Genesis replay fixture (1435 lines), GET /v1/load (clashes with POST /v1/load), PREFIX_DIAG. PLAN §5 'Not to submit'",
              "verdict": "confirmed",
              "correction": "419a4dc9 vendors serve/fixtures/frontend_90f73293.py with 1435 lines. Upstream's server.py:8262 handles POST /v1/load and /v1/unload. PLAN-0212-upstream-behaviour.md:161 confirms the clash. The PLAN-0212-port.md:289 'Not to submit' list also includes PIPE_OVERLAP_MAX_QUEUE."
            },
            {
              "claim": "Proposed PR split (waves 1-5)",
              "verdict": "confirmed",
              "correction": "The split is consistent with CONTRIBUTING.md:13-17. Two amendments: #2526 (OPEN, by hotschmoe, 'Preserve decoded batch-slot conversations before slot reuse') is the natural thread for SLOT_PARK (PR9). The ProgressWriter PR references #4271, which is closed. I could not verify the finding's specific claim that 'park_slot skips with --mtp but no --batch-mtp'."
            }
          ],
          "missed": [
            "Against stock, PT is not faster on every metric, even in the (penalised) #98 numbers. results/w212/summary.txt has B/C short TTFT p95/max at 3.27/3.66 and 3.19/3.57 s; PT has 4.28/4.81 s (w212c). Multi-turn follow-up TTFT p50 is B 1.87 and C 2.07 s against PT 2.43-2.57 s. The cold 32K joiner's own TTFT is C 5.83 s against PT 8.15 s, because the overlap costs the joiner. A public 'we are faster' claim has to address these tails, or re-measure them unpenalised.",
            "Not every metric is necessarily invalid under the penalty. TESTLOG #99 lists what the penalty turns off: the solo pipeline, the lanes, ADMIT_MERGE_FIRST and EARLY_SAMPLE. PIPE_OVERLAP is not on that list. The stock join gap of about 5.8 s may therefore survive a re-measurement, but that is unverified. The task context's 'PT 529 vs C 529' compares unpenalised PT with penalised C.",
            "CONTRIBUTING.md:16 explicitly lets 'benchmark reports' go straight to a PR. A clean unpenalised benchmark report of PT vs stock v0.2.12, with the presence_penalty finding, is a valid first submission that needs no code PR. The finding does not mention this route.",
            "CONTRIBUTING.md (after line 93): 'Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.' 8e703929 changes behaviour inside the HIP path, and several commits touch shared files. We only build CUDA, so this is a blocker the plan does not mention.",
            "#2940 (CLOSED, 0.1.45: 'non-thinking requests inherit the card's presence_penalty 1.5 ... ~25% slower decode') shows that a8091f4d/#2685 brought back an earlier fixed regression. That is a strong point for the #3354 evidence comment.",
            "8e703929 means production on a v0.2.12 engine must set STRATA_SERVE_STOP_BOUND=1 to keep today's behaviour, according to its commit message. This bears on the 'deploy PT' step and on what the CUDA default in a PR should be."
          ]
        }
      },
      {
        "key": "our-evidence",
        "scout": {
          "findings": [
            {
              "claim": "The main stock-vs-ours numbers (arm B = stock v0.2.12 and arm C = B + the sm_86 table) were measured with the model-card presence_penalty 1.5. They have not been re-run since the fix. That makes the stock decode, aggregate, TTFT and join-gap figures in the headline invalid as a fair stock baseline.",
              "evidence": "linux/TESTLOG.md #99 says: 'All 212-server arms of #98 (and w212b P/PT before 14:02) were measured penalised; their decode/agg numbers are invalid'. The B/C numbers (agg16s 485.8 / 528.5, dec 119.0/81.4/71.1 and 130.6/86.6/74.2, join gap 5.84/5.83 s, TTFT p50 1.06/1.03) all come from results/w212/summary.txt, the night of 2026-11-07/08. The fix is linux/window-212.sh:351 (c["sampling"] = {"presence_penalty": 0}), added in commit 211827a on 2026-11-08 14:04. configs/T212-B.json and T212-C.json now carry it. w212c (results/w212c/summary.txt) ran only A P PT PTD PT4 PTX, with no B or C. So the task-context comparison 'PT vs C vs B' mixes penalty-free PT with penalised B/C.",
              "importance": "high"
            },
            {
              "claim": "The only valid result is ours vs our own old production (A). PT is +3.8 % short and +3.2 % long at agg16, and +5.0/+5.2/+4.7 % decode at 1/2/4 users. Replies are 62/62 identical and the join gap is equal (0.13 s). That is a gain over our own fork, not over upstream.",
              "evidence": "results/w212c/summary.txt 'decisions' section: A 509.65 vs PT 529.05 short, 394.10 vs 406.60 long, dec 138.25/123.05/87.20 vs 145.15/129.40/91.30. Each figure is a mean of 2 runs. TESTLOG #99.",
              "importance": "high"
            },
            {
              "claim": "Arm B is not stock as upstream ships it. It is the stock binary and stock server, run with our production config: about 30 STRATA_* switches, our args, and pins that turn off the upstream sm_86 table and the bit-changing defaults. Upstream reviewers will ask for a 'setup defaults' arm, and we have never measured one.",
              "evidence": "configs/T212-B.json env includes STRATA_MMQ_DIRECT, SLOT_COPY_DIRECT, ADMIT_MERGE_FIRST, QSA_SLOTS, HANDOFF_PIPE, SLOT_CKPT_MOVE=1 (memory upstream-0145-port: upstream marked it 'do not use'), and the pins SM86_PERF/GDN_TC/GDN_CONVL2/PROMPT_ATTN_IMMA/MMQ_KQ/AUTO_NOBORROW/PF_W16_CACHE_MIB=0. Its args are production's: --spec 4, --resident-experts, --conversation-cache-mib 126976, --kv int8, --max-context 262144. window-212.sh:29-31 documents this. No arm in window-212.sh runs v0.2.12 with its own defaults; PTX (line 61, 309) removes the pins only on our binary.",
              "importance": "high"
            },
            {
              "claim": "Our bench workload differs from upstream's official protocol. We have never run upstream's own bench command (`strata bench run` or tools/strata_bench.py) on either stock or our build.",
              "evidence": "Strata-212/docs/BENCHMARK_PROTOCOL.md: protocol strata-qwen38-thinking-v1-draft.1 runs with thinking on, temperature 1.0, top_p 0.95, presence_penalty 0.0, 21 requests at concurrency 1, and an optional 12 at concurrency 4. It takes 2-4 min after load, plus 1.5-2 min for the optional stage, and is still a 'draft, pilot validation pending'. COMMUNITY_BENCHMARKS.md describes ./bench.sh / tools/strata_bench.py as the quick check. Ours is bench_concurrency.py/bench_multiturn.py: greedy, thinking off (reasoning_effort none, bench_concurrency.py:78-82, bench_multiturn.py:36-38), 1-16 users. grep for strata_bench / 'strata bench run' / standard-v1 in linux/*.sh, *.py and results/*.md: not found. Thinking off is exactly what triggers the #2685 penalty; upstream's protocol (thinking on, penalty 0) does not hit it.",
              "importance": "high"
            },
            {
              "claim": "Our multi-user advantage (lanes, solo pipeline, pipe overlap) only shows at concurrency of 2 or more. Upstream's protocol measures concurrency 1, plus 4 at most. The 16-user and join-gap results need our own scripts, published with fixtures.",
              "evidence": "BENCHMARK_PROTOCOL.md:104-108 (optional concurrency: four requests, 3 rounds). w212 TESTLOG #98: our lanes give +15 % decode at 1 user and +40-50 % at 2 (penalised stock). Join gap A 0.12 s vs 5.8 s for every 212 arm (also penalised). The join test uses upstream's own tools/piped_read_join_bench.py (window-212.sh:610), which is a credibility plus.",
              "importance": "medium"
            },
            {
              "claim": "Identity evidence is strong at engine level: stock v0.2.12, our port and production give identical tokens. Over HTTP, stock B's replies differ from A's because the server code differs. So 'bit-identical to stock' holds for the engine (batch_test), not for full HTTP replies.",
              "evidence": "results/w212/summary.txt: [IDENT bt B vs A], [IDENT bt H vs B], [IDENT bt I vs H] solo 16/16 and batch 16/16 PASS, and B 'vsA:DIFF' on HTTP short replies. w212b-penalised/window.log:40 [IDENT bt P vs A] PASS. bt-PT and bt-PTX gave 16/16 solo-equal (lines 33-39). TESTLOG #98: 'B vs A over HTTP differs (other server code)'; #99: P, PT, PT-r2, PTD, PT4 62/62 identical to A. Since greedy output is identical, a quality eval is not needed for exact arms; only PTX is non-exact.",
              "importance": "medium"
            },
            {
              "claim": "The statistics are thin. Each arm has 2 runs, with a fixed arm order, not randomized or interleaved. There are no error bars or CI, it is one machine, and the power limit has changed across windows. Run-to-run spread is about 1 %, against a claimed PT-vs-A gain of +3.8 %.",
              "evidence": "results/w212c/summary.txt: A 509.1 / A-r2 510.2, PT 526.1 / PT-r2 532.0 (1.1 % spread). Across windows, A gave 514.8 in w212 (night) and 509.1 in w212c (day). The r2 repeats run after all first runs (window-212.sh:104 R2_FIRST). w212c ran at 320 W (window.log:12, 'owner's night cap is 280 W, day 320 W'), and #95 shows a 280 to 320 W change mid-window. bench_concurrency.py has no stdev/CI output (grep stdev/confidence: not found). Short runs: --rounds 2 --max-tok 512 --prompt-tok 100 (window-212.sh:570).",
              "importance": "medium"
            },
            {
              "claim": "Hardware coverage: every v0.2.12 result is from the 4x RTX 3090 host (sm_86 split, 16 slots, 262K context). There is no single-GPU or other-GPU v0.2.12 data. Older 0.1.44.1 single-card data exists for an RTX PRO 6000, and it shows our multi-user gains do not carry over to one card.",
              "evidence": "linux/RTXPRO6000.md and COMPARISON-2026-10-15.md (2026-10-15, 0.1.44.1): one card tops out at ~300 tok/s; 16 users short 274 vs vLLM 911. Our fork is fastest only at 1 user (221.7 tok/s). memory vision-rollout.md mentions an RTX 5090 Windows box that runs solo. Whether either host is still available: not found.",
              "importance": "medium"
            },
            {
              "claim": "Vision: PT is correct with images (GO 8/8, 4/4). There is no vision performance comparison, and stock was benched text-only because of the batch M-RoPE bug #1242. The summary line 'GO (0 GO, 2 NO-GO, 3 INCONCLUSIVE)' is a summary-parser artifact.",
              "evidence": "results/w212c/PT-vision.txt: 'GO text requests beside images == alone 8 of 8; GO image requests in a busy server == alone 4 of 4 ==> GO'. results/w212c/window.log:80 carries the misleading count. window-212.sh:4-5: 'Text only (the stock binaries still have the batch M-RoPE bug #1242)'. Whether v0.2.12 stock still has #1242 was not re-verified in this inventory.",
              "importance": "low"
            },
            {
              "claim": "Open test failures and known gaps a maintainer would see: batch_overlap_test FAIL on PT, bgc / bgc-I FAIL (counting artifact), stock serve unit tests time out, arm E did not engage, and there is no soak run.",
              "evidence": "results/w212b-penalised/overlap-PT.txt ends 'engine: ERR BGEN: no such free slot (--batch 16) or a bad max_new' after 14 background slots (TESTLOG #99: 'batch_overlap PT FAIL ... open'). results/w212/summary.txt: [bgc-move1/0] FAIL, [bgc-I verify] FAIL, [serve unittests] rc 124, arm E notengaged (PIPED_CONCURRENT off beside in-place GGUF). TESTLOG #99 'Next: lane VRAM calibration + soak'. Lowest VRAM free on PT is 814 MiB on card 3 (w212c summary).",
              "importance": "medium"
            },
            {
              "claim": "The prepared public material is stale. The paper (v1.3) and LinkedIn post are from 2026-10-17/18 and use 0.1.44.1-era numbers. The LaTeX source was not found locally. Our earlier upstream community report (PR #2364) is not in v0.2.12's COMMUNITY_BENCHMARKS.md.",
              "evidence": "/media/raid0/StrataQ38FN/paper-draft/ holds Strata-agentic-serving-v1..v1.3 PDFs (last 2026-10-18), Strata-onepager-v3-2026-10-18.pdf and linkedin-post-2026-10-17.md. The post cites 'paper v1.1 (src/main.tex)' and headlines 118.9 vs 271.2 s vs vLLM 0.31.0, stock 0.1.44.1 222.2 s. find for *.tex under /media/raid0/StrataQ38FN and /home/administrator (depth 5): not found; git ls-files: none. Strata-212/docs/COMMUNITY_BENCHMARKS.md: grep 3090/havok/2364 gives no matching report. PR #2364 is the 2026-10-14 0.1.44 report (+76 %/+71 %/+65 %, TESTLOG #76, community_bench_144.sh), and memory upstream-0212-port says the 0.2.0-beta tracker reset closed all our PRs. The public-launch-gate memory says nothing goes public without the user's go, and the user posts LinkedIn himself.",
              "importance": "medium"
            },
            {
              "claim": "Reusable tooling already exists for the missing runs. window-212.sh already supports a B/C/PT/A re-run with the penalty fix. compare_replies.py, bench_multiturn.py, bench_agentic.py, prefix_fanout.py, join_pause.py, eval/quick_eval.py and community_bench_144.sh (template for a stock-vs-ours community report) are all present.",
              "evidence": "window-212.sh:70-72 (ARMS / RUN / NOW env; --dry-run, --gen-configs, --summary), with configs/T212-B.json / T212-C.json now carrying presence_penalty 0. linux/eval/quick_eval.py: MMLU-Pro 240 + GSM8K 150. eval/eval_results.jsonl: strata-ud-iq4xs 77.9 / 94.0, 14.3 min. community_bench_144.sh: stock 0.1.44 vs ours, ~20 min, port 8095. Under the current test-port rule it must be moved to 8091.",
              "importance": "low"
            },
            {
              "claim": "Production downtime for the missing benchmarks. Each GPU window stops strata.service. The cost is about 5 min prep/stop plus about 2 min restore, and about 6.7 min per arm run (boot 92-98 s, warm-up, short 1-16 x2, long 16 x2, multiturn 16x3, join).",
              "evidence": "Measured: w212c ran 14:05:10 to 15:21:54 (77 min for 11 arm runs, about 6.7 min each, window.log:15, 226-229, 322). w212 ran 23:56:58 to 02:25:59 (149 min for 18 runs plus about 25 min of identity/ctest before arm A). bt identity takes about 2.3 min per binary (w212b-penalised log lines 31-39). PT vision check took 26 s. Estimates: (1) penalty-free B, C, PT, A x2 each = 8 runs, about 60 min. Minimum B, C x2 + PT x1 is about 40 min. Add a 'setup-defaults' stock arm x2 for about +14 min. (2) Upstream `strata bench run` (+ --optional-concurrency) on stock and on PT, 2 reps each: about 4-6 min suite + 1.5 min boot + first-run model hashing of 1-5 min (per the protocol doc), so about 45 min. tools/strata_bench.py prompt sweep 4K/32K/128K: duration not found, estimate 10-15 min per arm. (3) Error bars, 5 interleaved reps of A/B/C/PT: about 20 runs, about 2.3 h (night window). (4) Single GPU (CUDA_VISIBLE_DEVICES=0, stock vs ours, x2): about 45 min. Production uses all 4 cards, so it goes down anyway. (5) quick_eval on PT (+ stock): about 17-35 min, optional because greedy output is bit-identical. (6) Vision correctness on stock B (#1242 still present?): about 5-10 min. (7) Soak 1-2 h, already planned before deploy. A combined night window of (1)+(2)+(6) is about 2 h; adding (3) makes it about 4 h.",
              "importance": "high"
            }
          ],
          "summary": "Our current evidence cannot yet back an upstream-credible 'we are faster than stock' claim. The only valid measurement is PT against our own old production (A), from w212c (results/w212c/summary.txt, TESTLOG #99). PT gives +3.8 % (agg16 short) and +3.2 % (agg16 long), +5 % decode at 1/2/4 users, an equal join gap, and 62/62 identical replies. Every stock v0.2.12 arm (B, C, and E-I) ran in w212 (TESTLOG #98) before the presence_penalty fix (window-212.sh:351, commit 211827a), so those numbers are penalised and invalid. That covers the task-context figures C 529 / B 486, the 5.8 s join gap and the stock decode/TTFT.

Arm B is also not 'stock as shipped'. It uses our tuned production config plus parity pins that switch off upstream's sm_86 table and bit-changing defaults (configs/T212-B.json). Engine-level identity is solid: bt B=A, H=B, I=H and P=A, all 16/16. Over HTTP, stock B's replies differ from A's because the server code differs. Since greedy output is bit-identical, quality evals are only needed for the non-exact PTX.

The gaps are:
- a penalty-free B/C rerun, ideally with a stock setup-defaults arm
- runs on upstream's own tooling, `strata bench run` from BENCHMARK_PROTOCOL.md (thinking on, temperature 1.0, concurrency 1 and 4) or tools/strata_bench.py; neither has ever been run on our builds
- error bars: there are only 2 runs per arm in a fixed order, against about 1 % spread
- a single GPU or other hardware: there is no v0.2.12 data off the 4x3090; the RTX PRO 6000 data from 2026-10-15 shows our multi-user gains disappear on one card
- a vision performance comparison
- a soak run, and an explanation of the batch_overlap FAIL on PT

The public material is stale. The paper v1.3 PDF, the one-pager v3 and the LinkedIn post are all from 2026-10-17/18 with 0.1.44.1-era numbers. The LaTeX source was not found. Our earlier PR #2364 report is not in v0.2.12's COMMUNITY_BENCHMARKS.md.

Downtime, measured at about 7 min per arm run plus about 7 min overhead:
- penalty-free B/C/PT/A rerun: about 40-60 min
- upstream protocol, stock vs ours x2: about 45 min
- 5-rep error bars: about 2.3 h
- single GPU: about 45 min
- quick_eval: about 17-35 min
- soak: 1-2 h

A combined window (rerun + upstream protocol + stock vision check) is about 2 h, or about 4 h with error bars.

Key files: /media/raid0/StrataQ38FN/linux/TESTLOG.md, /media/raid0/StrataQ38FN/linux/results/w212c/summary.txt, /media/raid0/StrataQ38FN/linux/results/w212/summary.txt, /media/raid0/StrataQ38FN/linux/window-212.sh, /media/raid0/StrataQ38FN/linux/configs/T212-B.json, /media/raid0/StrataQ38FN/Strata-212/docs/BENCHMARK_PROTOCOL.md, /media/raid0/StrataQ38FN/paper-draft/.",
          "open_questions": [
            "Should the stock baseline be 'stock binary + our tuned config with parity pins' (B/C as defined), or 'stock v0.2.12 with the config its setup generates for 4x3090'? Upstream will likely expect the second, and we have never measured it.",
            "Does stock v0.2.12 still have the batch M-RoPE bug #1242? window-212.sh:4-5 asserts it; this inventory did not re-verify it. If it is still there, image correctness under batching is a separate selling point.",
            "Is the RTX PRO 6000 host (RTXPRO6000.md) or the RTX 5090 Windows box still available for a single-GPU v0.2.12 run? Not found.",
            "Where is the paper LaTeX source (src/main.tex, cited in paper-draft/linkedin-post-2026-10-17.md)? Not found under /media/raid0/StrataQ38FN or /home/administrator; it may have lived in an old scratchpad.",
            "Do our gains hold under upstream's protocol (thinking on, temperature 1.0, concurrency 1/4)? Sampling with temperature > 0 may change which engine paths engage (lanes, solo pipeline, early sample), and bit-identity checks don't apply there.",
            "Is the batch_overlap_test FAIL on PT ('ERR BGEN: no such free slot', results/w212b-penalised/overlap-PT.txt) a test-harness issue or a real slot-accounting bug? A maintainer reviewing our PRs would hit it."
          ]
        },
        "verify": {
          "checked": [
            {
              "claim": "The stock arms B and C were measured with the model-card presence_penalty 1.5 and have not been re-run since the fix, so the stock decode, aggregate, TTFT and join-gap baselines are invalid.",
              "verdict": "confirmed",
              "correction": "Confirmed by TESTLOG.md:131 (#99, 'All 212-server arms of #98 ... measured penalised; their decode/agg numbers are invalid'), window-212.sh:351, commit 211827a (Sun Nov 8 14:04:39 2026), and presence_penalty 0 in configs/T212-B.json:107 and T212-C.json:106. w212c ran only A P PT PTD PT4 PTX (results/w212c/summary.txt). All B/C figures match results/w212/summary.txt rows B, C and C-r2. One correction: C-r2's TTFT p50 is 0.84, not 1.03. Nuance: the bench sends temperature 0 explicitly (window-212.sh:559, 607), so only the card's presence_penalty 1.5 survived the merge (upstream serve/server.py, a8091f4d `_run`). Penalised B/C were therefore not greedy and generated different text, a second reason they are invalid. The join gap may stay high on stock even without the penalty, because PIPE_OVERLAP is our P9 and not in stock. That is unmeasured."
            },
            {
              "claim": "The only valid result is PT vs our own old production A: +3.8 % short, +3.2 % long, +5.0/+5.2/+4.7 % decode at 1/2/4 users, 62/62 identical replies, equal join gap 0.13 s.",
              "verdict": "confirmed",
              "correction": "Numbers match the decisions section of results/w212c/summary.txt. I re-ran compare_replies.py on results/w212c/cmp.requests.jsonl: A vs P, PT, PT-r2, PTD and PT4 are 62/62 identical, and PTX is 0/62. These are short replies only; the table's ident column shows '-' because P_REF was none (window.log 14:05:08)."
            },
            {
              "claim": "Arm B is not stock as shipped. It is the stock binary with our production config (about 30 STRATA_* switches, production args, parity pins), and no setup-defaults stock arm was ever measured.",
              "verdict": "confirmed",
              "correction": "configs/T212-B.json has 28 STRATA_* env keys. They include MMQ_DIRECT, SLOT_COPY_DIRECT, ADMIT_MERGE_FIRST, SLOT_CKPT_MOVE=1, QSA_SLOTS, HANDOFF_PIPE=8, and the pins SM86_PERF, GDN_TC, GDN_CONVL2, PROMPT_ATTN_IMMA, AUTO_NOBORROW, PF_W16_CACHE_MIB and MMQ_KQ set to 0. The args are production's (--spec 4, --resident-experts, --conversation-cache-mib 126976, --kv int8, --max-context 262144). The arm definitions in the window-212.sh header have no setup-defaults arm."
            },
            {
              "claim": "Upstream's protocol differs from ours, and we never ran `strata bench run` or tools/strata_bench.py.",
              "verdict": "confirmed",
              "correction": "BENCHMARK_PROTOCOL.md:3 (draft, pilot pending), :68/:71 (temperature 1.0, presence 0), :104-107 (21 requests at concurrency 1, optional 12 at concurrency 4), :7-10 (2-4 min, hashing 1-5 min). grep for strata_bench / 'strata bench run' in linux/*.sh, *.py and results/*.md: not found. One addition is wrong by omission. Upstream's own quick bench, tools/strata_bench.py:141-145 (behind bench.sh), sends temperature 0 with thinking off and no presence_penalty, so on a v0.2.12 server it is penalised too (BENCHMARK_PROTOCOL.md:507 confirms the temperature 0 / thinking-off shape). Only `strata bench run` avoids the penalty."
            },
            {
              "claim": "Our multi-user advantage shows only at concurrency 2 or more; upstream's protocol measures at most concurrency 4, so the 16-user and join results need our own published scripts. The join test uses upstream's tools/piped_read_join_bench.py.",
              "verdict": "confirmed",
              "correction": "Partly overstated: at 1 user PT is also ahead (145 vs A 138, and vs penalised stock 119-131), so the gains are not only at 2 or more users. The '+15 % at 1 user, +40-50 % at 2' in TESTLOG #98 was measured against penalised stock, so it is unproven against a fair baseline."
            },
            {
              "claim": "Engine-level identity holds (bt B=A, H=B, I=H, P=A at 16/16). Over HTTP, B's replies differ from A's because the server code differs.",
              "verdict": "wrong",
              "correction": "The engine-level part is confirmed (results/w212/summary.txt:46-49, TESTLOG #98/#99). The HTTP explanation is wrong. B vs A 'vsA:DIFF' is explained by the penalty: the v0.2.12 server merged the card's presence_penalty 1.5 into the temperature-0 requests (a8091f4d, serve/server.py `_run`), while A's server sends none. 'Other server code' is the explanation TESTLOG #98 gave before #99 found the penalty. A penalty-free B should be re-checked over HTTP and may well be 62/62 equal to A, which would be a stronger identity claim for upstream."
            },
            {
              "claim": "The statistics are thin: 2 runs per arm, fixed order, no CI, about 1 % spread, and the power limit changed across windows.",
              "verdict": "confirmed",
              "correction": "A gives 509.1 / 510.2 and PT 526.1 / 532.0 (w212c). A gave 514.8 in w212 vs 509.1 in w212c, a drift of about 1.1 % across windows, which matters when comparing w212c PT with w212 C. Both w212 and w212c ran at 320 W (TESTLOG #98 and #99; w212c window.log line 12), so there was no power difference between those two windows. The 280 W case only applies to other windows. The R2_FIRST definition is near window-212.sh:77, not line 104 (minor)."
            },
            {
              "claim": "All v0.2.12 data is from 4x3090. The only older single-card data (RTX PRO 6000, 0.1.44.1) shows our multi-user gains do not carry over to one card.",
              "verdict": "unverifiable",
              "correction": "I did not re-open RTXPRO6000.md or COMPARISON-2026-10-15.md. That all v0.2.12 data is from 4x3090 matches every w212* run I checked. Whether those hosts are still available: not found."
            },
            {
              "claim": "Vision on PT is GO 8/8 and 4/4. The summary's '0 GO, 2 NO-GO, 3 INCONCLUSIVE' is a parser artifact, and stock was benched text-only because of #1242.",
              "verdict": "confirmed",
              "correction": "Confirmed at results/w212c/PT-vision.txt:24-31 and the w212c window.log 14:26:18 line. #1242 is a PR, not an issue: 'Isolate concurrent image positions and batch-MTP draft paths'. It was CLOSED on 2026-10-19 (tracker reset) and not merged (gh pr view 1242). I found no M-RoPE batch fix in the 146df3c4 log. That the bug is still present in stock v0.2.12 is likely but was not run-verified."
            },
            {
              "claim": "Open failures: batch_overlap FAIL on PT, bgc / bgc-I FAIL, serve unittests rc 124, E not engaged, no soak; lowest VRAM 814 MiB.",
              "verdict": "confirmed",
              "correction": "Confirmed at results/w212b-penalised/overlap-PT.txt (14 background slots, then 'ERR BGEN: no such free slot'), results/w212/summary.txt:39, 51-57, and the w212c VRAM column (3:814)."
            },
            {
              "claim": "Public material is stale (paper v1.3, one-pager v3, LinkedIn post from 2026-10-17/18). PR #2364 is not in COMMUNITY_BENCHMARKS.md, and the tracker reset closed our PRs.",
              "verdict": "confirmed",
              "correction": "paper-draft/ listing matches. gh pr view 2364: author havok2-htwo, created 2026-10-14, CLOSED 2026-10-19T17:19:24Z (not merged). grep for 3090, havok and 2364 in Strata-212/docs/COMMUNITY_BENCHMARKS.md: no match. I did not re-run the .tex search."
            },
            {
              "claim": "Reusable tooling exists (window-212.sh with ARMS/RUN/NOW, compare_replies.py, bench_* and others), and community_bench_144.sh must move to port 8091.",
              "verdict": "confirmed",
              "correction": "Confirmed for window-212.sh (header and lines 66-78) and compare_replies.py, which I ran. I did not open the eval files."
            },
            {
              "claim": "Downtime: about 6.7-7 min per arm run plus overhead; a combined window of about 2 h, about 4 h with error bars.",
              "verdict": "confirmed",
              "correction": "Measured: w212c ran from the 14:05:10 stop to the 15:21:54 'window done', 76.7 min for 11 arm runs including the PT vision check (26 s). The remaining figures are the agent's estimates, not measurements."
            }
          ],
          "missed": [
            "Upstream already has the penalty finding as issue #2940: '0.1.45: non-thinking requests inherit the card's presence_penalty 1.5 (temperature 0 too), ~25% slower decode', by Cass67, 2026-10-17. Others reproduced it there, and Vop3p explained the mechanism (any penalty sends hist_n>0 to the serial loop and turns off --pipeline-windows). It was closed NOT_PLANNED on 2026-10-19 in the 0.2.0-beta tracker reset. Vop3p re-tested on 0.2.2 ('still there') and commented /reopen. Our w212 data (stock B/C with and without the penalty, same binary) would be a strong supporting comment, but only with the user's text approval and as havok2-htwo. A related open issue is #3354, '[Engine] qwen35moe: the model-card sampling is applied silently and disables MTP drafting'.",
            "Upstream's own quick bench (tools/strata_bench.py:141-145, bench.sh) sends temperature 0 with thinking off and no presence_penalty, so on 0.1.45+ servers it is penalised too. Stock numbers in community reports made with bench.sh on 0.1.45+ are therefore penalised. Any comparison we publish must state that both arms send presence_penalty 0, or use `strata bench run`, which pins presence 0.",
            "The penalty explains B vs A 'vsA:DIFF' over HTTP, not 'other server code'. A penalty-free B could give bit-identical HTTP replies to A and PT, a stronger exactness claim for upstream. That should be checked in the re-run.",
            "B/C 'greedy' replies were not greedy-equivalent: the penalty changes tokens, so the penalised throughput figures compare different generated text, not only different engine paths.",
            "The comparison in the task context mixes windows: PT from w212c (day, 14:05) with B/C from w212 (night). A's own throughput drifts about 1 % between them (514.8 vs 509.1). A fair stock-vs-PT figure needs B, C and PT in the same window, ideally interleaved."
          ]
        }
      }
    ]
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Scout"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Verify"
    },
    {
      "type": "workflow_phase",
      "index": 3,
      "title": "Plan"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "scout:bench-infra",
      "phaseIndex": 1,
      "phaseTitle": "Scout",
      "agentId": "ae861313fdcc0d5a7",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149382121,
      "queuedAt": 1794149382100,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "Upstream Strata (origin https://github.com/Niko1221/Strata.…",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149574767,
      "tokens": 130188,
      "toolCalls": 32,
      "durationMs": 192645,
      "resultPreview": "{"summary":"Upstream Strata (origin https://github.com/Niko1221/Strata.git, v0.2.12 = 146df3c4) has two official ways to benchmark, plus a set of A/B tools for engine PRs. No CI workflow runs benchmarks.\
\
1. **Standard benchmark** (`strata bench run`). Spec: docs/BENCHMARK_PROTOCOL.md, still a draft, protocol id strata-qwen38-thinking-v1-draft.1. It sends frozen fixtures from bench/prompts/stand…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "scout:github-discussion",
      "phaseIndex": 1,
      "phaseTitle": "Scout",
      "agentId": "a593d4ff24ffbc40c",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149382121,
      "queuedAt": 1794149382100,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "Read-only survey of what Niko1221/Strata upstream (https://…",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149632594,
      "tokens": 115918,
      "toolCalls": 30,
      "durationMs": 248748,
      "resultPreview": "{"summary":"Read-only survey of what Niko1221/Strata upstream (https://github.com/Niko1221/Strata, remote from /media/raid0/StrataQ38FN/Strata-212) has said about havok2-htwo's work. Our public events (gh api users/havok2-htwo/events/public, the complete list for this repo) are: the fork on 2026-10-14, 3 PRs (#2364 bench report, #3022 HC_Q8 fix, #3559 HC_Q8 fix resubmitted with the [Engine] prefix…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "scout:our-delta",
      "phaseIndex": 1,
      "phaseTitle": "Scout",
      "agentId": "a20c672e870ae6d99",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149382121,
      "queuedAt": 1794149382101,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "I inventoried our series on top of v0.2.12 (146df3c4). It i…",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149645819,
      "tokens": 138457,
      "toolCalls": 29,
      "durationMs": 261973,
      "resultPreview": "{"summary":"I inventoried our series on top of v0.2.12 (146df3c4). It is 56 commits on g3-212-pipe (HEAD a099d295), all by havok2-htwo noreply, touching 86 files with +20,951/-464 lines. src/program/generate.cpp alone has +4,585/-226, and serve/server.py has +920. The commits are in 5 patch dirs under /media/raid0/StrataQ38FN/linux/patches/: v0.2.12-p1p2, -convmem, -vision, -serve and -pipe. Those…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "scout:our-evidence",
      "phaseIndex": 1,
      "phaseTitle": "Scout",
      "agentId": "a0b418e9fd79d0837",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149382121,
      "queuedAt": 1794149382101,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "Our current evidence cannot yet back an upstream-credible '…",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149721455,
      "tokens": 109608,
      "toolCalls": 30,
      "durationMs": 337609,
      "resultPreview": "{"findings":[{"claim":"The main stock-vs-ours numbers (arm B = stock v0.2.12 and arm C = B + the sm_86 table) were measured with the model-card presence_penalty 1.5. They have not been re-run since the fix. That makes the stock decode, aggregate, TTFT and join-gap figures in the headline invalid as a fair stock baseline.","evidence":"linux/TESTLOG.md #99 says: 'All 212-server arms of #98 (and w212…"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "verify:bench-infra",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a65ba282a7dfc2095",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149577077,
      "queuedAt": 1794149577061,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149740860,
      "tokens": 96855,
      "toolCalls": 25,
      "durationMs": 163783,
      "resultPreview": "{"checked":[{"claim":"Standard benchmark is `strata bench run`, defined in docs/BENCHMARK_PROTOCOL.md, still draft (strata-qwen38-thinking-v1-draft.1), fixtures bench/prompts/standard-v1, corpus SHA e75eff8b..., from 0.2.1.","verdict":"confirmed","correction":"Confirmed in BENCHMARK_PROTOCOL.md lines 14-19 (it also says BENCH.bat/bench.sh/strata_bench.py do NOT implement it and that the approved m…"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "verify:github-discussion",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a2af5ebc0583b7ecc",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149634323,
      "queuedAt": 1794149634306,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149812397,
      "tokens": 64853,
      "toolCalls": 15,
      "durationMs": 178074,
      "resultPreview": "{"checked":[{"claim":"Complete inventory: fork 2026-10-14, 3 PRs (#2364, #3022, #3559, all closed, none merged on GitHub), 0 own issues, 8 comments on 6 threads (#1242, #2364, #2275, #2326 x2, #2597, #2235), nothing open.","verdict":"confirmed","correction":"gh api users/havok2-htwo/events/public lists exactly these 11 events. search/issues involves:havok2-htwo returns 7 items, all closed. Per-thr…"
    },
    {
      "type": "workflow_agent",
      "index": 7,
      "label": "verify:our-delta",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a7ffdfee581394121",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149647651,
      "queuedAt": 1794149647639,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149794333,
      "tokens": 77299,
      "toolCalls": 16,
      "durationMs": 146682,
      "resultPreview": "{"checked":[{"claim":"Series size and shape: 56 commits by havok2-htwo noreply on 146df3c4..a099d295, 86 files +20951/-464, generate.cpp +4585/-226, server.py +920, 5 patch dirs (8/17/3/8/17), only one README (patches/v0.2.12-README.txt), upstream main = 146df3c4","verdict":"wrong","correction":"Almost all of it holds. I checked with git log/diff in /media/raid0/StrataQ38FN/Strata-212p, ls on linu…"
    },
    {
      "type": "workflow_agent",
      "index": 8,
      "label": "verify:our-evidence",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "aa5022af2db502574",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149723384,
      "queuedAt": 1794149723367,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149871703,
      "tokens": 78565,
      "toolCalls": 19,
      "durationMs": 148319,
      "resultPreview": "{"checked":[{"claim":"The stock arms B and C were measured with the model-card presence_penalty 1.5 and have not been re-run since the fix, so the stock decode, aggregate, TTFT and join-gap baselines are invalid.","verdict":"confirmed","correction":"Confirmed by TESTLOG.md:131 (#99, 'All 212-server arms of #98 ... measured penalised; their decode/agg numbers are invalid'), window-212.sh:351, commi…"
    },
    {
      "type": "workflow_agent",
      "index": 9,
      "label": "plan",
      "phaseIndex": 3,
      "phaseTitle": "Plan",
      "agentId": "ae6e7970b367bdd33",
      "model": "claude-opus-5-5",
      "state": "done",
      "startedAt": 1794149873422,
      "queuedAt": 1794149873420,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /media/raid0/StrataQ38FN; sed -n 1,110p linux/window-212…",
      "promptPreview": "HARD RULES (all agents):
- Strictly read-only. Never post, comment, react, open/close PRs or issues, push, or run any gh command that writes (no gh pr create/comment/review, no gh issue create/comment, no gh api -X POST/PATCH/PUT/DELETE). Read-only gh (gh issue list/view, gh pr list/view, gh api GET, gh search) is fine.
- Never touch production: no systemctl on strata.service, no requests to port …",
      "promptFramed": true,
      "lastProgressAt": 1794149983531,
      "tokens": 88365,
      "toolCalls": 1,
      "durationMs": 110109,
      "resultPreview": "**Plan: submitting our v0.2.12 series upstream and showing "we are faster" (Niko1221/Strata)**

Sources: the four scout+verify results, `/media/raid0/StrataQ38FN/linux/window-212.sh` header lines 1-78, `/media/raid0/StrataQ38FN/Strata-212/docs/BENCHMARK_PROTOCOL.md` lines 1-60 and `/media/raid0/StrataQ38FN/Strata-212/tools/strata_bench.py` CLI lines 1031-1037. Claims the verify step marked wrong a…"
    }
  ],
  "totalTokens": 900108,
  "totalToolCalls": 197
}
