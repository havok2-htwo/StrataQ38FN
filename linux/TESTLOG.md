# Test log (chronological)

Every attempt is recorded here **before** the next one starts: date, what was tested, result, verdict, raw data. Before a
new test, check here whether it has already been run. Load profile unless stated otherwise: `bench_concurrency.py`, 2
requests per user, 512 output tokens (fixed), thinking off; "short" ≈ 170 prompt tokens (measured 162–178; this header
said ≈ 120 until 2026-10-12, which was wrong for every run), "long" / "1.2K" ≈ 1.23K (1,232–1,243). The prompt embeds the
run label, so its length varies by a few tokens with the config name (~2 ms per admission, ~0.2 %). Temperature: **0.7 up
to #47** (`run_strata_variant.sh` and the series scripts use the bench default), **greedy from #49 on** (`ab_quick.sh`,
`--temperature 0`); #48 is engine level. The two protocols are not directly comparable: the same config `T16-142` gives
466.9 / 342.1 tok/s at 0.7 (#46, 3 runs) and 469.7 / 349.4 greedy (#49, 2 runs), +0.6 % / +2.1 %, partly session noise.
Numbers = total throughput in tok/s (output) or decode per user, TTFT in s. All Strata runs: UD-IQ4_XS, 4× RTX 3090.

**Noise:** at 16 users the run-to-run sd is ~0.2–0.4 % for the current configs and up to 0.8 % (`T16-142`, #46); the
largest single-run span seen is 1.5–1.6 % (#46 short, #49 `T16-142` ab-fc vs ab-fc2). A single run per arm resolves
differences of about **±1.2 %**; smaller differences are ties and are marked "within noise". Every adopted default so far
rests on ≥ 2 % effects; sub-2 % levers are judged at engine level or with repeated, interleaved arms.

Verdicts: ✅ adopted · ➖ no measurable effect · ❌ worse / discarded · 🔬 insight only

## 2026-10-10

| # | Test | Result | Verdict | Raw data |
|---|---|---|---|---|
| 1 | Strata setup default, solo (no `parallel`), with/without `--resident-experts` | 121 tok/s single, ~88 total; without resident mode 60 GB RAM → oomd kill | ✅ `--resident-experts` mandatory | `strata-A-solo*` |
| 2 | `parallel 8` (upstream limit), split auto | ~177 short/long, 31 tok/s per user | 🔬 baseline | `strata-B-batch8` |
| 3 | Engine patch 1: 16/32 slots (8 rows per group) | 16 slots bit-identical, batch decode 603 tok/s; over HTTP only ~195 | ✅ patch (HTTP limit = prefill) | `batch_test_16/32.txt`, `strata-C/D-*` |
| 4 | Prefill chunks 256/512/768/1024 + decode share 1/2 | 512 best (~246–250 long); decode share has no effect (see #21) | ✅ `--prefill 512` | `strata-E*`, `strata-F*` |
| 5 | Slots 12/16/20/24/28 (short prompts) | plateau ~330–361, 12 slots best latency | 🔬 | `strata-H*-short` |
| 6 | Users 24–40 at 12 slots | flat 346–358, wait time rises | 🔬 | `strata-U-b12-users` |
| 7 | 3 GPUs (`--trim-stage-weights`, split 16/32) | 117 single / 266 batch | 🔬 4th card only useful under load | `strata-T3-*` |
| 8 | UD-Q4_K_XL | 97% of experts in VRAM, 100 single / 241 batch | 🔬 IQ4_XS remains the target | `strata-K-*` |
| 9 | MTP off (`--spec 4` without `--mtp`) | 81 instead of 117 tok/s | ✅ MTP on (+45%) | `strata-M-nomtp` |
| 10 | Prefill across length 2K–28K: chunks 512 / auto / auto:32768 | 512: 1,247→2,239 tok/s, best overall picture | ✅ 512 | `strata-P-*` |
| 11 | vLLM 0.29 Flash-Next W4A16 (EP, FP8 table via patch), prefix cache on/off | 65–75 single, ~420–450 plateau, max ~21 concurrent | 🔬 | `vllm-w4a16-*` |
| 12 | vLLM 0.31 Flash-Next, prefix cache on | 79 single, ~400 long / ~710 short, cap ~22 | 🔬 | `vllm031-flashnext-pc-*` |
| 13 | vLLM 0.31 Flash-Next with MTP at 64 and 32 seqs | OOM and no KV space, respectively | ❌ | `vllm031_mtp*.log` |
| 14 | vLLM 0.31 prefill across length | ~3,500 tok/s constant | 🔬 | `vllm031-P-*` |
| 15 | vLLM 0.31 27B FP8 preset | 64 single, 409 long / 838 short (64 users), ~100 GB RAM | 🔬 | `vllm031-27b-fp8-*` |

## 2026-10-11

| # | Test | Result | Verdict | Raw data |
|---|---|---|---|---|
| 16 | vLLM 0.31 Flash-Next MTP with 8 seqs, 1,024 batch tokens, 0.93 | **136 single**, 490 short (8 users), 65K KV | 🔬 MTP only works with low parallelism | `vllm031-flashnext-mtp2-s8-*` |
| 17 | Quality set (MMLU-Pro 240 / GSM8K 150) | Strata IQ4_XS 77.9/94.0 · 27B 72.1/95.3 · vLLM-W4A16 70.4/94.7 · RAZOR 62.5/92.7 | 🔬 | `eval/eval_results.jsonl` |
| 18 | RAZOR-65B (50% of experts pruned, NVFP4) in vLLM | 111 single, ~412 long; quality dropped sharply, German output flawed | ❌ model deleted | `vllm031-razor65-*` |
| 19 | Hub survey of 910 Flash-Next repos | no smaller vLLM-compatible quant without quality loss | 🔬 | `quant-survey/` |
| 20 | 16 slots in 2 groups; 12 slots in 2 groups | 266–272 / 264–270 short, 200 long | ❌ half the cards idle | `strata-Q2-*` |
| 21 | Decode share 4.0 | 366–375 / 259 | ➖ no effect in pipeline mode (`generate.cpp:9706` upstream) | `strata-Q3-*` |
| 22 | 2 instances of 2 GPUs each (round-robin) | 152–182 short, 100 long, 90 single | ❌ | `strata-Q1-dp2-*` |
| 23 | **Split `12,24,36` + `--trim-stage-weights`** | **405 short / 288 long** (+14/+16%), no buffer borrowing | ✅ | `strata-T16-trim-*` |
| 24 | same, 20 / 24 slots | 409/288, 416/293 | ➖ plateau | `strata-T20/T24-*` |
| 25 | same, chunks 1,024 / 2,048 | 406/281, 403/290 | ➖ | `strata-T16-pf*` |
| 26 | Multi-turn 8/16 agents, trim config 32K vs. agent config (262K, `--kv-resident`, 24 GB conv. cache) | follow-up turn TTFT 7.1/10.4 s → **4.8/5.0 s**, 16 agents 43 → 73 tok/s | ✅ agent config | `multiturn.jsonl` |
| 27 | Multi-turn against vLLM 0.31 (+20 GB RAM KV tier) | 8: 89 tok/s, 6.4 s; 16: 77 tok/s, **32 s** | 🔬 Strata ahead under agent load | `multiturn.jsonl` |
| 28 | Engine patch 2: direct slot copy GPU→GPU | copy 65.6 → 34.0 ms, 405/294, agent TTFT 4.5 s; bit-identical | ✅ (small) | `exact_direct_*`, `strata-T16-direct-*` |
| 29 | Profiling the batch window (`STRATA_VERIFY_PROFILE`) | GPU 5.1–6.1 ms of ~6.95 ms per step; card 4 +0.73 ms head | 🔬 overhead small | `prof_batch16*` |
| 30 | Split `12,24,37` / `13,25,37` | 399/289, 397/282 (one run each, HTTP only) | ➖ **superseded by #56**: on 0.1.42 with B-tab and `--aux-cpus` the same 12,24,37 gives +5 % and is the default since 0.12.0; why it was flat here was not re-checked at engine level | `strata-T16-split*` |
| 31 | `--ple-io ram` | **417 / 297**, 1.2K prompt 821 instead of ~1,100 ms | ✅ | `strata-T16-pleram-*` |
| 32 | Overlapped admission, prototype (`STRATA_PIPE_OVERLAP_SPIKE`) | 413/294 (no gain), **TTFT p50 2.03 → 0.57 s** | 🔬 latency lever only | `strata-S16-spike-*`, `patches/experimental/` |
| 33 | Prefill timing of short prompts (`STRATA_PREFILL_TIMING`) | 154 tok.: 78–87 ms per card, 31% "dequant" = expert copy for MMQ | 🔬 | `prefill_timing.txt`, `upstream/ANALYSIS-short-prompt-moe.md` |
| 34 | `STRATA_PF_FUSED=1` | no change (only kicks in from 1,024-token chunks) | ➖ | `prefill_timing_fused.txt` |
| 35 | Design "MMQ direct on padded cache slots" (idea B) — code review | rejected: cache is profile-ordered (layers interleaved) and changing MMQ's channel count changes stream-k rounding → not bit-identical; exact alternative "B-tab" = own copy of MMQ's outer kernel with a per-expert pointer table | 🔬 design in `upstream/DESIGN-mmq-direct.md` | `upstream/DESIGN-mmq-direct.md` |
| 36 | Upper bound: skip the resident-expert gather (`STRATA_HACK_SKIP_GATHER=1`, wrong outputs, branch `g3-hack-skip-gather`) | prompt read ~162 tok: 397–413 → **240–292 ms** (−35–40%); ~1,228 tok: 900–908 → **609–619 ms** (−32%) | 🔬 B-tab worth building (est. +12–15% multi-user throughput, −30–40% TTFT, also single GPU) | engine log 2026-10-11 ~12:00, `configs/HK-*.json` |
| 37 | **B-tab implemented** (branch `g3-mmq-tab`, build `-DSTRATA_MMQ_TAB=ON`, env `STRATA_MMQ_DIRECT=1`; patches `patches/mmq-tab/`) | unit test 874/874 bitwise; batch_test16 16/16 and interleave identical; old vs new: tokens **and** fp32 logits byte-identical (4 GPUs, 1 GPU, streamed experts); prompt read −20..−24% (~170 tok), −18..−20% (~1.2K); default build (option off) produces byte-identical device code | ✅ | workflow report, `upstream/DESIGN-mmq-direct.md` |
| 38 | B-tab through the HTTP server, 16 users (same session reference in brackets) | short **443.6** (417), long **327.0** (295.5), 16 agents multi-turn **72.4 tok/s** (61.6), follow-up TTFT 4.57 s (5.05), single-user TTFT 1.2K 0.73 s (0.88) | ✅ new default (`configs/T16-mmqtab.json`, agent: `X16-agent-mmqtab.json`) | `results/series_mmqtab.out`, `strata-T16-mmqtab-*`, `multiturn.jsonl` |
| 39 | **M0 baseline for overlapped admission** (engine level, reference engine, 15 slots decoding, `results/overlap/m0/`) | all-slot stall per admission **0.28 s short / 0.67 s 1.2K** (design assumed 0.44/0.94); prompt read (`sp.run`, the overlappable part) 201 / 609 ms = 64 % / 85 %; fixed rest ~112 ms (drain 27, header windows 24, ramp 24, checkpoint 17, first window 11.5, copy 7.8); D_ref 591 rows/s (16 slots). Gate holds in T16-mmqtab (no loans, 100 % resident, `--ple-io ram` = Mmap mode, no PleReader); X16 agent configs (`--kv-resident`) are refused | 🔬 model `D/(1+D·R/L)` predicts 446/333 vs measured 444/327 | `results/overlap/m0/` |
| 40 | **Overlapped admission implemented** (branch `g3-overlap`, 5 commits, env `STRATA_PIPE_OVERLAP=1`, engine `strata-g3-overlap`): M1 PLE concurrency setters, M2 gate + pre-capture of all group layouts, M3 read on a worker thread while `pump` keeps the groups decoding; review found 2 real defects (pad-up over a parked slot cache; spin waits not released on a CUDA fault), both fixed | exactness matrix all green: env off = reference engine (batch_test16/32, interleave, both splits); env on: batch_overlap_test every slot/admission = solo, env 0 vs 1 token- and RESUME-identical, forced pad-up/wait arms identical, 64-request soak 64/64, BSTOP/STOP/BYIELD mid-read, fault cases (a)–(e) exactly one ERR + bounded exit; 1.2K read beside decoding 826 ms (609 drained), decode lost 420 ms → x ≈ 0.7 | ✅ exact; throughput → #41 | `results/overlap/{m2,m3,r1,exactness-r1}/` |
| — | Found on the way (reference engine too, not the overlap): a next turn continued from its parent's slot after a reply decoded in the batch can differ from a fresh solo read (soak2: r16/r32/r40/r48 differ at token 15–54, RESUME ~845) | Review 2026-10-12: most likely a **reference artefact**, not an engine defect. soak64b/reuse4 compute the child's solo after 6 unrelated GENs; the engine's prompt cache keeps 6 checkpoints, so the parent is evicted and the solo child is a fresh prefill re-read ("0 reused"), while the slot child resumes ~845 tokens of decode-path state. Every RESUME-0 child is identical; the interleave test, whose solo continues the live chain, is identical in every arm. Not yet proven: only one prompt shows slot-resumed = live-continued solo | 🔬 **re-check pending**: solo-only check (GEN parent; GEN child live-continued vs. GEN child after an unrelated GEN), then fix the soak64b/reuse4 reference | `results/overlap/exactness-r1/soak2-ref*.out`, `reuse4.py` | **Resolved in #59**: solo draft windows leave a different state than one-row slot windows.
| 41 | **Overlapped admission through the HTTP server (M5)**: `T16-overlap` vs `T16-overlap-off` (same binary), 3 repetitions each, means; TTFT p50/p95 in s | 16 users short: 440.8 → 436.8 tok/s (−1 %), TTFT p50 **1.74 → 0.47**, p95 5.25 → 6.36, e2e p50 18.5 → 17.0 · 16 long: 327.1 → 327.7 (0 %), TTFT p50 **4.13 → 1.00**, p95 10.96 → 13.72, e2e p50 25.0 → 21.0 · 24 short: 424.1 → 433.6 (+2 %), p50 3.09 → 3.87 · 24 long: 316.4 → 337.5 (+7 %), p50 6.83 → 9.32, p95 36.1 → 28.9 · 1 user unchanged. Engine stats: short read 305 ms beside decode (201 drained), decode lost 158 ms; 1.2K read 840 ms (609), lost 418 ms; 0 pad-ups, 0 waits | ✅ latency lever (opt-in): median TTFT −73/−76 % and e2e −8/−16 % at 16 users, throughput neutral; ❌ p95 +20–25 % (stretched reads queue up), p50 worse above 16 users. Why no throughput: the admitted slot itself idles during the longer read, which eats the decode the other slots win | `results/series_overlap.out`, `strata-T16-overlap*-r{1,2,3}.txt`, `engine-strata-T16-overlap-*-r3.log` |
| 42 | **M4 knob arms of the overlap** (1 repetition each, 16/24 users; reference = #41 "on" means 436.8/327.7 tok/s, p95 6.36/13.72 s) | `PRIO=0` (decode streams at normal priority): 16 short 441.0 (p95 6.11), 16 long 332.1 (p95 12.58), 24 long 345.9; 1.2K read 742 ms instead of 840, decode lost 459 instead of 418 ms · `SLEEP_US=500`: 431.1 / 326.9, no change · `PRIO=0`+`SLEEP_US=500`: 437.3 / 329.5 · `CUDA_DEVICE_MAX_CONNECTIONS=32`: 435.5 / 325.7, no change | ➖ all within ~1–2 %; PRIO=0 slightly ahead (shorter reads, p95 −8 %), not enough to change the default without repetitions | `results/series_ovm4.out`, `strata-T16-ovM4-*` |
| 43 | `STRATA_PIPE_OVERLAP_MAX_QUEUE=0/1` (engine `strata-g3-ovq`, Strata 4e3e05a6: overlap a read only while ≤ n further BGEN lines wait in the engine's input), exactness + 2 repetitions 16/24 users | exact (batch_overlap_test token- and RESUME-identical to #40); throughput/TTFT = #41 "on" within noise (16 short 435, long 327; p95 6.4 / 13.8 s) — **0 reads drained**: the server takes admissions one at a time (control lines), so the queue sits in `server.py`, never in the engine's input | ❌ no effect as built → server now passes `waiting=N` with BGEN (Strata 645b4bce), see #44 | `results/series_ovq.out`, `results/overlap/ovq/` |
| 44 | `MAX_QUEUE=0/1` **with the server's `waiting=N`** (Strata 645b4bce), 2 repetitions, 8/16/24 users; plus off/on at 8 users | ~90 % of reads now drained (q0: 84 of 94, q1: 77 of 94). 16 short: 446 tok/s, TTFT p50 **2.2–2.35** (off 1.74, on 0.47), p95 4.8 (off 5.25, on 6.36) · 16 long: 330, p50 **5.6** (off 4.13, on 1.00), p95 11.0 (= off) · 8 users off → on: short 337.4 → 337.7, p50 1.26 → **0.44**, p95 2.60 → 3.10; long 266.1 → 271.2, p50 2.89 → **0.94**, p95 5.49 → 6.56 | ❌ the queue rule makes the median worse than both off and on and buys only ~10 % of p95; in a closed loop someone almost always waits. Plain overlap stays the latency option (8 users: p50 −65/−67 %) | `results/series_ovq2.out` |
| 45 | **Rebase onto upstream 0.1.42** (61b3fb5d, +235 commits; worktree `Strata-142`, branch `g3-142`, engine `strata-g3-142`): all 17 commits carried over (none upstream yet); adapted to upstream's `batch_rows_error` (48f2c47d), non-blocking snapshot copies (a6346cdc, D2D slot copy now **opt-in** `STRATA_SLOT_COPY_DIRECT=1`), `--aux-cpus` pinning, BYIELD in batch reads (cec7e2b1); review found 5 real defects (group re-resolve after carving could exceed 8 rows/group, HIP compile break on `cudaErrorIllegalAddress`, …) - fixed (650001fb, e20b3a18) | exact: batch_test16 16/16, batch_test32 32/32 (4 groups of 8, 574 tok/s engine level), tokens = old engines; batch_overlap_test on = off (token + RESUME); prefill_mmq_tab_test 874/874; MMQ_DIRECT 0 = 1. `STRATA_ROUTE_TAIL_SKIP=7` does nothing here (it only skips experts missing from VRAM; all resident). HTTP 16 users (slot copy direct off): short **455.0** (443.6), long **341.1** (327.0), TTFT p50 1.59 / **2.68 s** (1.73 / 4.15), 1 user TTFT 0.27 / 0.62 s (0.33 / 0.73) | ✅ new base (+3/+4 % throughput, long TTFT −35 % from upstream's changes) | `results/rebase142/` |
| 46 | **0.1.42 base vs + overlap**, 3 repetitions, means (`T16-142` = MMQ_DIRECT + SLOT_COPY_DIRECT; `T16-142-overlap` + STRATA_PIPE_OVERLAP=1), tok/s · TTFT p50/p95 s | base: 16 short **466.9** · 0.96/4.00, 16 long **342.1** · 2.60/9.64, 8 short 349.8 · 0.89/2.24, 8 long 274.9 · 2.59/5.03, 1 user 115.7/106.4 · 0.27/0.67 · overlap: 16 short 462.2 · **0.43**/5.02, 16 long 346.7 · **0.90**/12.12, 8 short 348.2 · 0.40/2.70, 8 long 281.3 · 0.87/5.97, 24 long 354.5 (base 327.3). Std. dev. at 16 users 1–3 tok/s | ✅ **T16-142 new default** (+6 % short / +5 % long vs T16-mmqtab, TTFT p50 −45/−37 %); overlap stays the opt-in latency variant (p50 −55/−65 %, p95 +25 %) | `results/series_142.out`, `strata-T16-142*` |
| 47 | 24 / 32 slots on 0.1.42 (`T24-142`, `T32-142`, 1 run) | T24 @24 users: short **477.0** (T16 @24: 433.0), long 348.7 (327.3), TTFT p95 6.75 / 15.3 s (T16: 21.2 / 30.0); @32 users 455.5 / 340.0. T32 @32: 474.8 / 344.4. Per-user decode 21 / 16 tok/s instead of 31 / 25 at 16 slots | ➖ throughput plateau (+2 %); ✅ for more than 16 concurrent users T24 cuts queueing (p95 −68 %) at a lower per-user speed | `strata-T24-142-*`, `strata-T32-142-*` |
| 48 | **Fixed per-admission cost cut** (`upstream/DESIGN-admission-fixed-cost.md`; Strata-142 06633124..c9bbd1fc, engine `strata-g3-142fc`, 6 opt-in switches: `STRATA_PIPE_BYIELD_NODRAIN`, `STRATA_ADMIT_MERGE_FIRST`, `STRATA_CKPT_POOL`, `STRATA_CKPT_THREADS`, `STRATA_SLOT_CKPT_MOVE`, `STRATA_SLOT_COPY_BATCHSYNC`; review found 3 minor defects, fixed). Engine level, 24 admissions, all off → all on | exact (batch_test16 16/16, batch_overlap_test incl. RESUME, interleave = reference, 45 driver requests identical). Per admission: header windows 23.5 → 0, first window 11.5 → 27.7 (merged; each extra row costs 2.4–4 ms, plan assumed ≤1.45), checkpoint save 17.1 → 6.7, checkpoint copy into slot 7.3 → 0, slot copy 1.3 → 0.5, done → BADM 8.7 → 0.5; send → first BT −25 ms short / −21 ms long (plan: −40; critique: −40 → step results −7.3/−10.4/−7.3/−0.8). BYIELD no-drain saves ~25 ms only for a late BYIELD (the server's normal BYIELD costs 0) | ✅ exact; throughput → #49 | `results/fixedcost/{step0-o1,ab,k1k2,s1,s2,final}/` |
| 49 | Fixed-cost switches through the server (`ab_quick.sh`, 16 users, greedy, 2 runs each, means): `T16-142` vs `T16-142fc` (all 6 switches) vs `T16-142fc-overlap` (1 run) | short **469.7 → 493.0 tok/s (+5.0 %)**, TTFT p50 0.95 → 0.90, p95 4.52 → 3.58 · long **349.4 → 357.0 (+2.2 %)**, p50 2.80 → 3.03, p95 9.45 → 9.11 · + overlap: 486.3 / 361.2, p50 **0.37 / 0.98**, p95 4.71 / 11.62 | ✅ **T16-142fc new default**; overlap on top stays the opt-in latency variant. Note: both arms greedy, so +5.0 % / +2.2 % is the same-protocol gain; a 0.9.0 → 0.10.0 comparison 467 / 342 (#46, T=0.7) → 493 / 357 reads +5.6 % / +4.4 % and mixes protocols | `results/ab-fc.out`, `results/ab-fc2.out` |
| 50 | Smaller prefill chunks for short prompts (`T16-142fc` with `--prefill 128` / `64`, quick A/B, 16 users) | short 481.3 / 465.7 (512: 493.0), long 275.0 / 213.5 (357.0); a ~173-token read takes 202–228 ms (128) and 234–259 ms (64) | ❌ every chunk re-reads the routed experts (memory bound), so overlapping the cards with micro-chunks loses more than it gains. Consequence: reading SEVERAL queued prompts in one chunk should amortize the expert reads → design next | `results/ab-pf.out` |
| 51 | **Read time vs tokens** (engine level, `strata-g3-142fc`, idle and 15 slots decoding; k prompts joined as one read) and **queue probe** through the server (proxy logging `waiting=`) | single read: 68 tok 100 ms, 143 tok 145, 293 tok 230, 593 tok 356, 1193 tok 529, 2393 tok 911, 4793 tok 1721 ms; joined prompts read as fast as one prompt of the same length. Short reads: ~52 % expert-weight traffic (~93 ms fixed per chunk over 4 cards). Joined read would save 21–38 % (short) / 15–19 % (1.2K). Queue: in `ab_quick` 3 requests wait at most admissions, but only because every reply is exactly 512 tokens (the 4 slots of a group finish in the same window); with mixed lengths k ≈ 1 | 🔬 batching several admissions per read mostly helps the lockstep benchmark, not real traffic → only stage 1 (short admission hold) worth it; joined/merged reads dropped. Side finding: prefill stage threads all run on the host's CPU 0 → #52 | `results/batchadmit/{scaling,qprobe,pipecheck}/`, `upstream/DESIGN-batched-admission.md` |
| 52 | **`--aux-cpus`** (upstream 0.1.42 option, no code change): the host thread is pinned to CPU 0 and every thread it creates inherits CPU 0 while the option is off - including the 3 prefill stage threads and the PLE gather (19 threads on one core; "host grouping" 5 → 60–68 ms per card per chunk once 3–4 cards read). Arms: `auto` (only CPU 64 is spare), `65-79`, `65-127` (`T16-142fca`) | exact (batch_test16 16/16, solo + batch dumps equal with/without). 1.2K read 557 → 484 ms (2.4K: 911 → 673). 16 users: auto 488.4 / 365.5; 65-79 491.2 / 375.6; **65-127: 492.9 / 376.2** (no flag: 493.0 / 357.0) → long **+5.4 %**, long TTFT p95 9.1 → 8.1 s | ✅ **`T16-142fca` new default**; 65-127 vs 65-79 (+0.35 % / +0.16 %) is **within noise**, either range is fine; upstream: stage threads should leave the host's CPU even with the option off | `results/ab-aux*.out`, `results/auxcheck/`, `results/batchadmit/{pipecheck,auxab}/` |
| 53 | **Mixed reply lengths** (`bench_concurrency.py --max-tok-min 128`: 128–512 per request, fixed per id; `ab_quick.sh` with `MIX=128 ROUNDS=4`), 16 users, `T16-142fca` vs + overlap | default: short 409.8, long 290.3 tok/s, TTFT p50 **0.21 / 0.54 s**, p95 3.16 / 7.12 · + overlap: 419.3 (+2 %) / 308.6 (+6 %), p50 0.34 / 1.35, p95 4.17 / 10.13 | 🔬 without lockstep bursts there is little queueing, so plain TTFT is already low and the overlap only stretches each read (worse TTFT) while winning a little throughput. Default stays without overlap. New reference for realistic traffic | `results/ab-mix.out` |
| 54 | `--prefill 1024` again, now with `--aux-cpus` (#25 was before the CPU fix), with and without `STRATA_PF_FUSED=1` (GPU grouping, only from 1024-token chunks) | short 490.5 / 488.9, long **345.9 / 347.6** (512: 492.9 / 376.2) | ❌ 512 stays best; the fused path does not pay here | `results/ab-pf1k.out` |
| 55 | **Profile of the default** (prefill phases per stage, decode windows, nsys; `results/profile2/`, `upstream/PROFILE-next-levers.md`) | decode: GPU3 (last 12 layers + head + sampling) 98 % busy, GPU1/2 ~81 % with ~1.25 ms idle per 6.5 ms period; 616.6 rows/s at 16 slots, 670.1 at 32; +1.34 ms per extra row on GPU3. Prefill 1.2K: chunk compute 90 % of the critical path, MTP draft layer on GPU3 31 ms (6.5 %), hand-offs 16 ms; each card ~41 % busy (150 tok: 24 %); MMQ 57 % of a 512 chunk | 🔬 ranked levers: split, MTP batch, MMQ group sort, batch self-commit, multi-slot QSA | `results/profile2/` |
| 56 | Split `12,24,37` / `12,25,37` (one layer off GPU3) and `STRATA_SPLIT_MTP_BATCH=1` (draft-layer prompt K/V batched; tokens unchanged), quick A/B vs `T16-142fca` (492.9 / 376.2) | 12,24,37: **519.2 / 398.0**; 12,25,37: 517.2 / 396.0; MTP batch alone: 495.9 / 386.4 (1.2K read 484 → 449 ms). All experts stay resident, no loans | ✅ both; 12,24,37 vs 12,25,37 (+0.4 % / +0.5 %) is **within noise**, either split is fine. Reverses #30 (same split flat on 0.1.41, HTTP only, before B-tab and `--aux-cpus`) | `results/ab-lv.out` |
| 57 | **`T16-142fcb`** = 12,24,37 + MTP batch + aux + fixed-cost, back to back vs `T16-142fca`, levels 1/16 | exact (batch_test16 solo + batch dumps equal to the 12,24,36 reference). 1 user 130.9 vs 129.6 tok/s decode (draft acceptance unchanged); 16 users **522.4 vs 490.5 (+6.5 %) short, 405.3 vs 377.6 (+7.3 %) long**, TTFT p95 3.51 / 7.32 s | ✅ **new default** | `results/ab-fcb.out`, `results/auxcheck/bt16-fcb.*` |
| 58 | **Levers 3 + 4 implemented** (Strata-142 faddbb2e, 7e7e169e, 00ff1fb2; engine `strata-g3-142lv`): `STRATA_MMQ_GROUP_SORT=1` sorts the full 16-expert MMQ groups by row count, keeping the partial (stream-k tail) group; `STRATA_BATCH_SELF_COMMIT=1` lets a one-row-per-slot batch window advance GDN state itself (no commit graph). Multi-slot GDN tried and dropped (−0.8 %). Engine level, back to back vs `strata-g3-142fc` | exact: prefill_mmq_tab_test 890/890, all-rows R dumps byte-identical, batch_test16/32, overlap test incl. RESUME. Read 150 tok 152.4 → 135.3 ms, 1.2K 435.4 → 360.7 ms (−17 %); decode 16 slots 653.7 → **686.0 rows/s (+4.9 %)**, 32 slots 714.1 → 752.6 (+5.4 %) | ✅ exact; HTTP → #60 | `results/levers/` |
| 59 | **Exactness gaps from the review closed** (reference `T16-142fcb` and candidate `T16-142gsc`, `results/gaps/`) | (1) `batch_interleave_test` fixed for a pipelined batch (Strata-142 14c16bf2: no give-way required there, new step 7b = start from a slot's turn checkpoint): all 10 checks IDENTICAL, steps 4–7 run for the first time; (2) sampled batch = solo at T=0.7, seed 1234: 16/16; (3) agent config on 0.1.42 (`X16-agent-142fcb/gsc`): with split 12,24,37 CUDA2 **borrowed 306 slots** → agent configs use **12,25,37** (no loan on any card, 100 % resident, 16/16 = solo); reference = candidate in all dumps. (4) The "next turn from the slot differs" row: with a corrected reference (child solo right after the parent solo) still 3 of 6 differ (at 53/59/92 tokens), the same with all fixed-cost switches off - **with drafts suppressed (`--spec-min-p 1.0`) all 6 IDENTICAL**: a solo turn decoded in multi-row draft windows leaves a numerically different KV/GDN state than one-row windows (its tokens are equal), so a later turn can diverge. Upstream behaviour, not our patches | ✅ gaps closed; (4) is an upstream finding for the issue | `results/gaps/` |
| 60 | **Levers 3 + 4 through the HTTP server** (`ab_quick.sh`, greedy, 16 users; reply sha256 per request compared with `compare_replies.py`): one run `T16-142fcb` / + group sort (`T16-142gs`) / + both (`T16-142gsc`), then 2 ABBA pairs fcb vs gsc | single run: short 525.2 / 529.3 (+0.8 %, within noise) / **550.7 (+4.9 %)**, long 403.1 / 426.2 (+5.7 %) / **440.9 (+9.4 %)**. ABBA (means): short 511.6 → **549.1 (+7.3 %, pairs +8.1 / +6.6)**, long 397.1 → **439.2 (+10.6 %, pairs +11.1 / +10.1)**, TTFT p50 long 2.49 → 2.04 s. **All replies identical through the server (32/32 in every pair)** - the first token-level exactness check over the production path | ✅ **`T16-142gsc` new default** (engine `strata-g3-142lv`); agent default `X16-agent-142gsc` (split 12,25,37, #59) | `results/ab-lev34.out`, `results/ab-lev34c.out`, `results/sweep.requests.jsonl` |
| 61 | **Rebase onto upstream 0.1.43** (worktree `Strata-143`, branch `g3-143`, engine `strata-g3-143`): all 32 commits kept (26 clean, 3 context moves, 4 small conflicts with upstream's NaN guard #879 and DETAILS.md); llama.cpp pin unchanged (MMQ-tab hash check passes). Review found 1 real defect: `STRATA_SLOT_CKPT_MOVE` could resume from a running slot's checkpoint after upstream's NaN flush (fixed in 42740f84, reproduced with `STRATA_NAN_INJECT` before/after) | exact and **byte-identical tokens to the 0.1.42 default** in every dump: batch_test16/32, interleave (11 checks), overlap test on/off (token + RESUME), T=0.7 seed 1234, prefill_mmq_tab_test 890/890; 100 % resident, no loans, MMQ direct + group sort 48/48, self-commit on all 4 stages. NaN guard on by default (no token changed). Upstream: server timings now split `prompt_n`/`cache_n` | ✅ exact; speed → #62 | `results/rebase143/` |
| 62 | 0.1.43 vs 0.1.42 through the server (quick A/B, greedy, 16 users, each arm with its own checkout's server.py): `T16-142gsc` / `T16-143gsc` (NaN guard on, upstream default) / `T16-143gsc-noguard` (`STRATA_NAN_GUARD=0`) | short **555.3** / 537.2 (−3.3 %) / 547.8 (−1.4 %, within noise); long **437.5** / 432.6 (−1.1 %) / 439.0 (+0.3 %); all replies identical to 0.1.42 (32/32 each). The NaN guard (#879) costs ~2 % here (upstream measured < 0.2 %) | ➖ no speed gain from 0.1.43; default stays `T16-142gsc`; development continues on `g3-143` (upstream base for PRs); the guard's cost on pipelined batch windows is an upstream note | `results/ab-v143a.out`, `results/ab-v143b.out` |
| 63 | Upstream bitwise switches `STRATA_LFUSE=1` (shared-expert gate launch + KV append folded) and `STRATA_TSUM=1` on `T16-142gsc` (quick A/B, 1 run) | short 545.8 / 549.2 (+0.6 %) / 551.6 (+1.1 %), long 437.6 / 438.3 (+0.2 %) / 440.5 (+0.7 %); replies identical (32/32 each) | ➖ within noise (single run resolves ~1.2 %); re-test both in an ABBA run together with lever 5 | `results/ab-lt.out` |
| 64 | **Lever 5: multi-slot QSA attention** (`STRATA_QSA_SLOTS=1`, Strata-143 8013f726 + test fix a3ee9b06, engine `strata-g3-143l5`): block scores, top-k and attention of a batch window's QSA layers in one launch over the window's rows (one row per slot, each with its own KV); streamed K/V and kernel variants that would differ keep the per-row loop | exact: SASS of the existing kernels unchanged, new kernel parity test bitwise (4 KV formats, 1–8 rows, contexts 37–32768), batch_test16/32 + T=0.7 + interleave byte-identical to the reference dumps, overlap test token + RESUME identical; agent config (`X16-agent-143l5`, 12,25,37) 16/16 = solo, equal to the 0.1.42 agent dump, no loan. Engine: decode 16 slots 681 → **725 rows/s (+6.3 %)**, 32 slots 750 → 802 (+6.9 %); QSA category 0.90 → 0.52 ms per stage window. HTTP, 2 ABBA pairs on 0.1.43 (guard off): short 548.8 → **577.9 (+5.3 %)**, long 438.4 → **456.1 (+4.1 %)**, replies identical (32/32 each pair). + `TSUM`/`LFUSE`: 581.2 / 459.5 (within noise), replies identical | ✅ **`T16-143l5` new default** (0.1.43 + lever 5, `STRATA_NAN_GUARD=0`, server Strata-143); agent: `X16-agent-143l5` | `results/levers/qsa-slots/`, `results/ab-l5*.out`, `results/gaps/agent143/` |

## Analyses (no GPU)

- `upstream/ANALYSIS-batching-levers.md`: 4 engine levers (overlapped admission, MTP in the batch, window overhead,
  quick wins) with effort estimates and cross-checks.
- `upstream/DESIGN-overlap.md`: complete design of overlapped admission (6 milestones).
- `upstream/ANALYSIS-short-prompt-moe.md`: short-prompt prefill, options A (decode kernel for ≤ 8 rows),
  B (MMQ directly on the cache slots), C (fused kernels).

## Open / next steps

- Dashboard (user wish 2026-10-12): extend `dashboard/` into one cross-platform monitor that also runs without the Strata
  server: Strata's own monitor features (the live "answering" progress bar per request, slots), multi-GPU view for 4 cards
  incl. PCIe bus throughput per card, always showing a UI.

Test protocol from #49 on: `ab_quick.sh` (one server start per arm, 16 users, greedy, short + 1.2K, ~4–5 min per arm);
three repetitions only to confirm a new default.

- Overlapped admission: fix the p95 tail (overlap only when no other admission is queued; or read queued prompts
  together), knob arms (M4: PRIO, SLEEP_US, MTP_PRIO), docs (M6).
- Rebase onto upstream 0.1.42 (+235 commits); the Windows side measured `STRATA_ROUTE_TAIL_SKIP=7` +12-13 % decode
  there (upstream keeps it off with every expert in VRAM and with `--batch`; it changes the output).
- Slot-resume mismatch after a batch-decoded reply (see the row after #40): likely a reference artefact, solo-only re-check pending.
- Parked: dynamic switching pipeline ↔ expert parallel (see the memory/ideas note in the README section).

## Appendix: all measurement runs (generated automatically from the JSONL files)

### `results/sweep.jsonl`

| Label | Users: total tok/s (TTFT p50) |
|---|---|
| `vllm-w4a16-ep4-nomtp` | 1: 72.2 (0.99 s), 2: 121.6 (0.57 s), 4: 210.3 (1.46 s), 1: 65.0 (1.34 s), 2: 115.1 (0.58 s), 4: 197.2 (1.45 s), 8: 269.3 (2.67 s), 16: 375.8 (2.42 s), 32: 384.2 (7.2 s), 48: 418.4 (29.54 s), 60: 400.1 (50.93 s), 96: 414.2 (87.52 s) |
| `vllm-w4a16-ep4-nopc` | 1: 74.9 (0.37 s), 8: 255.9 (1.81 s), 16: 374.7 (2.26 s), 24: 419.8 (3.4 s), 32: 418.1 (6.62 s), 48: 452.2 (27.14 s), 64: 442.0 (48.21 s) |
| `strata-A-solo` | 1: 91.5 (1.5 s), 2: 87.1 (7.37 s), 4: 87.4 (18.94 s), 8: 87.6 (41.64 s), 16: 89.7 (85.95 s) |
| `strata-B-batch8` | 1: 85.9 (1.69 s), 2: 106.2 (2.51 s), 4: 147.9 (3.32 s), 8: 177.0 (5.03 s), 12: 167.8 (10.84 s), 16: 170.8 (11.74 s), 32: 174.4 (35.66 s), 64: 175.8 (65.32 s) |
| `strata-C-batch16` | 1: 84.8 (1.7 s), 8: 171.0 (5.94 s), 16: 198.4 (9.26 s), 24: 193.6 (19.38 s), 32: 192.8 (20.26 s), 64: 194.9 (46.04 s) |
| `strata-D-batch32` | 16: 179.9 (10.61 s), 32: 191.4 (8.41 s), 48: 191.5 (21.26 s), 64: 192.1 (27.28 s), 96: 191.9 (37.75 s) |
| `strata-E1-b16-pf512-share1` | 16: 246.3 (4.58 s), 32: 240.8 (15.1 s) |
| `strata-E2-b16-pf256-share1` | 16: 228.7 (5.42 s), 32: 226.5 (17.78 s) |
| `strata-E3-b16-prefillhelp` | 16: 193.6 (7.17 s), 32: 193.4 (20.26 s) |
| `strata-F1-b16-pf1024-share1` | 16: 226.4 (6.65 s), 32: 221.9 (12.89 s) |
| `strata-F2-b16-pf512-share2` | 16: 246.0 (4.58 s), 32: 250.6 (12.25 s) |
| `strata-F3-b16-pf768-share1` | 16: 245.3 (4.56 s), 32: 238.6 (15.62 s) |
| `strata-G16-long` | 1: 95.6 (1.15 s), 2: 117.6 (1.12 s), 4: 164.4 (2.28 s), 8: 209.4 (4.52 s), 16: 248.4 (4.57 s), 24: 237.7 (11.02 s), 32: 233.9 (11.07 s), 64: 246.0 (46.48 s) |
| `strata-G16-short` | 1: 103.7 (0.53 s), 8: 281.0 (2.0 s), 16: 357.5 (2.6 s), 24: 364.6 (6.55 s), 32: 364.5 (20.27 s), 64: 354.3 (29.53 s) |
| `strata-G32-short` | 16: 318.8 (2.73 s), 32: 342.4 (2.98 s), 48: 330.0 (7.63 s), 64: 340.9 (12.86 s), 96: 344.7 (14.22 s) |
| `strata-H12-short` | 12: 361.1 (1.6 s), 18: 335.9 (2.62 s) |
| `strata-H16-short` | 16: 347.9 (2.07 s), 24: 358.1 (6.31 s) |
| `strata-H20-short` | 20: 335.9 (2.97 s), 30: 326.6 (6.51 s) |
| `strata-H24-short` | 24: 331.2 (2.63 s), 36: 328.3 (7.43 s) |
| `strata-H28-short` | 28: 328.0 (3.16 s), 42: 328.3 (8.21 s) |
| `strata-A-solo-mtp` | 1: 87.2 (1.5 s) |
| `vllm031-27b-fp8-p1000` | 1: 59.1 (0.76 s), 4: 130.7 (2.62 s), 12: 241.0 (4.3 s), 16: 283.1 (4.34 s), 32: 370.6 (5.13 s), 64: 409.1 (9.2 s) |
| `vllm031-27b-fp8-p100` | 1: 63.5 (0.13 s), 4: 203.7 (0.46 s), 12: 382.0 (1.24 s), 16: 484.9 (1.25 s), 32: 705.2 (2.25 s), 64: 838.1 (3.35 s) |
| `vllm031-flashnext-pc-p1000` | 1: 75.0 (0.38 s), 8: 238.2 (1.85 s), 12: 365.9 (1.9 s), 16: 385.8 (1.95 s), 24: 416.0 (4.16 s), 32: 397.5 (22.27 s), 48: 395.0 (33.44 s), 64: 398.2 (49.53 s) |
| `vllm031-flashnext-pc-p100` | 1: 78.0 (0.15 s), 8: 413.5 (0.56 s), 12: 499.3 (0.61 s), 16: 621.8 (0.68 s), 24: 590.8 (0.88 s), 32: 706.4 (0.95 s), 48: 683.2 (16.21 s), 64: 710.0 (31.46 s) |
| `strata-K-q4kxl-solo` | 1: 70.0 (2.26 s) |
| `strata-K-q4kxl-b12-short` | 12: 241.0 (4.12 s), 24: 233.7 (11.02 s) |
| `strata-K-q4kxl-b12-long` | 12: 149.8 (9.32 s) |
| `strata-T3-solo` | 1: 95.8 (1.1 s) |
| `strata-T3-b12-short` | 12: 265.9 (2.48 s), 18: 253.0 (6.26 s) |
| `strata-M-nomtp` | 1: 64.6 (1.42 s) |
| `strata-U-b12-users` | 24: 347.3 (4.17 s), 28: 346.0 (5.06 s), 32: 357.3 (9.11 s), 36: 358.2 (15.81 s), 40: 351.2 (17.2 s) |
| `strata-P-pf512-pf1600` | 1: 17.1 (1.56 s) |
| `strata-P-pf512-pf6500` | 1: 7.3 (4.06 s) |
| `strata-P-pf512-pf13000` | 1: 4.2 (7.28 s) |
| `strata-P-pf512-pf24000` | 1: 2.4 (12.73 s) |
| `strata-P-auto-pf1600` | 1: 11.6 (2.44 s) |
| `strata-P-auto-pf6500` | 1: 5.0 (6.16 s) |
| `strata-P-auto-pf13000` | 1: 3.7 (8.35 s) |
| `strata-P-auto-pf24000` | 1: 2.5 (12.45 s) |
| `strata-P-auto32k-pf1600` | 1: 11.5 (2.46 s) |
| `strata-P-auto32k-pf6500` | 1: 4.7 (6.46 s) |
| `strata-P-auto32k-pf13000` | 1: 2.7 (11.36 s) |
| `strata-P-auto32k-pf24000` | 1: 1.5 (20.84 s) |
| `vllm031-P-pf1600` | 1: 33.6 (0.56 s) |
| `vllm031-P-pf6500` | 1: 12.2 (2.22 s) |
| `vllm031-P-pf13000` | 1: 6.7 (4.41 s) |
| `vllm031-P-pf24000` | 1: 3.8 (8.13 s) |
| `vllm031-flashnext-mtp2-s8-p1000` | 1: 115.8 (0.66 s), 4: 268.9 (0.81 s), 8: 224.9 (1.53 s), 12: 234.6 (14.15 s) |
| `vllm031-flashnext-mtp2-s8-p100` | 1: 132.7 (0.15 s), 4: 346.4 (0.3 s), 8: 490.3 (0.3 s), 12: 486.1 (0.91 s) |
| `strata-eval-iq4xs` | 1: 10.8 (1.33 s) |
| `vllm031-razor65-mtp2-s12-p1000` | 1: 102.7 (0.36 s), 4: 263.9 (0.58 s), 8: 345.7 (0.76 s), 12: 411.7 (1.11 s), 24: 416.2 (15.04 s) |
| `vllm031-razor65-mtp2-s12-p100` | 1: 110.2 (0.15 s), 4: 304.1 (0.22 s), 8: 445.5 (0.37 s), 12: 536.1 (0.29 s), 24: 532.4 (11.49 s) |
| `strata-Q2-b16-groups2-short` | 16: 266.3 (1.59 s), 24: 271.6 (5.13 s) |
| `strata-Q2-b16-groups2-long` | 16: 199.5 (3.45 s) |
| `strata-Q2-b12-groups2-short` | 12: 269.8 (0.93 s), 24: 263.7 (2.91 s) |
| `strata-Q3-b12-share4-short` | 12: 366.4 (1.59 s), 24: 374.9 (4.13 s) |
| `strata-Q3-b12-share4-long` | 12: 258.9 (4.17 s) |
| `strata-Q1-dp2-short` | 1: 74.4 (1.11 s), 12: 152.5 (7.23 s), 24: 174.2 (7.17 s), 32: 182.3 (8.83 s) |
| `strata-Q1-dp2-long` | 16: 99.5 (25.39 s) |
| `strata-T12-trim-short` | 1: 108.7 (0.4 s), 12: 344.8 (2.03 s), 24: 366.2 (3.39 s) |
| `strata-T12-trim-long` | 12: 275.6 (4.35 s) |
| `strata-T16-trim-short` | 1: 110.6 (0.4 s), 16: 404.7 (2.32 s), 32: 389.1 (4.83 s) |
| `strata-T16-trim-long` | 16: 288.1 (5.37 s) |
| `strata-T20-trim-short` | 20: 409.4 (2.05 s) |
| `strata-T20-trim-long` | 20: 288.5 (5.35 s) |
| `strata-T24-trim-short` | 24: 416.5 (2.85 s) |
| `strata-T24-trim-long` | 24: 292.8 (5.34 s) |
| `strata-T16-pf1024-long` | 16: 280.7 (4.47 s) |
| `strata-T16-pf1024-short` | 16: 406.3 (2.33 s) |
| `strata-T16-pf2048-long` | 16: 290.5 (5.3 s) |
| `strata-T16-pf2048-short` | 16: 403.4 (2.15 s) |
| `strata-T16-trim-split-mt-warm` | 1: 11.6 (1.19 s) |
| `strata-X16-agent-trim-262k-mt-warm` | 1: 9.9 (1.41 s) |
| `strata-T16-direct-short` | 16: 405.3 (1.97 s), 24: 391.8 (3.37 s) |
| `strata-T16-direct-long` | 16: 294.5 (4.31 s) |
| `strata-X16-direct-mt-warm` | 1: 10.2 (1.43 s) |
| `strata-T16-split12-24-37-short` | 16: 399.2 (2.25 s) |
| `strata-T16-split12-24-37-long` | 16: 289.3 (4.47 s) |
| `strata-T16-split13-25-37-short` | 16: 397.3 (2.22 s) |
| `strata-T16-split13-25-37-long` | 16: 281.6 (4.27 s) |
| `strata-T16-pleram-short` | 16: 416.8 (2.03 s) |
| `strata-T16-pleram-long` | 16: 296.5 (4.32 s) |
| `strata-S16-spike-short` | 16: 412.9 (0.57 s) |
| `strata-S16-spike-long` | 16: 293.5 (1.18 s) |
| `strata-P16-timing` | 1: 6.1 (1.23 s) |
| `strata-P16-fused` | 1: 6.0 (1.23 s) |

### `results/multiturn.jsonl`

| Label | Users | tok/s | Follow-up turn TTFT p50 |
|---|---|---|---|
| `strata-T16-trim-split-mt-u8` | 8 | 49.2 | 7.05 s |
| `strata-T16-trim-split-mt-u16` | 16 | 42.8 | 10.4 s |
| `strata-X16-agent-trim-262k-mt-u8` | 8 | 57.1 | 4.76 s |
| `strata-X16-agent-trim-262k-mt-u16` | 16 | 72.8 | 4.95 s |
| `vllm031-full-offload20-mt-u8` | 8 | 89.2 | 6.37 s |
| `vllm031-full-offload20-mt-u16` | 16 | 76.7 | 32.18 s |
| `strata-X16-direct-mt-u16` | 16 | 62.3 | 4.47 s |

### `eval/eval_results.jsonl`

| Label | MMLU-Pro | GSM8K |
|---|---|---|
| `vllm-full-w4a16-fp8ple` | 70.4% | 94.7% |
| `strata-ud-iq4xs` | 77.9% | 94.0% |
| `vllm-qwen3.8-27b-fp8` | 72.1% | 95.3% |
| `vllm-razor65-nvfp4` | 62.5% | 92.7% |
