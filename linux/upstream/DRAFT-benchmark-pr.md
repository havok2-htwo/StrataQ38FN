# [Benchmark] PR draft: 4x RTX 3090, stock v0.2.12 vs v0.2.12 + patch series

Nothing has been posted. This is a draft for review.

## Title

```
[Benchmark] 4x RTX 3090 / Qwen3.8-Flash-Next / UD-IQ4_XS / strata-qwen38-thinking-v1-draft.1
```

`.github/workflows/title-area.yml` requires the title to start with `[Benchmark] `. The rest follows the `<GPU> / <model> / <quant> / <protocol-id>` part of the title that BENCHMARK_PROTOCOL.md suggests. The doc's own `benchmark: ` prefix would fail that check. You could optionally add ` + multi-user (custom)` at the end.

---

## PR body

**The title must start with its area: [Engine], [GUI], [TUI], [Benchmark], [Docs] or [Other], for example "[Engine] fix decode stop on long prompts" or "[GUI] keep the model list after a tab switch".**

## Problem and result

`bench/results` has no 4x RTX 3090 multi-user report yet. TEST_REQUESTS.md lists "Three or four NVIDIA GPUs" as a wanted test (#880). This PR adds results only. It has no engine change, makes no speed claim and proposes nothing for the README headlines.

Three arms were measured on one machine on 2026-11-08, interleaved, each on a freshly loaded server. All arms use the same model files, the same expert profile (same SHA-256) and the same chat template (same SHA-256):

| Arm | What it is |
| --- | --- |
| **S** | Stock v0.2.12 (`146df3c4`) as `setup` writes it for this box: 128K context, `--batch 3`, `--prefill auto`, `--layer-split auto`, int8 KV, MTP, vision off. |
| **C** | The same stock binary and stock server, with our production args/env: 16 slots, 262K context, `--prefill 512`, `--layer-split 12,25,37`, resident experts, conversation and prompt cache, the vision encoder loaded on the CPU, the stock sm_86 default table. Bit-changing defaults are pinned off. |
| **PT** | v0.2.12 plus an opt-in patch series by havok2-htwo (branch head `a099d295`, 56 commits on `146df3c4`). It uses the same args as C plus the series' engine switches, and runs with the series' server. |

Run order: C, PT, S, C-r2, PT-r2, S-r2.

**Standard protocol** (`strata bench run`, required stage plus `--optional-concurrency`, `--track custom`). All six runs are `complete` and score 12/12 on sanity. The table shows medians from run 1. Run 2 is within 2 % on every value shown, except S's c4 burst rounds 2 and 3 (184 / 184 tok/s in run 2).

| Case | Metric | S | C | PT |
| --- | --- | ---: | ---: | ---: |
| short | decode tok/s | 127.4 | 136.1 | 154.3 |
| short | TTFT s | 0.49 | 0.45 | 0.23 |
| prompt-4k | prompt tok/s | 1,921 | 4,998 | 4,999 |
| prompt-4k | decode tok/s | 128.3 | 138.4 | 155.2 |
| prompt-16k | prompt tok/s | 3,219 | 5,774 | 6,009 |
| prompt-16k | decode tok/s | 127.9 | 117.4 | 124.6 |
| prompt-16k | TTFT s | 5.12 | 2.86 | 2.75 |
| short-burst (c4) | aggregate tok/s, 3 rounds | 198 / 202 / 196 | 279 / 283 / 268 | 282 / 296 / 278 |
| short-burst (c4) | TTFT s | 0.75 | 0.40 | 0.49 |

**Multi-user, custom harness.** Greedy (temperature 0), with `sampling.presence_penalty: 0` in every arm's config. Values are the mean of 2 runs per arm, or the range where both runs are shown.

| Measure | S | C | PT |
| --- | ---: | ---: | ---: |
| 16 users, short (~150-token prompts), aggregate tok/s | 220.1 | 539.0 | 527.2 |
| 16 users, long (~1.2K-token prompts), aggregate tok/s | 174.7 | 400.8 | 405.1 |
| 8 users, short, aggregate tok/s | 216.6 | 412.8 | 400.1 |
| Per-user decode p50 at 1 / 2 / 4 users | 142 / 86 / 82 | 131 / 87 / 75 | 145 / 128 / 90 |
| TTFT short at 16 users, p50 / p95 (s) | 14.6-20.5 / 44.5-51.5 | 0.99 / 2.95-3.08 | 0.37-0.38 / 4.23-4.32 |
| TTFT long at 16 users, p50 / p95 (s) | 18.7-18.8 / 65.8-69.4 | 2.54 / 7.76-7.78 | 0.79 / 9.39-9.42 |
| 16 users x 3 turns (8K doc): wall / turn-1 TTFT p50 / follow-up TTFT p50 (s) | 266-281 / 46.6-67.1 / 23.6-48.8 | 97.6-97.7 / 17.4-17.6 / 2.03-2.08 | 90.8-91.3 / 26.4-27.9 / 2.74-2.95 |
| Cold 32K joiner beside 4 streams (`tools/piped_read_join_bench.py`): worst stream gap / joiner TTFT (s) | queue-bound, not compared | 5.80 / 5.80 | 0.10-0.11 / 8.2-8.3 |
| `tools/strata_bench.py --clients 16 --long`, 16 clients, tok/s, median (range) of 3 runs, first server start only | 160.7 (158.1-161.5) | 275.3 (247.8-275.3) | 266.9 (253.3-267.7) |

**How to read this:**
- **Most of the 16-user gain over S comes from configuration, not engine code.** The configuration changes are 16 slots instead of 3, `--prefill 512` and the caches. With the stock engine, C already gets +145 % (short) and +130 % (long) over S.
- **What the patch series itself adds (PT vs C):**
  - per-user decode is +10 % at 1 user, +47 % at 2 users and +19 % at 4 users;
  - single-request decode in the standard protocol is +12-14 % on short and 4K, and +6-7 % on 16K;
  - a cold 32K joiner no longer freezes the running streams (gap 0.1 s instead of 5.8 s);
  - multi-turn wall time is -7 %.
- **Where PT is worse than C** (listed again under Limits):
  - the 16-user short aggregate is -2.2 %, and the 8-user short aggregate is -3.1 %;
  - TTFT p95 at 16 users is worse with short prompts (4.2-4.3 s instead of 3.0-3.1 s) and with long prompts (9.4 s instead of 7.8 s);
  - the 32K joiner's own TTFT is 8.2 s instead of 5.8 s;
  - c4 burst TTFT is 0.49 s instead of 0.40 s;
  - in the multi-turn test, turn-1 TTFT p50 is 26-28 s instead of 17-18 s, and follow-up TTFT p50 is 2.7-3.0 s instead of 2.0-2.1 s;
  - `strata_bench.py` at 16 clients is -3 %, inside the overlap of the two arms' run ranges.
- **At 16K, single-request decode is lower than S in both 16-slot configurations** (C -8 %, PT -3 %).

## Related discussion

- #880 / TEST_REQUESTS.md: multi-GPU NVIDIA reports.
- #3354, #3295 and #2940 (closed): a request that does not send `presence_penalty`, including a temperature-0 request, gets the model card's 1.5. That turns off MTP drafting and the pipelined paths.
  - All three arms were measured with `sampling.presence_penalty: 0` in the config, so neither stock nor PT is penalised. This covers our harness, `strata_bench.py` and `piped_read_join_bench.py`.
  - `strata bench run` sends 0 explicitly and is unaffected.
  - On stock as shipped, clients that send no sampling get the penalty. That case is not measured here.
- Our earlier multi-user series (#2364) was ported in v0.1.45 (merge `df44bcb7`). Its benchmark numbers never got a `bench/results` folder. This report adds current numbers for that kind of setup.
- Scope agreement: none (CONTRIBUTING.md lets benchmark reports go straight to a PR). The patch series is not part of this PR. Any engine PR from it will come separately, with an issue first where CONTRIBUTING.md asks for one.

## Approach

- **One fresh server per arm, with the standard runner first.** Each arm's server started fresh. Its first workload was `strata bench run --profile <arm> --no-start --optional-concurrency --track custom`. After that it ran, in order:
  - `tools/strata_bench.py --runs 3 --clients 16 --long` (first start of each arm only);
  - our `bench_concurrency.py`: 1/2/4/8/16 users, short and long prompts, 2 requests x 512 tokens per user;
  - `bench_multiturn.py`: 16 users x 3 turns, 8K document plus 3K per turn, 400 output tokens;
  - upstream's `tools/piped_read_join_bench.py --scenarios cold32k`.
- **Two stock baselines:**
  - **S** is what a user gets from `setup`.
  - **C** keeps the stock engine and server but uses PT's args. It isolates what the patch series adds.
  - Stock v0.2.12 already enables six switches from our earlier series by default when the server runs batch slots: CKPT_POOL, SLOT_CKPT_MOVE, SLOT_COPY_DIRECT, SLOT_COPY_BATCHSYNC, BATCH_SELF_COMMIT and QSA_SLOTS. C sets them explicitly and S gets them from the server default (`--batch 3`), so the PT-C delta shows only the part that is not yet upstream.
- **Env differences between PT and C** (full lists in each run's `results.json`, under `configuration.experimental_options`):
  - attention: `ATTN_DAILY=0` + `QSA_SLOTS_STREAM=1`;
  - `CONV_MMAP`, `CONV_TIER_*`, `EARLY_SAMPLE`, `GATE_EVICT`, `GIVEBACK_CHECKS`, `GR_DOWN_PF=2`, `MROPE_SLOTS`, `MSG_CKPT_TAIL=8192`;
  - `PARK_BORROW`, `PARK_CAPTURE`, `PIPE_OVERLAP(+_KVS)`, `READ_YIELD`, `SERVE_STOP_BOUND`, `SLOT_CHAIN_REUSE`, `SLOT_PARK`, `SOLO_PIPELINE`, `SPEC_LANES=4`.
  - The servers differ as well. Both set `ADMIT_SHORT_FIRST=8192`. PT's server adds `ADMIT_BY_NEW`, `YIELD_SHORT=2048`, `YIELDS_MAX=4` and `PREFIX_DIAG` (diagnostics only).
  - An earlier window on the same machine tried PT with the table's DAILY attention instead of slots-stream. It did not close the 16-user short gap (519 vs 529 tok/s). That window's numbers are not mixed into the tables above.
- **Hardware:**
  - 4x RTX 3090 24 GB, 320 W limit each, layers split over the 4 cards (S: `auto`; C/PT: `12,25,37`);
  - PCIe as reported under load: cards 0-1 gen3 x16, cards 2-3 gen4 x8; no NVLink;
  - Threadripper 3990X, 251.5 GiB RAM, model files on an NVMe RAID 0 (3x Crucial P310).
- **Software:** Ubuntu 24.04.5, kernel 7.0.0-38, driver 580.196.07, CUDA 13.0. Both binaries are source builds with CUDA 13.0, sm_86, `STRATA_MMQ_TAB=ON`, `NATIVE_EXPERTS=ON`, `Q6K_EXPERTS=ON` and `MMQ_KQUANTS=ON`. The engine reports version 0.1.58 in all arms.
- **Model files:**

  | File | SHA-256 |
  | --- | --- |
  | `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | `5ce89370720f8bf9...` |
  | `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | `577a38a2392b40ca...` |
  | `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | `d4634e6d84f0ebb0...` |
  | `expert-profile.bin` (same in all arms) | `8f59b4aa8873209d...` |

  The full hashes are in `results.json`.

## Validation

**Passed:**
- **`strata bench run`: all six runs are `complete`, with 12/12 on sanity each.**
  - S: Strata `146df3c4`, runs `e32ea56b` and `8be2bf4a`;
  - C: `146df3c4`, runs `eb4f9649` and `be712cdc`;
  - PT: `a099d295`, runs `d8cef8bb` and `45387cb6`.
  - No request failed. No request was flagged for unexpected reuse or for a length outside tolerance. Performance requests capped at 512 tokens stay in the summaries, as the protocol says. The protocol ID and the corpus SHA-256 `e75eff8b...` match.
- **A freshly loaded engine every time:** every runner README says "the server had answered no request before this run" (`server_requests_before_run: 0`).
- **Greedy reply identity:** we compared 62 short greedy replies of 512 tokens each, one per request at 1/2/4/8/16 users with 2 requests each.
  - Each of C, C-r2, PT and PT-r2 matches a PT reference run from an earlier window, 62/62 by output SHA-256. So C and PT match each other.
  - S differs (0/62). It runs another configuration (3 slots, 128K, auto split), so identical replies are not expected.
- **An earlier window of PT against our own previous build:** 62/62 identical HTTP replies, and `batch_test` 16/16 in both solo and batch mode.
- **Run-to-run spread:** run 2 is within 2 % of run 1 on every value in the standard-protocol table except S's c4 burst rounds (see above). On the 16-user short aggregate the spread is 0.6 % for C and 0.1 % for PT.

**Failed:**
- None observed in the benchmark runs.
- Outside this report:
  - Stock C fails our vision check (NO-GO: the long image request in a busy batch differs from the same request alone, 3 of 4). This matches the batch M-RoPE bug from #1242, which is not fixed in v0.2.12. PT passes the check with the per-slot fix. No vision workload is part of these numbers.
  - In an earlier window, our `batch_overlap` test failed on PT (`ERR BGEN: no such free slot`). This is still open.

**Not run:**
- `batch_test` engine identity and the GPU ctest in this window: switched off for time. Here, the only identity evidence for PT vs C is the HTTP greedy reply comparison.
- HIP, SYCL and Metal builds: no hardware.
- Single-GPU runs.
- A soak of an hour or more (#848).
- Multi-user runs with T 1.0 sampling: the standard protocol covers T 1.0 only at concurrency 1 and 4.
- Stock as shipped with a client that sends no sampling, which gets the penalty.
- A third run per arm for the custom harness. COMMUNITY_BENCHMARKS.md says to try at least three; there are 2 per arm.

## Limits

- **Where PT loses against stock with the same config (C):**
  - The 16-user short aggregate is -2.2 % (539.0 vs 527.2), and the 8-user short aggregate is -3.1 % (412.8 vs 400.1). PT's steady-window aggregate at 16 users is higher (599 vs 573 tok/s), but admission puts the late requests behind: short TTFT p95 is 4.2-4.3 s instead of 3.0-3.1 s, and long TTFT p95 is 9.4 s instead of 7.8 s.
  - The cold 32K joiner's own TTFT is 8.2 s instead of 5.8 s. PT keeps the 4 running streams going (gap 0.1 s instead of 5.8 s), and the joiner pays for it.
  - c4 burst TTFT is 0.49 s instead of 0.40 s.
  - Multi-turn TTFT p50 is 26-28 s instead of 17-18 s on turn 1, and 2.7-3.0 s instead of 2.0-2.1 s on follow-ups. Total wall time is lower (91 s vs 98 s).
  - `strata_bench.py` at 16 clients is -3 %, inside the overlap of the run ranges.
- **No end-to-end speed claim at 16 users:** over two interleaved pairs, PT vs C is -2.2 % (short) and +1.1 % (long). The gains are in per-user decode at 1-4 users, in single-request decode, and in stream continuity during a join.
- **The large S-vs-PT ratios at 16 users come from configuration:** 16 slots, prefill 512 and the caches. S queues 16 clients on 3 slots. These ratios do not measure the engine.
- **Memory:** PT leaves less VRAM headroom: the lowest free VRAM was 810 MiB on card 3, against a minimum of 1,372 MiB for C. Peak host memory of the server's scope is 90-101 GiB, against 82-93 GiB for C.
- **One machine, one model and quant, one day.** The protocol is still a draft (`draft.1`). The model-card, tokenizer, source-model and quant-repo revisions are `null`, each with a reason.
- **Prompt reuse:** the conversation and prompt caches are on in C and PT. In the standard runs the runner flagged no unexpected reuse, so it excluded no request.

---

## Files to add

The standard runs follow step 2 of "Submit and review" in BENCHMARK_PROTOCOL.md: one folder per run, holding `README.md`, `results.json`, the sanitized config, engine logs and `outputs.jsonl`. The folder names use the runner's own pattern, `<UTC-date>-standard-<hardware>-<last 8 characters of the run id>`, with a short hardware label instead of the runner's long per-GPU slug.

```
bench/results/2026-11-08-standard-4x-rtx-3090-e32ea56b/   # S
bench/results/2026-11-08-standard-4x-rtx-3090-8be2bf4a/   # S-r2
bench/results/2026-11-08-standard-4x-rtx-3090-eb4f9649/   # C
bench/results/2026-11-08-standard-4x-rtx-3090-be712cdc/   # C-r2
bench/results/2026-11-08-standard-4x-rtx-3090-d8cef8bb/   # PT
bench/results/2026-11-08-standard-4x-rtx-3090-45387cb6/   # PT-r2
```

Each folder gets these files (source: `linux/results/w212d/stdbench-<arm>/`):

| Destination | Source / notes |
| --- | --- |
| `README.md` | the runner's README |
| `results.json` | after redaction (see below) |
| `outputs.jsonl` | |
| `config.json` | the effective engine options |
| `provenance/chat_template.jinja` | |
| `provenance/corpus-build.json` | |
| `provenance/corpus-manifest.sha256` | |
| `engine.log` | new: sanitized from `w212d/engine-<arm>.log`, cut from engine boot to the end of the standard run (the server's first workload) |
| `provenance/launch-config.json` | new: sanitized `w212d/cfg/run-<arm>.json`, which records the env and the server env |

The custom-track data goes into one community-format folder ("Submit a report" in COMMUNITY_BENCHMARKS.md). It is never relabelled as standard:

```
bench/results/2026-11-08-community-4x-rtx-3090-multiuser/
  README.md                         # community template: arms, method, the tables above, limits, presence_penalty 0
  strata_bench/S/README.md, results.json   # from w212d/community-S/2026-11-08-community-4x-rtx-3090-.../
  strata_bench/C/README.md, results.json
  strata_bench/PT/README.md, results.json
  data/sweep.jsonl                  # bench_concurrency, 12 labels w212-{S,C,PT}{,-r2}-{short,long}
  data/{S,C,PT}{,-r2}-{short,long}.txt
  data/multiturn.jsonl, data/{S,C,PT}{,-r2}-mt.txt
  data/join-{S,C,PT}{,-r2}.json     # piped_read_join_bench cold32k
  data/{S,C,PT}{,-r2}-ident.txt, data/ref.requests.jsonl, data/cmp.requests.jsonl
  data/vram-{S,C,PT}{,-r2}.csv, data/status-{S,C,PT}{,-r2}.json
  scripts/bench_concurrency.py, scripts/bench_multiturn.py, scripts/compare_replies.py
  FIELDS.md                         # fields and units of data/ and scripts/
```

Leave out:
- `summary.txt` and `window.log`: they contain production host lines and paths, so put a trimmed table in the README instead;
- `bin/`, `metrics-*.txt`, `server-*.out`, `vision-*.json` and `prod-before/`;
- anything from the model or pack folders.

---

## Before posting (not part of the PR body)

1. **Redaction.** Upstream's tools replace home folders, but `/media/raid0/...` is not a home folder, so they left it in:
   - `run.command` in all six `results.json` (`--profile`, `--out`) holds absolute `/media/raid0/StrataQ38FN/...` paths;
   - `strata_bench/C` and `strata_bench/PT` `results.json` hold 3 `/media/raid0` paths each;
   - the engine logs and the launch configs (`T212-*.json` / `cfg/run-*.json`) hold `/media/raid0/...` paths for the model, packs, MTP, `expert-profile.bin`, the log and the tier dir;
   - `config.json` (and `results.json` → `extensions.other_engine_options`) shows `--conversation-cache-min-tokens: <redacted>`, because the runner treated "tokens" as a secret. The real value is `1024`: restore it or explain it in the README;
   - `--track-notes` contains internal labels (`w212d`, `P9-P11`). Decide whether to keep them.
2. **Re-validate after editing.** After redacting:
   - add the `engine_log` entry, and the launch config as `config` or `provenance`, to `results.json` → `artifacts`;
   - recompute their SHA-256 values;
   - run `strata bench report <dir>` on each folder;
   - in the PR, state that the only post-run edits were path redaction and the added artifact entries, and list the original `results.json` SHA-256 values.
3. **Expert profile.** `expert-profile.bin` is hashed under `model.files`, but `model.custom_pack` is `null` ("none declared"). Either say in each README that this is the pack's expert profile, the same in all arms, or note that `--custom-pack` should have been passed.
4. **Patch branch.**
   - Push the `a099d295` branch to the havok2-htwo fork and link it. `Strata-212p` has only the upstream `origin` remote configured, and its HEAD has since moved on to `e55fa457`.
   - Add the full SHA-256 of the two engine binaries (`strata-212-stocktab-f83fd816...` and `strata-212p-74702164...`).
5. **Upstream queue.** No community [Benchmark] PR has landed since the 0.2.0-beta reset; #5428, #5463, #5396 and others are still open. Expect the maintainer to port the folders rather than merge the PR.
