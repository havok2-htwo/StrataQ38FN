# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `THIRD_PARTY_NOTICES.md`: MIT notices of Strata and llama.cpp/ggml (the engine patches contain their code), dataset
  licenses of the eval set. The rebased patch series carries the ggml notice in `moe_mmq_tab.cuh` itself.

### Measured
- Windows, engine 0.1.40.1 vs 0.1.42 on UD-IQ4_XS (series U, `series_u.sh`, `batch_bench.py --suite engine`, one
  request at a time, reference run before and after): decode **+12% prose (127 → 142 tok/s), +13% code
  (151 → 172 tok/s)**, all of it from `STRATA_ROUTE_TAIL_SKIP=7` (with the new defaults off, 0.1.42 equals 0.1.40.1).
  The measured PCIe share settles at 0.55 again (no change); the CPU prompt share gives -4% time on ~500-token
  prompts, nothing from 1K. `--prefill 512` on top: 1K prompts -9% time, 10K -1%, 3 GiB more commit free,
  prose decode -3% (137 instead of 142); longer prompts (>10K) not measured yet. Engine in worktree `Strata-142`.

### Planned
- Measurement series I (`series_i.sh`, see `HEBEL-Kandidaten-XS.md`): GPU kernel variants, CPU pool, PCIe copy mode,
  prefill chunk 16K/32K, drafting, learned expert profile, server options.
- Fixed swap file, then `--max-context 262144` and `--prompt-cache-every 8192 --prompt-cache 12`.

## [0.9.0] – 2026-10-11

Linux (4× RTX 3090).

### Changed
- Engine rebased onto upstream Strata 0.1.42 with all four patch series (`linux/patches/v0.1.42/`), exact. New
  default `configs/T16-142.json` (`start-strata.sh`): 16 users 467 tok/s short / 342 tok/s 1.2K (was 441 / 327),
  TTFT p50 0.96 / 2.60 s (was 1.74 / 4.13 s).
- Test protocol: `linux/ab_quick.sh` (one run per arm, 16 users, greedy, ~5 min) instead of 3-repetition series.

### Added
- `configs/T24-142.json` for more than 16 concurrent users (TTFT p95 −68 % at 24 users).
- `STRATA_PIPE_OVERLAP_MAX_QUEUE` and the server's `waiting=N` (measured, no gain, kept off).

## [0.8.0] – 2026-10-11

Linux (4× RTX 3090).

### Added
- Engine patch series 4 "overlapped admission" (`linux/patches/overlap/0001-0005`, env `STRATA_PIPE_OVERLAP=1`,
  opt-in, config `linux/configs/T16-overlap.json`): a new request's prompt is read while the other slots keep
  decoding. Exact (token- and resume-identical to the reference engine). 16 users: median TTFT 1.74 → 0.47 s (short)
  and 4.13 → 1.00 s (1.2K), median request time −8/−16 %, throughput unchanged, TTFT p95 +20–25 %.
- `linux/series_overlap.sh`, `linux/results/overlap/` (M0 baseline, exactness matrix, fault injection).

## [0.7.0] – 2026-10-11

Windows (RTX 5090 + 64 GB).

### Added
- `Start-Alles.bat` / `Stop-Alles.bat`: Strata + SearXNG + Open WebUI in one go; services already running are left alone.
- `cors_origins` (`http://localhost:3000`, `http://127.0.0.1:3000`) in all three configs: browser clients such as
  Genesis may call `/v1/*` directly. Before, `Access-Control-Allow-Origin` was missing and the browser discarded
  the responses ("Could not connect … after 3 attempts").
- `batch_bench.py` (`batch_results.jsonl`, configs `BB*`/`CB*`): decode/prefill with 1/2/4 concurrent requests and
  decode next to a long prompt, against a selectable Strata build (`--strata`). Measured 2026-10-10 with engine
  v0.1.40.2 (worktree `Strata-1402`, fixes the `--batch-mtp` crash): UD-IQ4_XS without batching 116–131 tok/s total,
  2 slots 122, 4 slots 119 (a single request drops to 85); Coder without batching 162, 2 slots 177, 4 slots 189.
  With UD-IQ4_XS ~60% of experts run on the CPU, so batching does not pay off there.

### Changed
- UD-IQ4_XS: parking raised from 3 slots / 6 GiB to `--conversation-cache-slots 12 --conversation-cache-mib 10240`,
  so ~10 alternating contexts are not re-read every time.
- `LIESMICH.txt`, `settings.cmd`: notes on Start-/Stop-Alles and the firewall rule.

## [0.6.0] – 2026-10-11

### Added
- Engine patch series 3 "pointer-table MMQ" (`linux/patches/mmq-tab/`, build `-DSTRATA_MMQ_TAB=ON`, env
  `STRATA_MMQ_DIRECT=1`): MMQ reads VRAM-resident experts in place instead of copying them per prompt chunk.
  Bit-identical (tokens and logits). 16 users: short prompts 417 → 444 tok/s, 1.2K prompts 296 → 327 tok/s,
  16 agents 62 → 72 tok/s; single-user TTFT for a 1.2K prompt −17%.
- `linux/TESTLOG.md` (chronological test log), `linux/upstream/ANALYSIS-*.md` and `DESIGN-mmq-direct.md`.
- `start-strata.sh` now defaults to `configs/T16-mmqtab.json`.

### Changed
- All project documentation in `linux/` and this changelog are now in English.

## [0.5.0] – 2026-10-11

### Added
- Linux: `--ple-io ram` in the recommended config (`linux/configs/T16-direct-pleram.json`, default of
  `start-strata.sh`): ~417 tok/s (short prompts) or ~297 tok/s (1.2K prompts) with 16 users.
- Profiling of the batch windows and of prefill (`linux/results/prof_*`, `prefill_timing*.txt`) and a
  kernel analysis for short prompts: 31% of prefill time is an expert copy for llama.cpp's MMQ, possible fixes
  in the report (`linux/README.md`, "Profiling and kernel analysis").
- Experimental engine patch "overlapping admission" (`linux/patches/experimental/`, for measurement only) and the
  complete, cross-checked design for it (`linux/upstream/DESIGN-overlap.md`): no throughput gain on Ampere,
  but −70% time to first token.

### Rejected (measured)
- Rebalancing the layer split (`12,24,37`, `13,25,37`): no gain.
- `STRATA_PF_FUSED=1` for the UD-IQ4_XS: only kicks in from 1,024-token chunks, no effect with `--prefill 512`.

## [0.4.0] – 2026-10-11

### Added
- **Linux operation on 4× RTX 3090** (`linux/`, report `linux/README.md`): setup via `Strata/setup.sh` with
  layer split across all 4 cards, `start-strata.sh` (own systemd scope, port 8090), bench tools
  `bench_concurrency.py` (load with many users) and `bench_multiturn.py` (agent load), runner
  `run_strata_variant.sh`, all measured configs and results.
- **Engine patches** for Strata (`linux/patches/`, tested bit-identical to solo):
  more than 8 batch slots with layer split (8 rows per pipeline group) and direct GPU→GPU slot copy
  (65.6 → 34.0 ms per admission). Upstream texts in `linux/upstream/` (not yet submitted).
- Recommended Linux config: split `12,24,36` + `--trim-stage-weights`, 16 slots, `--prefill 512`, MTP:
  ~121 tok/s for a single user, ~405 tok/s total (short prompts) or ~294 tok/s (1.2K prompts);
  agent variant with 262K context, `--kv-resident 32768` and 24 GB conversation cache.
- Comparison with vLLM 0.31 (Flash-Next W4A16 with FP8 n-gram table via patch, with/without MTP; Qwen3.8-27B-FP8),
  quality set (`linux/eval/`: MMLU-Pro 240 + GSM8K 150): Strata UD-IQ4_XS 77.9% MMLU-Pro vs. 27B-FP8 72.1%
  vs. vLLM-W4A16 70.4%; evaluation of all 910 Flash-Next repos on Hugging Face (`linux/quant-survey/`).

### Rejected (measured)
- `STRATA_BATCH_DECODE_SHARE`: ineffective with layer split and pipeline groups (prompt is read in one piece).
- Fewer pipeline groups (2 instead of 4), 2 instances of 2 GPUs each, more than 16 slots, larger prefill chunks.
- RAZOR-65B (50% of experts pruned) in vLLM: clearly worse quality (62.5% MMLU-Pro).

## [0.3.0] – 2026-10-09

### Added
- Git repo with README, changelog and setup guide (`docs/EINRICHTUNG.md`).
- `strata-configs/`: the three model configs as templates (plus the setup original of the IQ4_XS config for comparison).

## [0.2.0] – 2026-10-08

### Added
- Dashboard (`dashboard/`, `Start-Dashboard.bat`, port 8082): speed, caches, VRAM/RAM/commit, SSD traffic,
  estimated power at the wall socket, kWh, € and € per 1M tokens (input/output), runtime shares, 24-h projection,
  reset with archive. Starts along with every model and survives model switches.
- `agent_bench.py`: A/B runner with agentic load (workloads `agent`, `session`, `diverge`, `longctx`,
  `parallel`), also measures commit and physical RAM.
- `trace_analyze.py` for routing traces; measurement series E–H (`series_*.sh`, `configs/`, `logs/`, `agent_results.jsonl`).
- Final report `ANALYSE-Speicherhierarchie-XS.md` and lever list `HEBEL-Kandidaten-XS.md`.

### Changed
- UD-IQ4_XS config tuned: `--resident-experts` (instead of `--resident-budget-gib 40`), `--ple-row-cache 33554432`,
  `--conversation-cache-mib 6144 --conversation-cache-slots 3`, `--adapt-async 1`, `--vram-reserve-mib 1000`,
  `--kv-resident 32768`, `STRATA_KV_PREFETCH=1`. Decode ~125 → ~145 tok/s, agent switch ~25 s → ~1.5 s,
  no more SSD reads in the prompt path.
- Moved GGUF files from `X:` (Crucial P3) to `C:\StrataQ38FN-models` (SN850X), `--ple-io direct`.

### Fixed
- Engine hangs while reading prompts ("layer 1 of the prompt chunk", 9 of them on Oct 7/8); none since the move.

### Rejected (measured)
- Batching (`parallel: 2` + `--batch-mtp`): crash on concurrent requests (MTP bug), otherwise −18%.
- `--vram-reserve-mib` < 800, `pcie_frac` > 0.7, `--spec` ≠ 4, `--spec-min-p` ≠ 0.5.
- `STRATA_CACHE_MESSAGE_BOUNDARY=1`: no effect.

## [0.1.0] – 2026-10-06

### Added
- Strata setup for three models: Unsloth UD-IQ4_XS, Unsloth UD-Q4_K_XL, ISTA Coder IQ1_M.
- Start scripts (`Start-IQ4_XS.bat`, `Start-Q4_K_XL.bat`, `Start-Coder.bat`, `Stop-All.bat`), shared
  settings in `settings.cmd`, API key from `api_key.txt`, LAN operation on 8080/8081.
- vLLM facade (`vllm_facade/`, port 8081) for vLLM clients such as genesis3, with test suite.
- Benchmarks `bench_strata.py`, `run_variant.py`, `parallel_bench.py`, `stall_probe.py`; measurement series A–D.
- Resumable downloads `download_hf.py`, `download_q4kxl.py`.
- `LIESMICH.txt` as quick-start guide.
