# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added (dashboard: admin login, Strata settings) - 2026-10-18
- **Everyone sees everything, every change needs the admin login.** One login box for all tabs of ⚙ Einstellungen
  plus a header button (🔒 Anmelden / ✓ Admin); the password is G3's admin password (`dashboard/g3control.py`, no
  second password store). Without the login every field and button is locked and the server answers 401. Power
  price/consumption and "Neue Zählung" (`/api/settings`, `/api/reset`) were open before and now need the login too
  (same-origin + JSON check as the G3 writes).
- **Strata-Server** tab: `strata.service` status (since when, server and engine PID, systemd restarts), the config
  (`production.json` → its target) and engine binary, requests running now, **restart** with a confirmation (says
  how many requests break off; the RAM conversation cache is empty afterwards, the SSD tier is kept), and the tail of
  the engine log (login only).
- **Cache & SSD** tab: conversation cache in RAM (`--conversation-cache-mib` as GiB, `-slots`, `-min-free-mib` as
  GiB, `-min-tokens`) and the SSD tier (`STRATA_CONV_TIER_WRITE_GB_DAY` write budget per day, `_CAP_GB` the most it
  keeps on disk - full means the least recently used conversations are overwritten, `_MIN_TOKENS`, `_TTL_H` shown as
  "Aufbewahrungszeit" = time since the last use, `_MIN_FREE_GB`). Each field shows the value the running engine was
  started with (read from `/proc` cmdline + environ), so saved-but-pending changes are marked; "Speichern" writes the
  production config (the file the symlink points at, same JSON format, only the changed keys), "Speichern & Strata
  neu starten" applies it. Ranges are checked in the page and the server; the disk cap must fit next to what the disk
  keeps free. Disk line: tier size, free space, TB/year at a full day budget. Config, binary and model cannot be
  switched here on purpose (that stays a deploy).
- Main panel, SSD tier: "auf Disk" shows used / cap.
- New module `dashboard/strata_admin.py`; `dashboard.py` gets `GET /api/strata` (everyone; log only with the login),
  `POST /api/control/strata/config` and `/api/control/strata/restart` (login) and the option `--strata-unit`
  (default `strata.service`). Active on Linux with `--engine-config` and the G3 login.

### Changed (production) - 2026-10-18
- SSD tier: **1,000 GB/day** write budget (was 300; the day's 300 GB were used up by 16:00 and further evictions were
  dropped) and a **500 GB disk cap** (`STRATA_CONV_TIER_CAP_GB`, engine default 600), set through the new monitor
  tab; Strata restarted at 16:30, tier warm (279 conversations kept). Worst case 365 TB/year against ~3,600 TBW of the
  RAID members, about ten years.

### Changed (dashboard, "Strata Monitor")
- New layout: a status band (phase, throughput now, batch slots, GPUs, wall power, cost) and panels by question
  (Was läuft gerade? · Wie schnell? · Was kostet es? · Ist die Hardware ok? · Speicher & Caches · Verlauf ·
  Requests) with section chips. Tiles with long grey sub-sentences became tables and key/value lists: energy
  (active/idle/offline × W/kWh/€), the 24 h projection, one row per GPU and per SSD. Every value of the old page is
  still shown as text (inventory of ~230 values checked against the new page); phone layout one column.
- Fixed: the id `slots` was used twice, so the batch slot cards never rendered and "Expert-Slots" showed "–". Three
  charts now show their current value; values that only lived in tooltips (slot context, timing parts) are text.
- Parked conversations: besides GB also approx. tokens used / fitting ("ca. 718 K von 2,69 Mio Tokens"), from the
  bytes per token of the last 200 parks in the engine log (`conversation_cache.park_bpt`; approximate, since every
  parked conversation also has a fixed part of a few hundred MB).

### Added (dashboard)
- **⚙ Einstellungen** (side panel): power price & consumption (as before), **GPU power limits** per GPU for day and
  night plus the night window and on/off, and **fan curves** per GPU for day and night (auto or a curve, with a
  preview against the floor curve and the current temperature). Power and fans are owned by G3: the monitor reads
  G3's settings files without a login and writes only through G3's admin API (`/api/admin/power|fan/settings`)
  after a login with the G3 admin password (`dashboard/g3control.py`; the G3 session stays on the server, the
  browser gets an HttpOnly cookie; writes only from the page itself: `Sec-Fetch-Site`/`Origin` check + JSON).
  Fans: **more cooling only** - a curve may nowhere run below G3's default curve (40 °C 30 % · 60 °C 45 % · 70 °C
  65 % · 78 °C 85 % · 83 °C 100 %), minimum speed ≥ 30 %, the 100 % limit ≤ 83 °C, hysteresis ≥ 3 °C (checked in
  the page and in `dashboard.py`). New options `--g3 URL` (default `http://127.0.0.1:8000`, `off` = no menu) and
  `--g3-dir`.
- **ⓘ API & Nutzung**: endpoints (G3 :8000 OpenAI/Anthropic/Responses, Strata :8090), model name, context window
  (live), what happens on overlong prompts, parallel slots, thinking levels for OpenAI and Anthropic requests
  (`reasoning_effort` none/low/medium/high, `enable_thinking`, `thinking`, `output_config`,
  `reasoning_budget_tokens`), sending images (`image_url` with base64/URL, Anthropic image blocks) and copyable
  curl/Python examples.

### Added (Linux, images)
- Images (vision) through the API: OpenAI `image_url` parts (base64 `data:` URL, http(s), local path) and Anthropic
  image blocks, also through G3. Encoder `strata-vision` built for CPU and CUDA, mmproj from ISTA-DASLab (ed59f92),
  configs `T16-143-prod262r-vision-cpu.json` / `-vision-gpu.json`, check script `linux/vision_check.py`. Measured on
  the production config (TESTLOG #75): 15/15 checks (shapes, colours, exact text, no mix-up between two pictures
  behind one prefix, follow-up turns, repeat), ~2.2 s per 800×600 picture, 5.1 s for a large photo; 16 text users +
  pictures: 491 tok/s text, pictures right in 4.8–11.3 s. **Live since 2026-10-13 02:40**: `production.json` →
  `T16-143-prod262r-vision-cpu.json` (engine k5, server from `Strata-143-prod` at fcd7ed8f); text and images checked
  through G3 after the switch.
- Engine patches 0041–0043 (`linux/patches/v0.1.43/`, engine `strata-g3-143k5`, server Strata-143 fcd7ed8f):
  - 0042: with `--vision --batch` the engine refuses to mix image and text rows (`ERR vision: ...`). Upstream keeps
    ONE M-RoPE (image position) table per GPU for all batch slots and rebuilds it at every admission, so an image
    slot and text slots decoded with each other's positions: silently wrong text (found by a code audit).
  - 0043: the server runs an image request alone: new requests wait from its arrival, the slots finish, the picture
    runs on the solo path; `STRATA_IMAGE_WAIT_S` (120 s, then a 503 to retry). A dead image encoder now fails
    only image requests, not text.
- Design for per-slot image positions, so pictures can share the batch without the wait:
  `linux/upstream/DESIGN-vision-batch-rope.md` (one table region per slot, addressed through the position value; no
  kernel change, ~51 MiB VRAM per GPU). Implemented as experimental patch 0044
  (`linux/patches/v0.1.43/experimental/`, branch g3-143-vrope 49d0d3e0, engine `strata-g3-143v1`): CPU tests and three
  reviews pass; the GPU check (`tools/vision_batch_check.py`) has not run yet, so it is not in production.

### Fixed
- 0041: `STRATA_CKPT_POOL` did not clear `slot_turn` on a recycled checkpoint (with `STRATA_SLOT_PARK`), so such a
  checkpoint was left out of later slot chains. Tokens are unchanged.

### Added
- Dashboard for Windows and Linux (`dashboard/`, `linux/start-dashboard.sh`): Strata's live view (answering progress of
  the running request, prompt-reading progress, every `--batch` slot, queue), one card per GPU (load, VRAM, power,
  temperature, SM clock, PCIe link gen/width and RX/TX MB/s) with per-GPU charts, Linux memory (`/proc/meminfo`) and
  disks (md/dm volumes, no double counting). Runs without a Strata server (engine panels: offline / last seen);
  sources configurable (`--source URL[,KEYFILE]`) plus the servers found in the process list, answers that are not
  Strata's `/metrics` are ignored. NVML via Strata's `telemetry.py` (`--strata`) or an own ctypes reader.
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

## [0.19.0] – 2026-10-12

Linux (4× RTX 3090, 256 GB RAM): production.

### Added
- Production config `linux/configs/production.json` → `T16-143-prod262`: up to 262K context per conversation (K/V
  streamed, 32K cells in VRAM, split 12,25,37 without expert loans), 48 GiB conversation cache, async + slot parking,
  SSD emergency tier. 16 users 540 / 428 tok/s, a 170K-token prompt is read in ~30 s (TESTLOG #73).
- Stability soak: 13 cycles of mixed load (16 users, mixed lengths, 24 agents, 150K requests, dropped streams) without
  a single error; VRAM flat (TESTLOG #74).
- Live since 2026-10-12 20:05: Strata behind G3's OpenAI proxy on port 8000 as "Qwen3.8-Flash-Next" (G3 2.6.0:
  external backends, night-mode switch, optional GPU fan curves), Open WebUI on 8080, LAN dashboard on 8082, all as
  systemd user services with linger; GPU power by G3: 380 W day, 280 W night (22:00–07:00).

## [0.18.0] – 2026-10-12

Linux (4× RTX 3090, 256 GB RAM).

### Added
- KV tiering stage 3: SSD emergency tier (`STRATA_CONV_TIER_DIR`, exact): conversations the RAM cache must evict go to
  the RAID only if ≥ 16K tokens, written once, daily write budget 200 GB, cap, TTL; counters at GET /metrics and in
  the dashboard. +25 % throughput when the RAM tier overflows, no cost otherwise. Default
  `configs/T16-143-64khocast.json` (production config, `linux/configs/production.json`).
- Production setup: systemd user services for Strata, G3, Open WebUI and the dashboard (`linux/systemd/`),
  `linux/cutover.sh` / `linux/rollback.sh` for the switch from G3/vLLM to Strata behind G3's proxy.

## [0.17.0] – 2026-10-12

Linux (4× RTX 3090, now 256 GB RAM).

### Added
- KV tiering stage 2a: slot parking (`STRATA_SLOT_PARK=1`, exact, verifiable byte by byte): follow-up turns continue
  after the decoded reply. New default `configs/T16-143-64khocas.json`: multi-turn +2.6–2.9 %, one-shot TTFT −30–40 %.
- 256 GB RAM: Strata scopes may use 200 GB; `configs/T16-143-64khocas128.json` with a 128 GiB conversation cache.

## [0.16.0] – 2026-10-12

Linux (4× RTX 3090).

### Added
- KV tiering stage 1: asynchronous parking (`STRATA_PARK_ASYNC=1`, exact, every park verifiable byte by byte with
  `STRATA_PARK_ASYNC_VERIFY=1`). New default `configs/T16-143-64khoca.json`: the 48 GiB conversation cache no longer
  costs one-shot speed (573 / 451 tok/s vs 535 / 426 before, no cache 575 / 454); multi-turn 182.7 tok/s.
- `linux/upstream/DESIGN-kv-tiering.md`: staged plan VRAM slots → RAM conversation cache → SSD emergency tier.

## [0.15.0] – 2026-10-12

Linux (4× RTX 3090).

### Changed
- KV-first default `configs/T16-143-64khoc.json`: 64K context per slot (2×), 48 GiB conversation cache, pipelined stage
  hand-off (`STRATA_HANDOFF_PIPE=8`, exact, no VRAM). Multi-turn (24 agents × 4 turns): 176.8 tok/s and 5.6 s
  follow-up TTFT instead of 91.5 tok/s / 11.3 s; one-shot −6 % (parking). Throughput config without the cache:
  `configs/T16-143-64kho.json` (575 / 454 tok/s).

### Added
- KV-cache audit (`linux/results/kvaudit/`): VRAM per card, what each patch costs, where the re-prefill cascade came
  from.

## [0.14.0] – 2026-10-12

Linux (4× RTX 3090).

### Changed
- Engine rebased onto upstream Strata 0.1.43 (`linux/patches/v0.1.43/`), exact. New default `configs/T16-143l5.json`
  with lever 5 `STRATA_QSA_SLOTS=1` (multi-slot QSA attention, bitwise identical) and `STRATA_NAN_GUARD=0`:
  16 users 578 tok/s short / 456 tok/s 1.2K (+5.3 % / +4.1 %, ABBA), replies identical.

### Added
- One cross-platform LAN dashboard (`dashboard/`, `linux/start-dashboard.sh`, port 8082): Strata's live answering
  bar and batch slots, per-GPU load/VRAM/power/temperature/clock and PCIe link + RX/TX throughput for all cards; runs
  without a Strata server.

## [0.13.0] – 2026-10-12

Linux (4× RTX 3090).

### Added
- Engine levers 3 + 4 (opt-in, exact): `STRATA_MMQ_GROUP_SORT=1` (MMQ groups sorted by row count, 1.2K read −17 %),
  `STRATA_BATCH_SELF_COMMIT=1` (no commit graph per batch window, decode +4.9 %). New default `configs/T16-142gsc.json`:
  16 users 549 tok/s short / 439 tok/s 1.2K (+7.3 % / +10.6 % vs 0.12.0, same session, ABBA), replies identical.
- Exactness checks from the 2026-10-12 review: reply hashes through the server, `batch_interleave_test` on a pipelined
  batch (steps 4–7 now run), sampled batch = solo, agent config on 0.1.42 (`X16-agent-142gsc`, split 12,25,37).
- Benchmark tooling: deterministic prompts, per-request reply sha256 (`compare_replies.py`), GPU telemetry, steady-state
  rate, ABAB/ABBA mode in `ab_quick.sh`.

### Changed
- Root README in English, positioned as a performance fork of Strata for an RTX 5090 + RAM and 4× RTX 3090 + RAM.

### Found
- A solo turn decoded with MTP draft windows leaves a numerically different K/V/GDN state than one-row slot windows
  (its tokens are equal), so a later turn continued from the solo session can diverge from one continued from a slot
  (upstream behaviour, TESTLOG #59).

## [0.12.0] – 2026-10-12

Linux (4× RTX 3090).

### Changed
- New default `configs/T16-142fcb.json`: layer split 12,24,37 (GPU3 was the decode bottleneck) and
  `STRATA_SPLIT_MTP_BATCH=1`. 16 users 522 tok/s short / 405 tok/s 1.2K (was 491 / 378), output identical.

### Added
- `upstream/PROFILE-next-levers.md`: GPU profile of prefill and batch decode, ranked next levers.

## [0.11.0] – 2026-10-12

Linux (4× RTX 3090).

### Changed
- New default `configs/T16-142fca.json` = `T16-142fc` + `--aux-cpus 65-127`: the prefill stage threads no longer
  share the host's CPU 0. 16 users with 1.2K prompts 357 → 376 tok/s, short unchanged (493), output identical.

### Added
- `upstream/DESIGN-batched-admission.md`: reading several queued admissions together - measured, mostly a lockstep
  artifact of the benchmark (fixed `max_tok`); only a short admission hold is worth building.

## [0.10.0] – 2026-10-12

Linux (4× RTX 3090).

### Added
- Engine patch series 5 "fixed per-admission cost" (six opt-in switches, `linux/patches/v0.1.42/`), exact. New
  default `configs/T16-142fc.json`: 16 users 493 tok/s short / 357 tok/s 1.2K (was 470 / 349), TTFT p95 −21 %.
  Note: both figures are greedy (`ab_quick.sh`, baseline `T16-142` re-measured), i.e. +5.0 % / +2.2 %; the 0.9.0
  figures 467 / 342 were at temperature 0.7 (3 runs), so 0.9.0 → 0.10.0 must not be read as +5.6 % / +4.4 %.

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
- Rebalancing the layer split (`12,24,37`, `13,25,37`): no gain. *Later adopted:* on 0.1.42, after the `--aux-cpus`
  fix (0.11.0), split `12,24,37` gave +5 % and is part of the default since 0.12.0 (TESTLOG #30 → #56/#57).
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
