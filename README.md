# StrataQ38FN

**A performance fork of the [Strata](https://github.com/Niko1221/Strata) inference engine for running
Qwen3.8-Flash-Next (a 180B-class MoE) on consumer GPUs: one RTX 5090 with system RAM, and 4× RTX 3090 with system
RAM.**

Strata already runs this model on hardware far smaller than the model. This project pushes it further on two
concrete setups. We measure, profile and patch the engine, and tune every setting for real multi-user and agentic
load. Every engine change is an opt-in patch on top of upstream Strata, proven bit-exact (each request returns the
same tokens as when it runs alone). Every measurement is logged, so each number in this README can be traced back
to a test.

| Setup | Original Strata | This fork | Gain |
|---|---|---|---|
| **4× RTX 3090 + RAM**, 16 concurrent users, short prompts | ~88 tok/s (setup default) · ~175 tok/s (Strata's own batching, max. 8 slots) | **~578 tok/s** | **6.6×** · **3.3×** |
| **4× RTX 3090 + RAM**, 16 concurrent users, 1.2K-token prompts | ~88 tok/s · ~175 tok/s | **~456 tok/s** | **5.2×** · **2.6×** |
| **4× RTX 3090**, single user (decode) | ~121 tok/s | ~131 tok/s | +8 % |
| **RTX 5090 + 64 GB RAM**, single user (decode) | ~125 tok/s (setup default) | **~145 tok/s** | +16 % |
| **RTX 5090 + 64 GB RAM**, switch between main agent and sub-agent | ~25 s prompt re-read | **~1.5 s** | 16× |

4× RTX 3090: Strata 0.1.43 + our patch series, greedy, 512 output tokens per request (TESTLOG #57–#64 in
[linux/TESTLOG.md](linux/TESTLOG.md)). For comparison, vLLM 0.31 on the same box reaches 622 / 386 tok/s at 16
users, with a measurably lower quality checkpoint ([linux/README.md](linux/README.md)).

**Is this a fork?** In substance, yes: a downstream fork. It is not a GitHub fork of the engine repository. This
repository holds the deployment (configs, start scripts, a vLLM-compatible facade, a dashboard) and the engine patch
series. The engine itself is upstream Strata plus our patches, applied in a clone (`linux/patches/v0.1.43/`, 35
patches). The patches are kept upstream-ready (opt-in switches, default path unchanged), so they can be offered back
to Strata.

## What the fork changes

**4× RTX 3090 (multi-GPU, layer split)**: [linux/README.md](linux/README.md)
- More than 8 batch slots on a layer split (8 rows per pipeline group): 16–32 concurrent users instead of 8.
- Pointer-table MMQ: prompt reads use VRAM-resident experts in place instead of copying them.
- An admission path with ~25 ms less fixed cost per new request; checkpoints are saved on 4 threads and moved into
  the slot.
- MMQ expert groups sorted by row count (−17 % prompt read time).
- Batch windows that commit their own state (+5 % decode) and run QSA attention for all slots in one launch (+6 % decode).
- An optional overlapped admission: new prompts are read while the other users keep decoding (latency option).
- Tuning found by profiling: layer split 12,24,37 (GPU3 was the decode bottleneck), `--aux-cpus` (the prefill
  threads were stuck on the host's CPU core), batched draft-layer reads.
- A test and measurement toolkit:
  - exactness tests through the engine and through the HTTP server (reply hashes);
  - quick A/B runs with GPU telemetry;
  - a profiling report and ranked lever list ([linux/upstream/](linux/upstream/)).

**RTX 5090 + RAM (one GPU, experts partly in system RAM)**: this README, below
- Tuned residency, caching and prefetch for agentic load: resident experts, a conversation cache for parked
  sessions, asynchronous expert adaptation, KV prefetch, direct PLE I/O.
- Main/sub-agent switching drops from ~25 s to ~1.5 s, and the prompt path makes no SSD reads.
- Start/stop scripts, a vLLM-compatible facade for vLLM clients, and a LAN dashboard.

## This repository

The engine and the models are **not** included. The repository holds everything around them:

- start/stop scripts for three model variants (only one runs at a time)
- tuned Strata configs (`strata-configs/`, `linux/configs/`)
- a **vLLM-compatible facade** (`vllm_facade/`), so vLLM clients (e.g. genesis3) connect unchanged
- a **LAN dashboard** (`dashboard/`, Windows and Linux) with live requests (answering progress, batch slots),
  speed, caches, memory, every GPU (load, VRAM, power, temperature, clocks, PCIe link and traffic), power and cost;
  it also runs without a Strata server (`Start-Dashboard.bat`, Linux: `linux/start-dashboard.sh`, port 8082)
- benchmark and A/B tools with their measurement data and analysis reports
- the engine patch series and its design documents (`linux/patches/`, `linux/upstream/`)

**Linux with 4× RTX 3090** (setup, batching up to 32 slots, comparison with vLLM): **[linux/README.md](linux/README.md)**

Step-by-step setup (Windows): **[docs/SETUP.md](docs/SETUP.md)** · Changes: [CHANGELOG.md](CHANGELOG.md) ·
Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Windows: RTX 5090 + 64 GB RAM

### Models

| Start | Model | Quality / use | Decode | Context |
|---|---|---|---|---|
| `Start-IQ4_XS.bat` **(recommended)** | Unsloth UD-IQ4_XS (~4 bit) | all-round, German | ~145 tok/s | 128K |
| `Start-Q4_K_XL.bat` | Unsloth UD-Q4_K_XL (~5 bit) | slightly better quality, needs ~45 GB RAM | 85–95 tok/s | 64K |
| `Start-Coder.bat` | ISTA Coder IQ1_M (half the experts) | code / English only | 165–230 tok/s | 128K |

`Stop-All.bat` stops the model and the facade. Starting a model stops the one that was running.

`Start-Alles.bat [IQ4_XS|Q4_K_XL|Coder]` also starts SearXNG (web search) and Open WebUI. Services that are already
running are left alone, and a loaded model is not reloaded. `Stop-Alles.bat` stops everything except the dashboard.

### Services and ports

| Port | Service | Address |
|---|---|---|
| 8080 | Strata: web chat, OpenAI and Anthropic API | `http://<PC>:8080/`, `/v1`, `/v1/messages` |
| 8081 | vLLM facade (for vLLM clients) | `http://<PC>:8081/v1` |
| 8082 | Dashboard (numbers only, no prompts) | `http://<PC>:8082/` |
| 8090 | benchmarks only (`agent_bench.py`) | – |
| 3000 | Open WebUI (own setup, `X:\dev\OpenWebUI`) | `https://<PC>:3000/` |
| 8888 | SearXNG (web search for Open WebUI, `X:\dev\SearXNG`) | – |

All API calls need the header `Authorization: Bearer <key from api_key.txt>`. Ports, LAN/local binding and the model
name for genesis3 are set in `settings.cmd`.

**Browser clients (CORS):** web apps that call `/v1/*` directly from the browser (e.g. Genesis at
`http://localhost:3000`) need their origin in `cors_origins` of the Strata config. `http://localhost:3000` and
`http://127.0.0.1:3000` are listed. If the origin is missing, the browser discards the reply and the client reports a
misleading "Could not connect". Restart Strata after a change.

### Tuning results (UD-IQ4_XS)

Measured on 2026-10-08 (details: [ANALYSE-Speicherhierarchie-XS.md](ANALYSE-Speicherhierarchie-XS.md), sections 7–8):

| | Setup default | Tuned |
|---|---|---|
| Decode, short context | ~125 tok/s | **~145 tok/s** |
| Switch main agent ↔ sub-agent | ~25 s prefill | **~1.5 s** |
| SSD reads in the prompt path | yes (1.1 TB) | none |

The key flags are:
- `--resident-experts`
- `--conversation-cache-mib 10240 --conversation-cache-slots 12`
- `--adapt-async 1`
- `--kv-resident 32768` + `STRATA_KV_PREFETCH=1`
- `--vram-reserve-mib 1000`
- `--ple-io direct`

Engine 0.1.42 adds +12 % (prose) / +13 % (code) decode through upstream's `STRATA_ROUTE_TAIL_SKIP=7`, which skips
experts that are not in VRAM (see CHANGELOG, Unreleased).

**Parking:** up to 12 conversations (max. 10 GiB RAM, ~0.9 GB each with a short context) stay in RAM between
requests. An evicted conversation is not swapped to SSD; it is read again the next time. If you switch between more
contexts, raise the slots and MiB (Strata keeps a 2.5 GB RAM floor free).

**The limit is the Windows commit charge, not physical RAM.** A 262K context runs without a speed loss, but it is
only safe with a fixed-size page file (see setup, step 9).

**Dead ends** (measured, do not repeat):
- **Batching** (`parallel` + `--batch-mtp`) gives no throughput with UD-IQ4_XS.
  - The crash was an engine bug, fixed in Strata v0.1.40.2.
  - Since then 2 slots give 122 instead of 128 tok/s in total, because ~60 % of the experts run on the CPU.
  - Only the Coder model, with all experts in VRAM, gains: +17 % at 4 slots (`batch_bench.py`,
    `batch_results.jsonl`).
- `--vram-reserve-mib` < 800 reports LOW.
- `pcie_frac` > 0.7, `--spec` ≠ 4 and `--spec-min-p` ≠ 0.5 are slower.

## Directory

```
Start-*.bat, Stop-*.bat       operation (double-click)
_start-model.cmd, _run-*.cmd  internal: starts Strata + facade (+ dashboard)
settings.cmd                  ports, bind address, genesis3 model name
strata-configs/               Strata configs (templates, copied into Strata\)
vllm_facade/                  vLLM facade, own README and tests
dashboard/                    LAN dashboard (dashboard.py, index.html, settings.json)
agent_bench.py                A/B runner with agentic load (port 8090)
bench_strata.py, run_variant.py, parallel_bench.py, stall_probe.py, trace_analyze.py
series_*.sh                   the measurement series that were run
configs/, logs/, *.jsonl      variants, engine logs and results of the measurement series
ANALYSE-*.md, HEBEL-*.md      final report and open levers (Windows, German)
download_hf.py, download_q4kxl.py  resumable Hugging Face downloads
linux/                        Linux on 4x RTX 3090: start/bench scripts, configs, results, engine patch series,
                              design and profiling reports, vLLM comparison, quality eval
```

Not in the repository (`.gitignore`): `Strata/` and `Strata-*/` (engine clones + venv), `Strata-data/` (packs, MTP),
`api_key.txt`, session files, traces, runtime logs, dashboard counters.

## Tested environments

**Windows:** Windows 11 Pro · AMD Ryzen 7 9800X3D · 64 GB RAM · NVIDIA RTX 5090 (driver 610.88) · Python 3.12.4 ·
Strata commit `82f46a8` (0.1.42 in the Unreleased measurements) · models on NVMe (WD SN850X, `C:\StrataQ38FN-models`).

**Linux** (see [linux/README.md](linux/README.md)): Ubuntu, kernel 7.0 · AMD Threadripper 3990X · 128 GB RAM ·
4× NVIDIA RTX 3090 (driver 580, CUDA 13.0) · Strata v0.1.43 plus our patch series · models on RAID
(`/media/raid0/Models/Strata-GGUF`).

## License

Strata is under the MIT license (see the upstream repository). Our engine patches contain Strata and llama.cpp/ggml
code (MIT); the notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The models are under the licenses of
Qwen and Unsloth.
