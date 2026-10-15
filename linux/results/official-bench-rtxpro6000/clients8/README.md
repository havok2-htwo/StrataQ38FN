# Community benchmark on NVIDIA RTX PRO 6000 Blackwell Workstation Edition

Measured on 2026-10-15 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GiB VRAM
- CPU: AMD Ryzen 5 9600X 6-Core Processor (6 cores, 12 threads); installed RAM: 123.4 GiB; storage: not measured
- OS: Linux 7.0.0-34-generic; GPU driver: 595.91.07; CUDA: 13.2
- Strata: project 0.1.44.1, commit not a git checkout; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs
- Context: 32768 tokens; parallel slots: 8
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --clients 8` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 219.8 (184.2-300.4) | 0.13 (0.12-0.14) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3978.4 (3953.4-4001.0) | not measured | 1.02 (1.01-1.03) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4079.9 (4069.0-4082.8) | not measured | 3.95 (3.95-3.96) |
| agent session, follow-up turns | 14559 (14551-14604) at the last turn | see below | up to 64 | 3 | see below | 217.6 (155.0-270.6) | 1.01 (0.52-1.17) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.7 % (68.5-68.8); earlier-turn tokens read again per session: 3 (2-143) (earlier turns that the cache did not hold; the 13728 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 0.29 s (0.29-0.29).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 2184.4 | 0.288 | 217.4 |
| 1 | 3745 | 623 | 3122 | 2925.1 | 1.08 | 268.3 |
| 2 | 5278 | 3809 | 1469 | 2881.5 | 0.522 | 197.4 |
| 3 | 8722 | 5342 | 3380 | 2968.8 | 1.152 | 217.6 |
| 4 | 11607 | 8745 | 2862 | 2895.9 | 1.003 | 244.9 |
| 5 | 14551 | 11652 | 2899 | 2911.8 | 1.011 | 155.0 |

### Concurrency: 8 clients

Total 210.0 tok/s (205.2-210.8); first token p50 1.017 s, p95 1.856 s; request latency p50 4.848 s, p95 4.974 s; errors 0. Server parallel slots: 8 (requests beyond that wait their turn).

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 55 s.
