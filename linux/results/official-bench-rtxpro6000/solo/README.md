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
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 224.1 (182.5-287.9) | 0.13 (0.12-0.14) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 4011.8 (3962.7-4013.8) | not measured | 1.01 (1.01-1.02) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4134.4 (4132.4-4145.1) | not measured | 3.90 (3.89-3.90) |
| agent session, follow-up turns | 14565 (14548-14606) at the last turn | see below | up to 64 | 3 | see below | 222.4 (199.8-246.1) | 1.00 (0.51-1.15) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.8); earlier-turn tokens read again per session: 72 (3-73) (earlier turns that the cache did not hold; the 13729 (13728-13731) new tokens of the files are read once); first turn (nothing cached) first token: 0.29 s (0.29-0.29).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 2198.6 | 0.287 | 243.5 |
| 1 | 3745 | 623 | 3122 | 2956.4 | 1.069 | 246.1 |
| 2 | 5278 | 3809 | 1469 | 2900.9 | 0.519 | 199.8 |
| 3 | 8722 | 5342 | 3380 | 2983.5 | 1.147 | 238.0 |
| 4 | 11648 | 8715 | 2933 | 2933.0 | 1.015 | 218.6 |
| 5 | 14565 | 11666 | 2899 | 2927.4 | 1.005 | 227.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 40 s.
