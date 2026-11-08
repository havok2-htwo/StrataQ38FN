**Penalty on greedy requests: numbers from 4x RTX 3090, Flash-Next, v0.2.12**

Adding numbers from a different setup: Flash-Next (`qwen4exp`) on CUDA, with greedy requests. The card's penalty does reach `temperature: 0` requests. On stock v0.2.12 with our config, it cost little speed in our runs. The cost is large on decode paths that refuse penalised requests.

**Setup**
- 4x RTX 3090 (layer split, 320 W), CUDA 13.0, driver 580.
- Qwen3.8-Flash-Next UD-IQ4_XS.
- Strata v0.2.12 (146df3c4), 16 slots, 262K context, `--kv int8`, `--spec 4`, `--prefill 512`, MTP drafting.

**What happens.** Take a chat request with `temperature: 0`, thinking off and no `presence_penalty`.
- The server merges in the instruct set from `MODEL_CARD_SAMPLING["qwen4exp"]` (a8091f4d / #2685; the merge is in `serve/server.py` around line 6220).
- The request's `temperature: 0` wins, and `sampling_keys` leaves temperature out, so the engine stays greedy.
- It still gets `top_p=0.8 top_k=20 penalty_present=1.5 penalty_last_n=64`.
- With `hist_n > 0`, the `--pipeline-windows` path decodes serially (`src/program/generate.cpp` around 18144-18152). The engine prints that once to stderr.
- We found nothing in the reply or in `/metrics` that shows the card's sampling was applied.

**Same binary, same server, same config. The only change is `"sampling": {"presence_penalty": 0}` in the run config.** The bench is greedy, 16 users x 2 requests x 512 output tokens, mean of 2 runs per column. The two columns come from two windows on the same day.

Stock v0.2.12 binary with our 16-slot config:

| | penalty 1.5 (card default) | penalty 0 | penalty 0 vs 1.5 |
|---|---|---|---|
| agg tok/s, 16 users, ~170-token prompts | 528.7 | 539.0 | +2.0 % |
| agg tok/s, 16 users, ~1.2K prompts | 397.0 | 400.8 | +0.9 % |
| decode per user at 1 / 2 / 4 users | 130.9 / 86.3 / 74.1 | 131.2 / 87.1 / 75.5 | +0.2 / +1.0 / +1.8 % |

On this config, stock's default decode path is barely gated. At 1 user, decode with MTP drafting does not change (130.9 vs 131.2 tok/s), so we do not see MTP drafting switched off here. We did not test qwen35moe or HIP, so we cannot confirm or rule out the slowdown reported in this issue for those.

**Paths that refuse penalties.** Our out-of-tree branch has a solo pipeline and spec lanes. Both build on `--pipeline-windows` and go serial under the same condition. On that branch's binary, only the penalty changed:

| | penalty 1.5 (1 run) | penalty 0 (mean of 2 runs) | change with the penalty |
|---|---|---|---|
| decode per user at 1 / 2 / 4 users | 118.3 / 81.6 / 70.6 | 135.7 / 120.1 / 87.4 | -13 / -32 / -19 % |
| agg tok/s at 16 users | 502.6 | 505.3 | barely changes |

These are not stock numbers. They show how large the cost gets on a gated path at low concurrency. They are in the same range as #2940 (~25 % slower decode).

**Some of upstream's own tools are affected.** These send `temperature: 0` with thinking off and no `presence_penalty`. On v0.2.12 they therefore measure penalised requests, unless the run config sets the penalty to 0:
- `tools/strata_bench.py:141`
- `tools/ab_engine.py:45`
- `tools/piped_read_join_bench.py:68-70`

`strata bench run` (protocol `strata-qwen38-thinking-v1-draft.1`) is not affected. It sends every sampling value on every request, including `presence_penalty: 0.0` (BENCHMARK_PROTOCOL.md, "Settings").

**Suggestions.** Any one of these would help:
1. `temperature: 0` means greedy and adds no card penalties (the #3295 approach).
2. The server logs once when the model card's sampling is applied to a request. The engine's one-time stderr note covers only `--pipeline-windows`.
3. The tools above send `presence_penalty: 0` explicitly, so their numbers do not depend on the card.

Workaround for anyone benchmarking v0.2.12 today: put `"sampling": {"presence_penalty": 0}` in the run config.

**Not tested:** HIP or SYCL, qwen35moe, sampled requests (T > 0) with the penalty, and whether answer quality changes with the penalty or without it.

A full report for this machine, stock against our patched build, will follow as a separate `[Benchmark]` PR. It is not a speed claim for this issue. Most of the 16-user gain over stock as shipped comes from configuration (16 slots, `--prefill 512`, caches). The patched build also loses in some places.
