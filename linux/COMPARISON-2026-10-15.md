# Strata vs vLLM, 4x RTX 3090 vs 1x RTX PRO 6000 (as of 2026-10-15)

Same model family on both engines: Strata serves Unsloth **UD-IQ4_XS**, vLLM 0.31 serves `VnimanieAI/Qwen3.8-Flash-Next-W4A16`
(the only checkpoint vLLM runs). Strata is **0.1.44.1 + our series** (`g3-1441`) on both hosts. One run per cell; greedy
unless noted; thinking off. Sources per cell in the notes; raw data in `results/`.

| | **4x 3090 · Strata** | **4x 3090 · vLLM** | **PRO 6000 · Strata** | **PRO 6000 · vLLM** |
|---|---:|---:|---:|---:|
| **Hardware** | 4x 24 GB, 3.74 TB/s, 251 GiB RAM | same | 1x 96 GB, 1.79 TB/s, 123 GiB RAM | same |
| **1 user, decode** (tok/s) | 114.6 | 79.9 | **221.7** | 116.8 |
| **8 users, short prompts** (total tok/s · TTFT p50) | 394.4 · 0.76 s | 426.7 · 0.42 s | 297.6 · 0.35 s | **541.1** · 0.37 s |
| **16 users, short prompts** (total tok/s · TTFT p50 / p95) | 414.3 · 1.29 / 9.43 s ¹ | 636.5 · 0.62 / 0.73 s | 274.0 · 0.37 / 3.13 s ² | **911.2** · 0.19 / 0.20 s |
| **16 users, 1.2K prompts** (total tok/s · TTFT p50 / p95) | 354.6–400.3 · 1.8–2.3 / 6.7–7.2 s ³ | 466.1 · 1.92 / 5.40 s | 227.5 · 2.14 / 10.19 s ² | **809.6** · 0.83 / 1.19 s |
| **Prefill, one request** (tok/s) | ~3,300 at 1.2K, ~5,100 at 150K ⁴ | ~3,500 ⁴ | 3,241 at 4.8K, 3,387 at 19K | **14,012–14,477** |
| **16 agents x 3 turns** (tok/s · follow-up TTFT p50) ⁵ | 145.5 · **1.16 s** | 52.1 · 45.5 s | 84.1 · 10.24 s | **340.7** · 1.33 s; with 52 GiB RAM KV tier 283.2 · 0.96 s |
| **Coding agent (bench_agentic)**: main / sub / chat TTFT p50 | **2.44 / 1.08 / 0.56 s** | failed ⁶ / 1.57 / 0.63 s | 4.57 / 5.33 / 0.57 s | not run |
| … TTFT p95 main / sub / chat | 6.02 / 4.91 / 1.75 s | – / 9.85 / 1.18 s | 24.58 / 15.83 / 4.61 s | – |
| … total tok/s · wall time · cached prompt share | 62.2 · 132 s · 79 % | – | 45.6 · 207 s · 74 % | – |
| **Quality** MMLU-Pro (240) / GSM8K (150) ⁷ | **77.9 %** / 94.0 % | 70.4 % / 94.7 % | same model as left | same model as left |
| **RAM used** | ~3–5 GB private + conversation cache (128 GiB here) | ~64 GB pinned (FP8 n-gram table) | conversation cache 32 GiB | ~48 GB pinned, ~110 GiB with the KV tier |

**Reading it**

- **One user**: Strata is the fastest single stream on both hosts (MTP drafting); on the PRO 6000 it is ~2x the 3090 host.
- **Many short requests**: vLLM wins clearly (637 vs 414 tok/s on 4x 3090, 911 vs 274 on the PRO 6000). Strata on one
  card does not scale past 8 rows per verify window; the 3090 host gets its throughput from 4 windows in the layer pipeline.
- **Agentic work with long, growing contexts**: Strata on the 4x 3090 host has the shortest follow-up waits (1.2 s vs
  45 s for vLLM on the same host, whose prefix cache did not hold 16 conversations next to the weights). On the PRO 6000
  vLLM's 492K-token VRAM KV pool and 14K tok/s prefill make it faster than Strata (1.3 vs 10.2 s).
- **Quality**: the W4A16 checkpoint vLLM needs scores 7.5 points lower on MMLU-Pro than Strata's UD-IQ4_XS.

**Notes**

1. Production idle at 06:05 (`sweet-short`). The run after the restart (`cbs1441-short`, 312.4 tok/s) was disturbed by
   cold production reads. On 2026-10-14 the 0.1.43 engine on an isolated bench server gave 516.7 tok/s (TESTLOG #76,
   text only): the 0.1.44.1 production number is not yet measured on an isolated server.
2. 16 slots + `--batch-mtp` (`S3`). With 8 slots (`S2`): 292.8 / 241.0 tok/s, but TTFT p95 28.6 / 36.1 s.
3. Two runs of the same build: 354.6 (idle, `sweet-long`) and 400.3 (after the restart, `cbs1441-long`).
4. 3090 Strata: TESTLOG #58 (1.2K read 361 ms) and #73/#74 (150K-170K in 29-33 s); 3090 vLLM: README (2026-10-10).
5. `bench_multiturn.py`, 8K document + 3K per turn, 400 tokens out, T = 0.7 (tok/s depends on reply lengths).
6. All 20 main-agent requests got HTTP 400: vLLM ran with `--max-model-len 32768`, the main agent grows past it. Rerun
   with 131072 pending; sub-agent and chat rows are valid.
7. Same models on both hosts (README "Quality", 2026-10-10).

Open: 4x 3090 Strata 0.1.44.1 on an isolated bench server; 4x 3090 vLLM with 131K context and the RAM KV tier;
PRO 6000 vLLM on bench_agentic.
