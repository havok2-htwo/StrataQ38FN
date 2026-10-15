# StrataQ38FN on one RTX PRO 6000 Blackwell (96 GB)

Measurements from 2026-10-15 on a second Linux host: **one RTX PRO 6000 Blackwell Workstation Edition** (96 GB, sm_120,
driver 595.91), 12 CPU threads, 123 GiB RAM. Model **Unsloth UD-IQ4_XS** for Strata, `VnimanieAI/Qwen3.8-Flash-Next-W4A16`
for vLLM. Engine: Strata **0.1.44.1** (`Strata-1441`, our g3-1441 tree) built for sm_120, server from the same tree.
Compared with the 4× RTX 3090 production host (README.md, TESTLOG #67/#71/#74/#76) and vLLM 0.31.0 on the same card.

Short version: on one card Strata is the fastest single stream (222 tok/s, MTP) but tops out at **~300 tok/s total**
with many users; more slots than 8 do not add throughput, they only spread it (shorter waits, slower replies). vLLM 0.31
on the same card reaches **911 / 810 tok/s** at 16 users (short / 1.2K), prefill **~14,000 tok/s**, at lower quality
(MMLU-Pro 70.4 vs 77.9 %, README "Quality").

## Results

All runs: one run per arm, greedy unless noted, thinking off. Raw data in `results/sweep-rtxpro6000.jsonl` (+
`.requests.jsonl`), `results/multiturn-rtxpro6000.jsonl` and `results/official-bench-rtxpro6000/`.

### Concurrency (bench_concurrency.py: 2 requests per user, 512 tokens out, unique prefix)

Total tok/s · decode per user p50 · TTFT p50 / p95 s.

| Users | Strata, 8 slots (`S2`) | Strata, 16 slots + `--batch-mtp` (`S3`) | vLLM 0.31 W4A16 | 4× 3090 Strata prod (cb144, #76) |
|---|---|---|---|---|
| 1, short | **221.7** · 235 · 0.13 / 0.13 | – | 116.8 · 118 · 0.06 / 0.06 | 116.2 · 122 · 0.20 / 0.20 |
| 8, short | 297.6 · 38 · 0.35 / 1.38 | – | **541.1** · 71 · 0.37 / 0.62 | 399.4 · 55 · 0.65 / 1.66 |
| 16, short | 292.8 · 38 · 1.44 / 28.59 | 274.0 · 18 · 0.37 / 3.13 | **911.2** · 58 · 0.19 / 0.20 | 516.7 · 34 · 0.89 / 3.10 |
| 1, 1.2K | **183.8** · 220 · 0.46 / 0.46 | – | 115.6 · 118 · 0.10 / 0.10 | – |
| 8, 1.2K | 242.7 · 33 · 0.76 / 4.78 | – | **521.5** · 70 · 0.58 / 0.60 | – |
| 16, 1.2K | 241.0 · 33 · 9.66 / 36.10 | 227.5 · 16 · 2.14 / 10.19 | **809.6** · 55 · 0.83 / 1.19 | 406.5 · 28 · 1.91 / 6.61 |

**Slot scaling** (Strata, `--batch-mtp`, users = slots, short prompts): 8 slots 297.6, 16 slots 274.0, 18 slots 275.9 tok/s
(`S5`, TTFT p95 3.42 s, decode 16 tok/s per user). It does not scale past 8: one GPU runs one verify window of at most
8 rows (`kVerifyMaxT`) at a time, ~23 ms per window (engine log: 245–320 rows/s). The 4× 3090 host gets its 517 tok/s
from four such windows in flight in the layer pipeline. The 20/24/32-slot arms were not run (stopped on request after 18).
16 slots are still the better default for 16 users: same total, but the p95 wait for the first token drops from 28.6 s
to 3.1 s.

The official bench with `--clients 8` (128 tokens each, see below) shows the same ceiling: 210 tok/s total, because 8
admissions ~0.24 s apart dominate a 128-token reply.

### Prefill (one request, unique prompt, 16 tokens out; tok/s = actual prompt tokens / client TTFT p50)

| Prompt | Strata (`S3`) | vLLM 0.31 |
|---|---|---|
| 4,764 tokens | 3,241 tok/s (1.47 s) | **14,012 tok/s** (0.34 s) |
| 18,965 tokens | 3,387 tok/s (5.60 s) | **14,477 tok/s** (1.31 s) |

The official bench (engine timings, solo config `S1`) reports 4,012 tok/s at 4K and 4,134 tok/s at 16K. For reference the
4× 3090 host: 1.2K read 361 ms ≈ 3,300 tok/s (#58), 150K–170K TTFT 29–33 s ≈ 5,100–5,200 tok/s (#73/#74). Strata's
prefill rate grows with length, so the PRO 6000 at 150K is not known; not measured here (configs are 32K).

### Agents (bench_multiturn.py: 16 users × 3 turns, 8K document + 3K per turn, 400 out, T = 0.7, as cb144)

| | tok/s | turn-1 TTFT p50 / p95 | follow-up TTFT p50 | cached (turn 2 / 3) |
|---|---|---|---|---|
| Strata 16 slots (`S3`) | 84.1 | 50.55 / 114.54 s | 10.24 s | 11,382 / 15,859 |
| vLLM 0.31 | **340.7** | 7.19 / 11.81 s | 1.33 s | not reported (prefix cache on) |
| vLLM 0.31 + 52 GiB RAM KV tier | 283.2 | 7.38 / 11.87 s | **0.96 s** | 10,878 / 15,557 (70 / 74 %) |
| 4× 3090 Strata prod (cb144, #76) | 151.9 | – | 1.52 s | – |

Strata admits the 16 first turns one at a time at ~3,300 tok/s (16 × 11K tokens ≈ 54 s for the last one); vLLM reads
them together at ~14,000 tok/s. The tok/s of the two vLLM rows differ by reply length (T = 0.7: 15,105 vs 11,819 tokens
in ~43 s), not by the tier. All 16 conversations also fit into vLLM's 492,612-token VRAM KV pool, so the RAM tier only
shows its value with more open conversations than that.

### Official benchmark (tools/strata_bench.py of Strata-1441, against the running server)

| | `S1` solo | `S2` 8 slots, `--clients 8` |
|---|---|---|
| chat decode tok/s (9 requests) | 224.1 (182.5–287.9) | 219.8 (184.2–300.4) |
| chat TTFT | 0.13 s | 0.13 s |
| 4K prompt tok/s · TTFT | 4,011.8 · 1.01 s | 3,978.4 · 1.02 s |
| 16K prompt tok/s · TTFT | 4,134.4 · 3.90 s | 4,079.9 · 3.95 s |
| agent cache hit · follow-up TTFT | 68.6 % · 1.00 s | 68.7 % · 1.01 s |
| 8 clients total tok/s | – | 210.0 (205.2–210.8) |

Reports as written by the tool (already redacted): `results/official-bench-rtxpro6000/solo/` and `.../clients8/`.

### Projection: 2× RTX PRO 6000 with the layer pipeline (estimate, not measured)

Both hosts do the same batch work per memory bandwidth: 4× 3090 ~654 rows/s on 3.74 TB/s (~175 rows/s per TB/s), one
PRO 6000 ~320 rows/s on 1.79 TB/s (~179). Two PRO 6000 have 3.58 TB/s = 96 % of the 4× 3090 host, with 2 instead of 4
stages and 8-row instead of 4-row groups.

| | 4× 3090 (measured, best) | 2× PRO 6000, pipeline (estimate) |
|---|---|---|
| 1 user decode | 131 tok/s | ~220–235 tok/s (one token still crosses the stages in sequence; one hand-off instead of three) |
| 16 users short / 1.2K | 580 / 456 tok/s | ~500–560 / ~390–440 tok/s |
| prefill | ~5,200 tok/s at 150K | at least the one-card value at the same length, more if the stages overlap |
| VRAM / context | 96 GB, 262K with streamed KV | 192 GB, 262K with all KV resident |

Open: how the pipeline patches behave with 2 stages and `--batch-groups 2`; only 4 stages are measured.

## Setup on this host

1. **Data:** `StrataQ38FN-2026-10-15.zip` (repo folder + `unsloth-UD-IQ4_XS/` + `mmproj/`) unpacked into the env folder,
   model under `StrataQ38FN/model/`.
2. **CUDA:** the host has only the driver, no toolkit. CUDA 13 comes from pip into Strata's venv (`nvidia-cuda-nvcc`
   13.4.92, `-runtime`, `-cccl`, `nvidia-cublas`, `nvidia-curand`); `/home/administrator/G3_VLLM/cuda` is a symlink tree
   in the classic layout over it (CMake's FindCUDAToolkit and FlashInfer's JIT need `lib64/libcudart.so` and
   `lib64/stubs/libcuda.so`).
3. **Engine:** `build-engine-sm120.sh Strata-1441` (and `Strata-143`): sm_120, `STRATA_MMQ_TAB=ON`, ggml from
   `Strata/third_party/llama.cpp` (our fork), tests off. The shipped engines in `Strata/engine/` are sm_86 only (no PTX)
   and do not run on Blackwell.
4. **Configs** `configs/S*-rtxpro6000-*.json`: the setup config with this host's paths, `gpu: [0]`, no layer split and no
   `--trim-stage-weights`, no `--aux-cpus` (12 threads here), `--ple-io ram`, `--resident-experts` (experts 55.4 GiB,
   ~64 GB VRAM in total solo, ~75 GB with 16 slots), `lib_dirs` = the pip CUDA lib folder.
   `S1` solo, `S2` `parallel: 8`, `S3`/`S5`/`S6`/`S7`/`S4` `parallel: 16/18/20/24/32` + `--batch-mtp` (needed for more
   than 8 slots on one GPU). The production env switches (`STRATA_MMQ_DIRECT` etc.) are **not** set: an A/B with them is
   still open (B-tab cut prompt reads by 17–40 % on the 3090s, #36/#58).
5. **Start:** from `Strata/`, e.g.
   `systemd-run --user --scope --unit=strata-s3 -p MemoryMax=110G -p MemorySwapMax=0 Strata/.venv/bin/python Strata-1441/serve/server.py --engine strata --config linux/configs/S3-rtxpro6000-batch16-mtp.json --port 8090`.
   Dashboard: `STRATA_SERVE=…/Strata-1441/serve CONFIG=…/S2-rtxpro6000-batch8.json SOURCES=http://127.0.0.1:8090 ./start-dashboard.sh`.

### vLLM 0.31.0 on one card

- venv `/home/administrator/G3_VLLM/venv` (`pip install vllm==0.31.0`, torch 2.13 + cu130, plus `ninja`);
  `vllm/run_vllm031_flashnext_1gpu.sh` = `run_vllm031_flashnext.sh` with TP=1 (no `--enable-expert-parallel`), the same
  `patch031` FP8 n-gram table (47.7 GiB pinned host RAM, scale 1.995e-4), Marlin WNA16 MoE, port 8003.
  KV cache: 492,612 tokens = 15 × 32K. Model loaded straight from the SMB share (gvfs, ~475 MB/s): ~6 min per start.
- FlashInfer JIT-builds its sampling module at the first request: needs `ninja`, `curand.h` and `CUDA_HOME` in the
  classic layout (the script sets `CUDA_HOME=/home/administrator/G3_VLLM/cuda` and puts the venv on `PATH`).
- RAM KV tier: `EXTRA="--kv-offloading-size 52"` needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (the
  offloading connector rejects expandable segments); host RAM then ~110 GiB in use. `--enable-prompt-tokens-details`
  makes vLLM report `cached_tokens`.

## Reproduce

```bash
cd linux
LABEL=strata-s3 URL=http://127.0.0.1:8090 vllm/bench_1gpu.sh        # sweeps + prefill probe against a running Strata
MODEL=<W4A16 snapshot> vllm/run_vllm031_flashnext_1gpu.sh           # then:
LABEL=vllm031-1gpu VLLM=1 vllm/bench_1gpu.sh
python3 -I bench_multiturn.py --url http://127.0.0.1:8090 --label strata-s3-mt --users 16 --turns 3 \
  --out results/multiturn-rtxpro6000.jsonl                          # add --vllm --model qwen3.8-flash-next for vLLM
VLLM_MODEL=<W4A16 snapshot> vllm/slot_scaling_1gpu.sh               # 18/20/24/32 slots + vLLM at those user counts
(cd ../Strata-1441 && ../Strata/.venv/bin/python tools/strata_bench.py --url http://127.0.0.1:8090 [--clients 8])
```

## Limitations

One run per arm (the 3090 noise rule, ab_quick.sh: |Δ| < ~1.3 % is not resolved; the gaps here are far larger). Strata
0.1.44.1 here vs 0.1.43 (`143v2`) in the 3090 production, and without its opt-in switches. vLLM serves a different
checkpoint (W4A16, lower MMLU-Pro) than Strata (UD-IQ4_XS). Prefill rates from client TTFT include HTTP and tokenizer
time. The multi-turn runs use T = 0.7 like cb144, so their tok/s depends on reply lengths.
