## Task A: what v0.2.12 does differently from our production setup

I did not change anything. Sources I read: all release notes v0.1.45 .. v0.2.12 (saved to `/tmp/claude-1000/-media-raid0-StrataQ38FN/07837ea6-13aa-4738-916e-a62b73a0cbb0/scratchpad/inv/releases.md`), `git show v0.2.12:<file>` (tag v0.2.12 = 146df3c4), our engine 3af0006d, our server 22f4781f, the production log, and the g3 proxy source. Server copies are in the same `inv/` folder: `server-0212.py`, `server-ours.py`, `server-base.py`, `drafting.py`. All paths below are v0.2.12 unless they say otherwise.

### Summary
- **One name means two features: `STRATA_PARK_ASYNC`.** In v0.2.12 it is linxuhao's VRAM-stage park (f73b3c86, #3779). Our serve-loop capture (b1daa17a) was never ported and v0.1.45 has no `PARK_ASYNC` at all. If we keep `STRATA_PARK_ASYNC=1`, upstream's VRAM stage turns on and ours is gone.
- **Upstream sets defaults we have never measured.** These go on for us without any action on our side:
  - The server adds five engine variables (PARALLEL_STACK_ENV plus the GDN stack). Three of the five are new to our engine.
  - The engine turns on a table of nine sm_86 switches.
  - The FP16 weight cache comes on with `--batch >= 2`.
  - The int8 prompt attention (IMMA) comes on for cc >= 8.0.
  - Two of these change the answers' bits: `STRATA_GDN_TC=1` with `STRATA_GDN_CONVL2=2`, and IMMA. That breaks our identity A/B against production unless we set them to 0.
- **The image bug in batch mode is still there.** v0.2.12 keeps one M-RoPE table per GPU (generate.cpp:5256-5271, 5817-5823, 15862-15952). #1242 was closed by the tracker reset on 10-19 and a tester reproduced the bug on v0.2.2. The upstream server has no image gate. We must keep `STRATA_MROPE_SLOTS` and our server's image handling.
- **The glibc arena problem is not fixed upstream.** There is no `mallopt` or `malloc_trim` in src/serve. #2934 and #3115 were closed on 10-19 by the tracker reset; PR #2253 is still open. We keep `MALLOC_MMAP_THRESHOLD_` / `MALLOC_TRIM_THRESHOLD_` in the config env. `child_env` passes them through (server.py:4357-4358).
- **Of our 46 switches, 20 exist upstream.** The other 26 (our ll/borrow/tier/lanes/overlap series and the server admission switches) are ours only and must come with the merge.

---

### 1. CLI flags and config keys

**Engine flags.** Every flag in production.json "args" is still parsed in generate.cpp:

| Flag | Line(s) |
|---|---|
| --pack | 3393 |
| --native | 3365 |
| --expert-profile | 3727 |
| --expert-cache auto | 3526-3529 |
| --prefill | 3542 |
| --spec | 3573 |
| --spec-min-p | 3595 |
| --mtp | 3588 |
| --max-context | 3413 |
| --kv | 3448 |
| --kv-resident | 3449 |
| --resident-experts | 3735 (still sets mmap + resident cpu + pin + soft, 4 GiB headroom) |
| --trim-stage-weights | 3576 |
| --ple-io | 3443 |
| --aux-cpus | 3498 |
| --conversation-cache-mib / -slots / -min-free-mib / -min-tokens | 3606-3626 (now strictly range-checked, exit 2 on a bad value) |
| --vision | 3603 |
| --prompt-cache | 3605 |
| --prompt-cache-every | 3630 |
| --batch | 3574 |
| --layer-split | 3598 |

None was renamed or removed.

Defaults that changed around these flags:
- **`--prefill`:** 0.1.45 raised the default chunk and 0.2.11 raised the bare `auto` ceiling to 32768. Neither applies to us, because we pass `--prefill 512` (generate.cpp:3542-3555, 3891-3897).
- **Default context:** 0.2.10 made 131072 the default. We pass 262144.
- **VRAM reserve:** 0.2.1 lowered it to 512 MiB on a card that drives no display. The log shows "700 MiB reserved", which is the vision value.
- **New flags that might help us:** `--conversation-cache-park-gain`, `--memory-limit-mib`, `--session-min-free-mib` (3606-3626), `--prompt-cache-root` (3631), `--pool-cores` (3497), `--vram-reserve-later-mib` (3539).

**Top-level config keys.** `exe`, `args`, `cwd`, `tokenizer`, `model_name`, `log`, `lib_dirs`, `port`, `gpu`, `gpus_asked`, `layer_split`, `open_browser`, `parallel`, `env`, and the `vision` keys (`exe`, `mmproj`, `model`, `gpu`, `threads`, `max_tokens`) are all still read.
- **`layer_split`:** `layer_split_value` / `layer_split_of` (server.py:3669-3714); passed on at 3841-3842.
- **`parallel`:** `parallel_args` (3871-3918). Above 8 on a layer split it still passes `--batch 16`.
- **`env`:** `child_env` (4344-4361).

New keys that take effect when absent:
- **`drafting`** (default auto), applied in `engine_args` (server.py:3793-3808) through `strata_control/drafting.py:1073 apply()`. It is inert for us: `blocked_by` (drafting.py:795-811) returns True for slots > 1, so `--mtp` stays and DFlash2 is not used. Our `--spec 4` is kept, because only the DFlash2 branch strips `--spec`.
- **`vram_fit.fit_start`** (server.py:3814-3821) returns unchanged on a layer split (strata_control/vram_fit.py:289).
- **Split engine:** only when `exe` starts with `strata-split` (server.py:1152). Auto selection happens in `strata launch` and the app, and only for exactly 2 NVIDIA cards or 4 R9700. It does not apply to us.
- **`tool_call_recovery`:** the default flipped from False (ours, server-ours.py:3923 / 7917) to **True** (server.py:4885, 9866). Put `"tool_call_recovery": false` in the config to keep current behaviour.
- **`undeclared_tool_calls`:** new, default `text` (server.py:4884, 9853). Also settable as `STRATA_UNDECLARED_TOOL_CALLS`.
- **New but inert for us:** `user_args` (3789), `expert_profile_save` (3960), `pool_cores`, `vram_elastic`, `split_skip_if_fits`.

**Build.** `STRATA_MMQ_TAB` is still a build option, OFF by default (CMakeLists.txt:151). The prod build must keep passing `-DSTRATA_MMQ_TAB=ON`. The pinned llama.cpp (3cf03257) and the MMQ_TAB hash (CMakeLists.txt:2432) are unchanged, so the tab path still configures. The default CUDA arch list is "86-real;89-real;120" (CMakeLists.txt:233). KQUANTS and Q6K stay ON (134, 138).

### 2. Env switches: production.json "env" and the systemd server env

**What the server adds on its own (`child_env`, server.py:4344-4361).** After the config env is applied, `parallel_stack_env` (4299-4310) does `setdefault` with PARALLEL_STACK_ENV (4269-4273). Conditions: Linux, parallel >= 2, not HIP, SYCL or DeepSeek.
- PARALLEL_STACK_ENV holds `CKPT_POOL=1`, `CKPT_POOL_READY=4`, `SLOT_CKPT_MOVE=1`, `SLOT_COPY_DIRECT=1`, `SLOT_COPY_BATCHSYNC=1`, `BATCH_GDN_GROUPS=1`, `BATCH_ATTN_ROWS=1`, `BATCH_SELF_COMMIT=1` and `QSA_SLOTS=1`.
- `gdn_stack_env` (4325-4341) adds `STRATA_GDN_TC=1` and `STRATA_GDN_CONVL2=2`. It applies when `arch` is absent, which it is for us (= qwen4exp).
- **To override:** put `"KEY": "0"` in the config env, or set it in the process env. `setdefault` keeps any value already present.
- **New to us:**
  - `STRATA_CKPT_POOL_READY=4` (generate.cpp:384-394, linxuhao ca378670): 4 x ~118 MB ready images.
  - `STRATA_BATCH_GDN_GROUPS` (verify.cpp:1754-1757) and `STRATA_BATCH_ATTN_ROWS` (verify.cpp:2060-2066). Both are from yi-sudo #2234 (f629fb04, already in v0.1.45, not in our branch). ATTN_ROWS steps aside for KV-streamed rows (kv_mode 1), so it does nothing past `--kv-resident 28672`.
  - `GDN_TC` / `GDN_CONVL2=2` change the bits (KL 0.0019). They run on sm_86 through the 99-KB split plan (kernels.cu:3081, 3118-3129) for chunks of 128+ tokens, so they are active at our chunk of 512.

**What the engine sets on its own:**
- **sm_86 table** (nv_defaults.cpp:26-40, applied at generate.cpp:3862-3876 when all visible cards share one compute capability). It sets `ADAPT_NOWAIT=1`, `ADAPT_LAG=2`, `CPU_IQ_ALL_EXACT1=1`, `SIDE_MIXER=2`, `ATTN_DAILY=1`, `GR_UP75=1`, `GU_SWIGLU=1`, `QSA_ROWS=1` and `B6_MMVQ_ROWS=1`.
  - Upstream calls them exact: identity gate on a 3060 and an A4000, +10-13% decode for one request alone.
  - None of these switch names appears in our branch (git grep in 3af0006d: 0 hits), so we have never measured them.
  - All experts are on the GPUs (log: "GPU 0 hits", 0 RAM blobs), so ADAPT and CPU_IQ should do nothing. `SIDE_MIXER=2` was slower on gfx1201.
  - Off switch: `STRATA_SM86_PERF=0`.
- **FP16 weight cache:** on whenever `--batch >= 2`, with no layer-split exception (generate.cpp:8193-8197, gemm.cu:1405-1430). It keeps 3 GiB free and is exact, measured +3.8..6.6%. Our stages have about 1.6-2.6 GiB left after the expert cache (log: CUDA0 16.85 GiB free, cache 14.27; CUDA3 room 14.63, cache 13.00), so it will probably cache little or nothing. Off: `STRATA_PF_W16_CACHE_MIB=0`.
- **AUTO_NOBORROW:** single GPU only (generate.cpp:8158-8191), so it does not apply.
- **`STRATA_PROMPT_ATTN_IMMA`:** on for cc >= 8.0 (sm120_series.hpp:96). It changes the bits; upstream measured +1.1..1.8% prompt on 2x A4000. Our branch does not have this switch.
- **Off everywhere:** `STRATA_ROUTE_PRIOR` (sm120_series.hpp:78) and `STRATA_PREFILL_LAYER_MAJOR` (sm120_series.hpp:131).
- **`STRATA_DEFAULTS=026`** takes back the 0.2.7+ defaults in one step (defaults_026.hpp). It does not cover the sm_86 table, the W16 cache or the server stack.

**Our production switches, one by one:**

| Switch | In v0.2.12 | Notes |
|---|---|---|
| MMQ_DIRECT | moe_mmq.cu:661 | same, still needs MMQ_TAB |
| MMQ_GROUP_SORT | mmq_resident_sort.hpp:28 | same |
| SLOT_COPY_DIRECT, SLOT_COPY_BATCHSYNC | generate.cpp:14275 / 425 | now server default |
| PIPE_BYIELD_NODRAIN | generate.cpp:350 | same, opt-in |
| ADMIT_MERGE_FIRST | generate.cpp:362 | same, opt-in |
| CKPT_POOL | generate.cpp:377 | now server default; pool cap = max(2, READY) |
| CKPT_THREADS | generate.cpp:401 | same, opt-in |
| **SLOT_CKPT_MOVE** | generate.cpp:414 | 0.1.45 notes said "do not use"; since 0.2.9 (dd471d31, 10-30) it is in the default stack "qualified bit-exact (QUAL-PAR)". docs/BATCHING.md:118 notes that a SAVE or park right after an admission lacks the moved checkpoint. Our TESTLOG #90 found the failure was a counting artifact of the test, so keeping it is consistent with upstream. |
| SPLIT_MTP_BATCH | generate.cpp:12518, 17394 | same; it also turns PIPED_CONCURRENT off (batch_read.hpp:44-55) |
| BATCH_SELF_COMMIT | verify.cpp:329, generate.cpp:6976 | now default |
| QSA_SLOTS | verify.cpp:335 | now default |
| NAN_GUARD | verify.cpp:5084-5089 | default on, 0 turns it off; same |
| HANDOFF_PIPE | prefill.cpp:485 | same, 2..16 |
| **PARK_ASYNC** | conversation_snapshot.cpp:235, generate.cpp:371 | **different feature** (see summary). Upstream: CUDA only, a VRAM stage while 1.5 GB stays free, background 4 MiB pieces, plus `PARK_VRAM_MIB` (conversation_snapshot.hpp:80-104). Rename ours or drop it from the config before testing v0.2.12. |
| KV_PREFETCH | prefill.cpp:3346 | same (from our base) |
| GR_FAST | fused_gr.cu:2628 | same (unset = per-card default) |
| GR_NORM_FAST | fused_gr.cu:1531 | same, opt-in on CUDA |
| MMVF_ROWS | native_bf16.cu:390 | same |
| SERVE_STOP_BOUND | generate.cpp:18076 | upstream #4677 (243fdb21), on by default, the commit we ported as f81292e1 |

**Ours only (must be carried in the merge):** SLOT_PARK, CONV_TIER_DIR, CONV_TIER_WARM, CONV_TIER_MIN_TOKENS, CONV_TIER_WRITE_GB_DAY, CONV_TIER_CAP_GB, MROPE_SLOTS, SLOT_CHAIN_REUSE, GIVEBACK_CHECKS, MSG_CKPT_TAIL, QSA_SLOTS_STREAM, ATTN_VBATCH, GR_DOWN_PF, EARLY_SAMPLE, PIPE_OVERLAP, PIPE_OVERLAP_KVS, SPEC_LANES, SOLO_PIPELINE, READ_YIELD, PARK_BORROW, CONV_MMAP, GATE_EVICT.

**Systemd server env:**
- `STRATA_ADMIT_SHORT_FIRST`: upstream has it (server.py:356-362, 2201-2209). Its length is the total prompt; ours ranks by estimated new tokens when `ADMIT_BY_NEW` is on.
- `STRATA_ADMIT_BY_NEW`, `STRATA_YIELD_SHORT`, `STRATA_PREFIX_DIAG`: ours only.
- `STRATA_YIELDS_MAX`: upstream has a constant 2 (server.py:2073), not an env variable.

**Overlaps and conflicts:**
- **`STRATA_PIPED_MTP`** (generate.cpp:6778, batch_groups.hpp:61-117) only acts with `--batch-mgtp`... more precisely, only with `--batch-mtp`, which we do not use (we draft through SPEC_LANES and SOLO_PIPELINE). Opt-in, no conflict.
- **`PIPED_CONCURRENT`** stays off beside `SPLIT_MTP_BATCH` or a staged source (batch_read.hpp:44-55). #4972 (994029b0) now leaves IO_PREFETCH and the file tier off when every expert is on the cards, so one of those blockers may disappear.
- **Batch groups:** 4 cards with 16 slots still default to 4 pipelined groups. The 2-card serial and pipeline rules do not touch us (batch_groups.hpp:93-117).
- **Upstream #2526 vs our SLOT_PARK** (01e404c1, on by default): it parks an idle slot's decoded reply before an admission overwrites it, and it splits `park_current` into `park_source` and `park_slot`. That is a code conflict with our SLOT_PARK / borrow / READ_YIELD. It is inert for us, because it skips when "the drafter is on without --batch-mtp".

### 3. Upstream features since v0.1.45 that could help us, with risk

**Opt-in or new, and exact or bit-equal:**
- **`STRATA_IQ4_W4A8=1|2`** (0.2.12, cctan 7bf5ec4e; moe_iq4_w4a8.cu:442, docs/DETAILS.md:1082). An int8 tensor-core kernel for the IQ4_XS / IQ4_NL prompt expert products, same bits as MMQ, sm_80-88. Upstream reports kernel times on down products +9..+70% on sm_86 against MMQ, which presumably means faster. Our model is UD-IQ4_XS, so this is a direct prompt lever. Risk: low (exact, declines what it cannot cover), but its interplay with our `MMQ_DIRECT` pointer table needs a test.
- **`BATCH_GDN_GROUPS` + `BATCH_ATTN_ROWS` + `CKPT_POOL_READY=4`.** This is the part of the 0.2.9 stack we do not have. Upstream: whole stack +16% at c=8 on 2x A6000. Exact. RAM: 4 x 118 MB.
- **sm_86 table** (0.2.0, exact): +10-13% decode for one request alone on a 3060 or A4000. Unmeasured with 16 slots, lanes and overlap.
- **FP16 weight cache** (0.2.10.1, exact): probably no room on our cards.
- **`STRATA_SELECT_TWO_PASS=1`** (0.2.12, exact): +1.65% at 128K. Interesting for long agent contexts.
- **`STRATA_ADAPT_FETCH`** and the `PCIE_*` share rules: only matter with expert misses, so none for us.

**Changes the bits (would need KL or a quality check):** `GDN_TC=1` with `GDN_CONVL2=2` (prompt +4.4..5.7% at 4-32K), `PROMPT_ATTN_IMMA` (+1.1..1.8% prompt), `KVA=1` (lossy, not recommended), `STRATA_PF_FUSED`.

**Correctness and robustness fixes our engine does not have** (we ported only 53f72571 and 243fdb21):
- f8ff3420 #1503: never park one conversation with another's retained K/V.
- c7a0cef1 #4420: parking an idle batch slot keeps the shared-prefix checkpoint (round-2 first token 3.3 s to 0.66 s).
- 3478382e #3350/#3470: copy streams and events go back to a pool. This was a leak that shrank later-card VRAM on a layer split.
- 8c030148 #2409: a restore with no room for a copy stream falls back.
- 587a4744 #4518: the PLE lock is given back when it starves the conversation cache. Relevant with `--ple-io ram`.
- f225844d #5108: the MMQ row-table over-read that could fault long CUDA prompts.
- 40849df0 (#5096, #4844): transfer completion and lost adaptive swaps.
- 20885f7a #5158 and 994029b0 #4972 (file-tier defaults off when all experts are resident).
- abf2953b #3559: our HC_Q8 trimmed-split fix, now upstream (-414 MiB per card).

**Conversation cache and RAM:** PARK_ASYNC, PARK_VRAM_MIB, CKPT_POOL_READY and CKPT_GATHER are linxuhao's (#3779). The arena problem is not fixed upstream, so keep the MALLOC env. Park reserve 53f72571 is upstream and ported. Stop bound #4677 is upstream and ported.

**Vision:** the per-slot M-RoPE fix is still missing upstream (#1242 not fixed), so ours stays. 0.2.11 restricts image formats to JPEG/PNG/GIF/BMP/WEBP/TIFF/AVIF; we already ported this as 610e66bc. The vision config keys are unchanged (server.py:3271-3285, 3831-3839).

**Not for us:** the all-cards split engine (2 NVIDIA cards or 4 R9700 only), DFlash2 (one request at a time; Auto keeps MTP with parallel), split-serve, peer mode, layer-major prefill (off).

### 4. Server API changes that affect our clients

**Routes.** v0.2.12 adds `/v1/telemetry` (server.py:7905) and `/v1/v1/messages*` aliases (293). Upstream has no **GET /v1/load**: that is ours (44d8ff4b / 3850a79e, server-ours.py:6399). Upstream only has POST `/v1/load` and `/v1/unload`, now JSON-only with an origin check (server.py:8262, 8474). Our deploy and window scripts (linux/deploy-*.sh, window-*.sh, prefix_fanout.py) and the vllm_facade tests poll GET /v1/load. They would break unless we carry `load_view`.

**g3 proxy on :8000** (`/home/administrator/G3_VLLM`): it calls Strata (:8090) only through `/v1/models` and `/v1/chat/completions` (server_external.py), and scrapes `/metrics` for `vllm:*` names (server_analytics.py:345-367). `serve/prometheus.py` is identical between 22f4781f and v0.2.12, with the same vllm metric names. Our extra `/metrics` items would disappear without our server patches: `conversation_cache.tier` (4341ea6f) and the borrow/restore counts (f8fcc276).

**Responses API:** small changes only. max_output_tokens may be 0 or -1 (#4036), timings were added (#3291), and LaTeX-safe JSON recovery is opt-in (responses.py +21/-13 against ours).

**Tool calls** (affects agents and Genesis):
- Recovery is on by default (98.3% vs 92.9% calls read as intended).
- More opener aliases: `<function:NAME>`, `<toolcall>`, `tools.NAME`, the MCP last segment.
- A call for an undeclared tool stays text by default; a tool already used in the conversation counts as declared.
- Fenced or indented calls are not run.
- Streamed closing-tag holds (#5040).
- Forced `tool_choice` with several tools (#4828); Anthropic `tool_use_id` (#4832).
- Anthropic overflow wording (#4838); a 60 s stalled-client disconnect (#4271, already ported as 6ceb432e).

**Sampling:** repeat, frequency and presence penalties now count only the reply's tokens (#4590). `STRATA_PENALTY_PROMPT=1` brings back the old window.

**Slot routing:** a branching follow-up now goes to its chat's slot (#2518, bae1e1d4). This overlaps our lanes logic (bda03b22).

### Settings to pin before the first v0.2.12 A/B tonight
To start like with like, put these in the config env:
- `STRATA_GDN_TC=0`, `STRATA_GDN_CONVL2=0` and `STRATA_PROMPT_ATTN_IMMA=0`, until KL is checked.
- `STRATA_SM86_PERF=0` and `STRATA_PF_W16_CACHE_MIB=0` for the baseline arm.
- `STRATA_BATCH_GDN_GROUPS`, `STRATA_BATCH_ATTN_ROWS` and `STRATA_CKPT_POOL_READY` set explicitly, 0 or 1 per arm.
- Our PARK_ASYNC renamed or removed.
- `"tool_call_recovery": false` if client behaviour must stay the same.

Then measure the new defaults as separate arms (stack, sm86, IMMA, GDN, IQ4_W4A8).
