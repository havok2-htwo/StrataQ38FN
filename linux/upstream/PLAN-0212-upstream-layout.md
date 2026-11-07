## Task B: code tree, trial merge, build and tests (296b59d2 → v0.2.12)

Everything here was read-only. I made no checkouts, builds or GPU runs in any worktree. The trial merges were `git merge-tree --write-tree`, which only writes objects. Scratch outputs (per-file table, raw merge-tree output, conflict counts) are in `/tmp/claude-1000/-media-raid0-StrataQ38FN/07837ea6-13aa-4738-916e-a62b73a0cbb0/scratchpad/inv/`: `table`, `mt_3af0006d.txt`, `mt_22f4781f.txt`, `conflicts.txt`, `up_ns`, `up_numstat`, `setup_v0212.py`, `cmake_v0212.txt`.

### Bottom line
- **Our files still exist upstream.** Git found no renames or splits for any file we touch. The only file upstream deleted is `docker-entrypoint.sh`, which we never touch. Upstream grew a lot (3766 commits, 3094 files, +575k lines, 2640 new files), but in our areas it edited the existing files in place.
- **The trial merges hit the same 34–35 files with about 265 conflict hunks each:**
  - Engine merge (v0.2.12 + 3af0006d): 34 files, 267 hunks, about 8.6k lines inside conflicts.
  - Server merge (v0.2.12 + 22f4781f): 35 files, 265 hunks, about 7.8k lines.
  - `src/program/generate.cpp` is the main cost: 90 (engine) / 83 (server) hunks and about 4k lines.
- **About half the conflicts come from upstream's own port of our series.** Merging against v0.1.45 alone already gives 21 files / 127 hunks. Hunks per tag: v0.1.45 127 → v0.2.6 147 → **v0.2.7 217** → v0.2.9 231 → **v0.2.10 254** → v0.2.12 267. Only 16 of our 123 commits match upstream by patch-id; the rest were reworded. So the ported parts have to be resolved by content, preferring upstream's version.
- **Build:** an sm_86 / CUDA 13.0 build at v0.2.12 works the way we built `strata-g3-ll10b`. We must add `-DSTRATA_MMQ_TAB=ON` and must pass `-DCMAKE_CUDA_ARCHITECTURES=86`, because the default is now 120. The ggml pin did not change (3cf03257), and the `mmq.cuh` fingerprint in our `third_party/llama.cpp` matches the hash MMQ_TAB checks against, so MMQ_TAB stays usable.
- **Identity tests need a pinned environment.** v0.2.12 switches several settings on by default, some of them not bit-exact (details in section 4). Night identity tests have to set these off explicitly, or the merged build will not match production.

### 1. Files we touch and what upstream did to them
Our engine branch touches 114 files (296b59d2..3af0006d, 130 commits). Our server branch touches 113 (296b59d2..22f4781f, 124 commits; it splits off at bda03b22). Together that is 126 distinct files.

**Hot spots: files that exist at the base and upstream changed heavily (upstream lines added/removed vs ours):**

| file | upstream | ours (engine) | ours (server) |
|---|---|---|---|
| src/program/generate.cpp (14182 → 21496 lines upstream, 19770 ours) | +7894/−580 | +5846/−258 | +4769/−198 |
| serve/server.py (6976 → 10535 upstream) | +3957/−398 | +950/−51 | +1117/−58 |
| src/prefill/prefill.cpp (4362 → 8015) | +3822/−169 | +380/−32 | +367/−23 |
| src/kernels/cuda/qsa_select.cu | +2690/−145 | +68/−11 | same |
| src/core/verify.cpp | +2127/−312 | +329/−64 | same |
| src/kernels/cuda/fused_gr.cu | +1558/−101 | +392/−52 | same |
| docs/DETAILS.md | +1377/−1786 | +103/−3 | +100 |
| CMakeLists.txt (1793 → 3066) | +1303/−30 | +209/−1 | +193/−1 |
| sycl/src/program/generate.cpp | +970/−70 | +6/−1 | same |
| serve/frontend.py | +722/−79 | – | +61/−18 |
| src/kernels/cuda/qsa_decode_attn.cu | +544/−35 | +207/−51 | same |
| src/core/conversation_snapshot.cpp | +485/−15 | +61/−13 | same |
| src/core/conversation_state.cpp | +388/−22 | +88/−17 | same |
| src/core/conversation_cache_test.cpp | +361 | +417 | +185 |
| src/prefill/moe_mmq.cu | +338/−5 | +182/−1 | same |
| src/core/mtp.cpp | +285/−68 | +40/−13 | same |
| include/strata/core/verify.hpp | +267/−24 | +94/−1 | same |
| src/ngram/ple_reader_test.cpp / ple_reader.cpp | +250 / +111/−34 | +211 / +44/−5 | same |
| include/strata/core/conversation_cache.hpp | +201/−15 | +298/−9 | +150/−4 |
| src/core/conversation_copy.hpp | +194/−9 | +144/−6 | +141/−3 |
| src/kernels/ngram.cpp | +171/−20 | +84/−31 | same |
| include/strata/prefill/prefill.hpp | +162 | +52 | +44 |
| src/core/expert_cache.cpp | +144/−2 | +5/−2 | same |

The remaining existing headers changed by less than 100 lines upstream: mtp.hpp, qsa_select.hpp, moe_mmq.hpp, sampler.hpp, expert_cache.hpp, qsa_decode_attn.hpp, fused_gr.hpp, ngram.hpp, conversation_buffer.hpp, conversation_memory.hpp, metal stubs, sycl moe_mmq / ple_reader / snapshot_test, test_monitor.py, test_effort_paths.py.

**Base files upstream did not change at all** (our changes apply cleanly): batch_rows.hpp, session.hpp, mrope.hpp, native_router.hpp/.cu, ple_reader.hpp, direct_file.hpp/.cpp (and the sycl copy), message_boundary.hpp, conversation_file.cpp and its test, conversation_validation_test.cpp, tests/core/batch_rows_test.cpp.

**Files both sides added** (upstream ported them):
- Identical to our engine branch: `src/prefill/moe_mmq_tab.cuh`, `moe_mmq_tab.hpp`, `mmq_tab_instance.cu.in`, `src/prefill/mmq_resident_sort.hpp`, `tests/cuda/prefill_mmq_tab_test.cu`, `src/kernels/qsa_slots_parity.cpp`, `tests/core/expert_cache_slot_guard_test.cpp`, `tools/batch_interleave_test.py`, `include/strata/core/serve_stop_bound.hpp`, `tests/core/serve_stop_bound_test.cpp`.
- Different content:
  - `serve/test_admit_order.py`: add/add conflict.
  - `serve/test_parser_call_scan.py`: server branch, add/add conflict.
  - `serve/test_exif_orientation.py`: server branch; it exists at the base, upstream +62.
  - `test_parser_hold.py` and `test_quoted_call_values.py`: identical to our server branch.

**Only ours** (no upstream counterpart, so no conflict):
- Engine core: `conversation_tier.{hpp,cpp}` + test, `conversation_alloc.hpp` + test, `arena_carve.hpp` + test.
- `include/strata/program/`: mrope_regions, overlap_kvs, read_plan, read_sched, read_yield, slot_chain, solo_pipeline, spec_lanes headers, plus their tests in `tests/program/`, and `message_cut_test`.
- Kernel parity tests: `kv_stream_overlap_parity`, `qsa_slots_stream_parity`, `qsa_vbatch_parity`, `route_plan_fuse_parity`, `sampler_same_test`.
- Serve: `serve/prefix_diag.py`, `serve/admit_est.py` (server branch), and the serve tests test_admit_est, test_conv_cache_borrow, test_conv_cache_poll, test_genesis_shapes_replay, test_image_alone, test_image_history, test_load, test_prefix_diag, test_reads, test_read_yield, test_stream_write_timeout.
- Tools: batch_overlap_test.py, batch_test16.py, batch_test32.py, cache_reuse_check.py, msg_ckpt_check.py, vision_batch_check.py, vision_check_fake_engine.py, vision_check_smoke.py.

**Upstream additions close to our features:**
- `include/strata/program/conv_cache.hpp` + `src/program/conv_cache_test.cpp`: a checkpoint retention policy that pins the root and evicts the least recently used. This overlaps our message checkpoint and tier logic.
- `batch_read.hpp`, `batch_groups.hpp`, `draft_kv_plan.hpp`, `split_ring.hpp`, `stage_reserve.hpp`, `window_limits.hpp`.
- `src/core/conversation_copy_streams_test.cpp`.
- `src/core/nv_defaults.cpp`: an sm_86 default table (section 4).
- `src/split/`: the separate opt-in `strata-split` engine (sFNX, `-DSTRATA_BUILD_SPLIT=ON`, off by default; it does not touch our code).

### 2. Trial 3-way merge
Commands: `git merge-tree --write-tree --name-only --messages v0.2.12 <ours>`. Both exit with 1 (conflicts). Result trees: 8b982d1d (engine), 5f7785d0 (server). Columns below are hunks / lines inside conflicts.

| file | engine 3af0006d | server 22f4781f |
|---|---|---|
| src/program/generate.cpp | **90 / 3998** | **83 / 3577** |
| serve/server.py | 29 / 524 | 31 / 598 |
| src/core/verify.cpp | 24 / 326 | 24 / 326 |
| src/kernels/cuda/qsa_decode_attn.cu | 16 / 348 | 16 / 348 |
| src/prefill/prefill.cpp | 14 / 200 | 13 / 184 |
| src/kernels/ngram.cpp | 10 / 112 | 10 / 112 |
| include/strata/core/conversation_cache.hpp | 9 / 337 | 7 / 173 |
| src/kernels/cuda/fused_gr.cu | 7 / 56 | 7 / 56 |
| src/prefill/moe_mmq.cu | 7 / 54 | 7 / 54 |
| src/kernels/cuda/qsa_select.cu | 6 / 806 | 6 / 806 |
| src/core/conversation_copy.hpp | 6 / 171 | 6 / 171 |
| src/core/conversation_state.cpp | 5 / 324 | 5 / 324 |
| include/strata/core/verify.hpp | 5 / 82 | 5 / 82 |
| CMakeLists.txt | 4 / 72 | 2 / 52 |
| src/ngram/ple_reader.cpp | 4 / 65 | 4 / 65 |
| docs/DETAILS.md | 3 / 305 | 3 / 305 |
| src/core/conversation_snapshot.cpp | 3 / 112 | 3 / 112 |
| include/strata/core/conversation_snapshot.hpp | 3 / 51 | 3 / 51 |
| include/strata/core/conversation_buffer.hpp | 3 / 29 | – |
| sycl/src/ngram/ple_reader.cpp | 3 / 26 | 3 / 26 |
| serve/test_admit_order.py (add/add) | 2 / 19 | 3 / 22 |
| include/strata/kernels/qsa_decode_attn.hpp | 2 / 26 | 2 / 26 |
| src/core/conversation_cache_test.cpp | 1 / 297 | 1 / 109 |
| src/core/conversation_memory_test.cpp | 1 / 104 | – |
| src/core/mtp.cpp | 1 / 54 | 1 / 54 |
| include/strata/kernels/ngram.hpp | 1 / 27 | 1 / 27 |
| src/core/conversation_snapshot_test.cpp | 1 / 26 | 1 / 26 |
| mtp.hpp, fused_gr.hpp, native_dense.cpp, metal stubs.cpp, qsa_select.hpp, moe_mmq.hpp, prefill.hpp | 1 each, 5–11 lines | same |
| serve/frontend.py | – | 7 / 60 |
| serve/test_exif_orientation.py | – | 2 / 13 |
| serve/test_parser_call_scan.py (add/add) | – | 2 / 3 |
| **Total** | **34 files, 267 hunks, ~8604 lines** | **35 files, 265 hunks, ~7815 lines** |

These files merged cleanly: conversation_memory.hpp, expert_cache.hpp/.cpp, sampler.hpp/.cu, test_monitor.py, test_effort_paths.py, fused_gr_bench.cpp, gr_parity.cpp, ple_reader_test.cpp, sycl generate.cpp / moe_mmq.dp.cpp / snapshot_test.

**What the conflicts look like:**
- In 20 hunks the upstream side is empty and in 35 our side is empty (engine). Those are adjacency or "already ported" cases and are cheap to resolve.
- The CMakeLists conflicts are side-by-side test and source registrations; the resolution is to keep both. Example: upstream added `nv_defaults.cpp` in the same line where we added `conversation_tier.cpp`.
- The largest generate.cpp hunks: line ~16859 (ours 425 lines vs upstream 0), ~21691 (205 vs 406), ~13457, ~2776, ~23001.

**How conflicts grow by upstream tag** (engine branch; files / hunks / generate.cpp hunks):
- v0.1.45: 21 / 127 / 41. This is mostly the port of our series.
- v0.2.0 to v0.2.6: about 23 / 147 / 45. Flat.
- v0.2.7: 29 / 217 / 68. The 0.2.7 defaults rework, `defaults_026`.
- v0.2.9: 31 / 231 / 74.
- v0.2.10: 32 / 254 / 88. DFlash2 became the default drafter here.
- v0.2.12: 34 / 267 / 90.

Suggested merge order: merge v0.1.45 first, resolving the ported parts toward upstream. Then merge v0.2.6, then v0.2.7, then v0.2.12, and run the host tests after each step.

### 3. Building the CUDA engine at v0.2.12 for sm_86 with CUDA 13.0

**How setup.py builds it** (`build_engine()` → `cmake_build()`):
- Configure: `cmake -G Ninja -DCMAKE_MAKE_PROGRAM=<ninja> -S <repo> -B <repo>/build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=<archs> -DCMAKE_CUDA_COMPILER=<nvcc> -DCUDAToolkit_ROOT=<nvcc/../..> -DSTRATA_GGML_DIR=<llama.cpp>`, plus host compiler settings from CC/CXX/CUDAHOSTCXX.
- Build: `cmake --build build --target strata -j <nproc/2>`. It retries once because CUDA 13.0 ptxas sometimes fails to parse a PTX file it just wrote.
- Output: `build/strata` (and `build/libstrata_kq.so` if built) is copied to `engine/` and recorded in `engine/BUILD.json`.
- Architectures: the cards in use plus every other usable card in the PC (#4662).
- `STRATA_PORTABLE` stays off, so the build is native.

**How the release build does it** (`packaging/build_cuda_linux.sh`, VARIANT=cuda13): the same flags plus `-DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_EXPERIMENTAL_SM60=OFF "-DCMAKE_CUDA_ARCHITECTURES=75-real;86-real;89-real;120-real;120-virtual" -DSTRATA_GGML_DIR=third_party/llama.cpp`, with targets `strata strata_kq`. Neither setup.py nor the release passes **`STRATA_MMQ_TAB`, so it is OFF**: `STRATA_MMQ_DIRECT` and `STRATA_MMQ_GROUP_SORT` have no effect in stock builds. We already noted this in window-020.sh.

**How we built `strata-g3-ll10b`** (from `/media/raid0/StrataQ38FN/Strata-ll10-borrow/build/CMakeCache.txt` and linux/README.md):
- Unix Makefiles generator, Release, nvcc `/usr/local/cuda-13.0/bin/nvcc`, `CMAKE_CUDA_ARCHITECTURES=86`.
- `STRATA_MMQ_TAB=ON`, `STRATA_MMQ_KQUANTS=ON`, `STRATA_Q6K_EXPERTS=ON`, `STRATA_NATIVE_EXPERTS=ON`, `STRATA_PORTABLE=OFF`.
- `STRATA_BUILD_TESTS=ON`, `STRATA_BUILD_CONVERSATION_TESTS=ON`, `STRATA_GGML_DIR` empty (fetches the pinned commit).
- The binary `build/strata` (88 MB) is copied to `Strata/engine/strata-g3-ll10b`. No libstrata_kq.so is deployed next to the engines; production does not need it.

**What changed in v0.2.12 that matters for our build:**
1. **`CMAKE_CUDA_ARCHITECTURES` now defaults to 120** (#4023) when unset. We must pass `86`.
2. **`STRATA_MMQ_TAB` is still OFF by default.** CMake turns it off silently unless the region of `mmq.cuh` from `mul_mat_q_process_tile` to `DECL_MMQ_CASE` hashes to `36e4dc71…3eb0`. The pin is unchanged (3cf03257 in both setup.py and third_party/ggml/VERSION.txt). I hashed the same region in `/media/raid0/StrataQ38FN/Strata/third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh` and got exactly that value. After configuring, check for the line `STRATA_MMQ_TAB on: pointer-table MMQ for q2_0 … q8_0`. Production depends on this because it sets `STRATA_MMQ_DIRECT=1` and `STRATA_MMQ_GROUP_SORT=1`.
3. **New `STRATA_WINDOW_MAX_T`** (auto/8/12/16; CUDA auto = 8 rows per window, 16 = the wide build for 4 groups × 16 slots). We stay at 8 by default; `-DSTRATA_WINDOW_MAX_T=16` is a possible performance lever that costs VRAM.
4. **`STRATA_BUILD_TESTS` now defaults to OFF**, because the default requires a `tests/CMakeLists.txt` that does not exist. Pass it explicitly.
5. cmake ≥ 3.24 is required; the box has /usr/bin/cmake 3.28.3, the venv has 4.4.3, and ninja is installed. The CMake project version is now 0.1.58, the engine version setup checks.

**Proposed build for the merged tree**, in a new worktree. Nothing has been run. The CPU pinning is my suggestion, because production uses `--aux-cpus 65-127`.
```sh
WT=/media/raid0/StrataQ38FN/Strata-212m   # merge branch worktree (Strata-212 @146df3c4 already exists, clean, no build)
taskset -c 0-63 nice -n 10 cmake -G Ninja -S $WT -B $WT/build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_MMQ_TAB=ON -DSTRATA_GGML_DIR=/media/raid0/StrataQ38FN/Strata/third_party/llama.cpp \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
taskset -c 0-63 nice -n 10 cmake --build $WT/build --target strata -j 48   # then: --target all for the tests
cp $WT/build/strata /media/raid0/StrataQ38FN/Strata/engine/strata-g3-212m
```
For a "stock v0.2.12" comparison arm, build the same command without `STRATA_MMQ_TAB` in `Strata-212` (that matches the setup.py release flags), as we did for Strata-020-stock.

### 4. New v0.2.12 defaults that break identity with production
Production's `"env"` sets some of these switches, but not all.

- **sm_86 default table** (`src/core/nv_defaults.cpp`, applied in the engine at start unless the variable is already set; `STRATA_SM86_PERF=0` turns it off). It applies to the 3090s: ADAPT_NOWAIT=1, ADAPT_LAG=2, CPU_IQ_ALL_EXACT1=1, SIDE_MIXER=2, ATTN_DAILY=1, GR_UP75=1, GU_SWIGLU=1, QSA_ROWS=1, B6_MMVQ_ROWS=1. Upstream says these are bit-exact.
- **Server `PARALLEL_STACK_ENV`** (any config with parallel ≥ 2, Linux CUDA): CKPT_POOL=1, CKPT_POOL_READY=4, **SLOT_CKPT_MOVE=1**, SLOT_COPY_DIRECT=1, SLOT_COPY_BATCHSYNC=1, BATCH_GDN_GROUPS=1, BATCH_ATTN_ROWS=1, BATCH_SELF_COMMIT=1, QSA_SLOTS=1. Our memory note says SLOT_CKPT_MOVE is "do not use", and production currently sets it to 1.
- **Server `GDN_STACK_CUDA`: GDN_TC=1, GDN_CONVL2=2.** These are not exact (upstream reports KL mean 0.0019).
- **Engine 0.2.10 defaults:** `STRATA_PF_W16_CACHE_MIB` (FP16 weight cache, exact) and `STRATA_AUTO_NOBORROW` (not exact, KL 0.006; only when every expert is resident).
- **0.2.7 defaults:** GR_UP_WIDE, GR_V3, PROMPT_ATTN_IMMA (int8 prompt attention), PLE_AUTO_RAM. `STRATA_DEFAULTS=026` turns them all back to the 0.2.6 behaviour.
- **v0.2.10 made DFlash2 the default drafter.** Production passes `--mtp`; check that it still selects MTP.

For identity runs, pin the baseline arm with `STRATA_DEFAULTS=026 STRATA_SM86_PERF=0 STRATA_GDN_TC=0 STRATA_GDN_CONVL2=0 STRATA_AUTO_NOBORROW=0 STRATA_PF_W16_CACHE_MIB=0` plus production's own env. Then test each new default as its own performance arm.

### 5. Tests that need no GPU
The rule says no tests before 23:45. Whether these CPU-only runs count as "testen" is your call, so I list them separately.

- **ctest, host-only.** At v0.2.12 there are 288 ctest entries; by if-nesting I count about 107 host and 181 GPU. GPU tests return 77 (skip) when no card is visible; there are 64 `SKIP_RETURN_CODE` uses.
  - Upstream host tests include: conversation_cache/memory/file/split_failure, session_save_reclaim, conv_cache_test, batch_rows_test, batch_groups_test, batch_read_test, split_rules/split_walk/split_ring, window_limits_test, window_evict_test, window_rule_test, serve_stop_bound_test, spec_admission_*, slot_history_test, draft_kv_plan_test, nv_defaults_test, arch_defaults_test, message_boundary_test, prefill_plan_test, stage_reserve_test, native_arithmetic_parity*, tq_cpu_parity, pool_*_test, and others.
  - Our host tests: arena_carve_test, conversation_alloc_test (+ `_off`), conversation_tier_test, message_cut_test, mrope_regions_test, overlap_kvs_test, read_plan_test, read_sched_test, read_yield_test, sampler_same_test, slot_chain_test, solo_pipeline_test, spec_lanes_test.
  - Run with `CUDA_VISIBLE_DEVICES= ctest --test-dir $WT/build --output-on-failure -j16`. With no device visible, the GPU tests skip with 77.
  - Alternative: a CPU-only configure (`-DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_BUILD_TESTS=ON`) builds the conversation and tier tests without nvcc.
- **Serve tests.** There are 112 `serve/test_*.py` at v0.2.12, plus our 11–19. They use unittest, not pytest, and need no process or GPU. Run `cd $WT && /media/raid0/StrataQ38FN/Strata/.venv/bin/python -m unittest discover -s serve -p 'test_*.py' -t .`. pytest is not installed in the production venv; do not install it there.
- **tools/test_*.py.** About 150 files, mostly setup.py tests. The root `conftest.py` only isolates APPDATA, XDG_CONFIG_HOME, XDG_STATE_HOME, XDG_DATA_HOME and LOCALAPPDATA under pytest; one upstream run once wrote into the owner's real settings. Either use pytest in a separate venv, or export those five variables to a temp directory before `python -m unittest discover -s tools -p 'test_*.py'`. Static checks: `py_compile serve/*.py` and `git diff --check`.

### 6. GPU identity tests for the night window (from 23:45)
Run these after production is stopped, with test servers on 127.0.0.1:8091 and the environment pinned as in section 4.

1. **ctest GPU subset on one card** (`CUDA_VISIBLE_DEVICES=0 ctest -j1 -R …`): our parity tests are gr_parity_* (down_pf2/pf3, split2/4, norm_fast, up_split*), route_plan_fuse_parity, qsa_slots_parity(+lanecell), qsa_slots_stream_parity(+lanecell), qsa_vbatch_parity(+lanecell), kv_stream_overlap_parity, kv_stream_parity, kv_hybrid_parity, prefill_mmq_tab_test and expert_cache_slot_guard_test. Upstream's tests next to our features: batch_rows_parity, conversation_snapshot_test_ckpt_async / _ckpt_vram_slots, conversation_copy_streams_test, hc_q8_stream, hc_persist_parity, mmvf_rows_parity, fusions_parity, qsa_merge_daily_parity_*, spec_proposal_parity. Run the full `ctest -j1` if time allows; window-1441 allowed 1500 s.
2. **`tools/batch_test.py`** (upstream: solo GEN vs BGEN must match token for token) and our **`tools/batch_test16.py --batch 16 --n 16 --long-tokens 40000 --dump`**. Compare production's engine with the merged one, as window-reads2.sh step 5 does (btcmp/bootcmp).
3. **`tools/batch_interleave_test.py`** (identical on both sides: yield, turn checkpoint, slot continue), **`tools/batch_groups_cache_test.py`** (slot reuse; run SLOT_CKPT_MOVE on and off, since it is now an upstream default), **`tools/cancel_park_test.py`**, **`tools/parking_test.py`** and **`tools/session_pressure_test.py`** (conversation cache, park, RAM tier).
4. **Our tests for our features:**
   - `linux/ovkv_ident.py` (PIPE_OVERLAP_KVS with `--kv-resident`)
   - `tools/vision_batch_check.py` (MROPE_SLOTS)
   - `tools/cache_reuse_check.py` and `tools/msg_ckpt_check.py` (MSG_CKPT_TAIL, SLOT_CHAIN_REUSE)
   - `tools/batch_overlap_test.py` (PIPE_OVERLAP)
   - `tools/split_mtp_batch_parity.py dump/compare` (SPLIT_MTP_BATCH)
   - the borrow/mmap/gate-evict scenario from window-borrow.sh
5. **HTTP robustness** on the merged server: `tools/early_close_test.py` and `tools/stop_resend_test.py http://127.0.0.1:8091`.
6. **Non-exact defaults** (GDN_TC/CONVL2, AUTO_NOBORROW, PROMPT_ATTN_IMMA, DFlash2): judge them with `tools/force_check.py` (gen/score/check against two stock arms), then run the performance arms (bench_multiturn, bench_concurrency, bench_agentic) as separate arms from the exact baseline.

**Other notes:**
- Worktree `/media/raid0/StrataQ38FN/Strata-212` (branch g3-212 @146df3c4) was created today at 13:36. It is clean and has no build. I did not touch it.
- I found no upstream file that collides by name with our new files. The conceptual overlaps to resolve by content are `conv_cache.hpp` vs our tier and message checkpoints, `batch_read.hpp` vs our read_plan/read_sched/read_yield, and `PARALLEL_STACK_ENV` vs production's env.
