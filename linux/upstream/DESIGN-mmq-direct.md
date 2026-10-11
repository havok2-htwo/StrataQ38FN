# Design: MMQ reading resident experts in place ("B-tab", final after critique)

## How prefill calls MMQ today

Repo: /media/raid0/StrataQ38FN/Strata, branch g3-overlap-spike @7285036. The CUDA build compiles MMQ from STRATA_GGML_DIR=/media/raid0/StrataQ38FN/Strata/third_party/llama.cpp (build/CMakeCache.txt:941). That directory is gitignored. Without STRATA_GGML_DIR, CMake fetches llama.cpp 3cf03257 (CMakeLists.txt:1172-1180).

**Upstream entry point.** Upstream's mul_mat_q_case<T> is explicitly instantiated in the template-instances/mmq-instance-<t>.cu files that CMakeLists.txt:1278-1280 adds to strata_mmq. moe_mmq.cu includes mmq.cuh and calls mul_mat_q_case<T> through the extern DECL_MMQ_CASE declarations (mmq.cuh:1571ff).

**Sizes per layer** (src/prefill/prefill.cpp:3231-3254):
- mmq_gub = matrix_bytes(gu_type, 1280, N=2560) = 1,408,000 B for IQ3_S.
- mmq_db = matrix_bytes(d_type, N, 640) = 921,600 B for IQ4_NL, or 1,740,800 B for Q8_0.

**Setup per layer:**
- mmq::quantize(m.src_dev -> m.Xq) writes T*K q8_1 rows in `order` order. `order` is the experts in id order (3149-3153), optionally re-sorted for Aurora residency.
- bounds_host[j] = m.off[order[j]] (n+1 absolute entries), followed by per-group relative bounds at n+1+g*(MMQ_GROUP+1). They are uploaded to m.bounds_dev.

**Walk for chunks below stream_all_min()** (prefill.cpp:3693-3716):
- Resident entry: `compute(j, m.cache->device_slot(host_res[l*NE+e]), -1)` (3710).
- Streamed entry: waits on copied[sl], then `compute(j, stage_dev[sl], sl)`.

**compute() today** (prefill.cpp:3627-3673), with q = j % 16 (MMQ_GROUP = 16, :910):
- Native layer: `mmq::gather_native(blob, blob+up_off, mmq_gub/2, blob+down_off, mmq_db, grp_gu+q*mmq_gub, grp_d+q*mmq_db, m.cs)`. This is copy16_kernel (moe_mmq.cu:25), or copy1_kernel when an address is not 16-byte aligned. It is one launch per expert, resident or streamed. A ring slot then records its `used` event.
- Group end (q==15 or last entry):
  - Two cudaMemsetAsync calls zero MMQ_TAIL (4096 B) at grp_gu+ngx*mmq_gub and at grp_d+ngx*mmq_db.
  - Product gu = {w=grp_gu, type=gt, w_rows=1280, w_cols=N, expert_bytes=mmq_gub, n=ngx, xq=Xq, bounds=bounds_dev+j0, ids=ids_identity, total_rows=T*K, max_rows=max cnt in group, dst=GU, ld_dst=1280}.
  - swiglu, then quantize H[r0, r0+nr) into Hq.
  - Product dn = {w=grp_d, type=dt, w_rows=N, w_cols=640, expert_bytes=mmq_db, n=ngx, xq=Hq, bounds=bounds_dev+n+1+g*17, ids=identity, total_rows=nr, max_rows, dst=Dm+r0*N, ld_dst=N}.

**Walk for chunks of 1024 tokens or more** (3717-3756, group_gather defaults on):
- compute() stores gg.blob[q] and gg_slots, and returns early until the group ends (3636-3641).
- At the group end it calls flush() and resets gg = {}.
- release_to() can also call flush() in the middle of a group (gg.first..gg.n), so ring slots get their event.
- flush() (3601-3625) waits once on the last copied event, then calls gather_native_group (copy16_group_kernel, one launch per group) into grp_gu + i*mmq_gub (per-expert fallback when not aligned), then records one `used` event.
- Resident entries in this walk come from device_slot(host_res); peer entries come from m.pp->peer->slot_ptr (3746-3751).

**Group buffers.** grp_gu = 16*gu_max + MMQ_TAIL and grp_d = 16*d_max + MMQ_TAIL, carved from Alloc `o` (prefill.cpp:1209-1210), which sits in the borrowed tail slots when serving.

**Context::run** (moe_mmq.cu:192-226) builds mmq_args = {x=w, type, y=xq, ids_dst=ids, expert_bounds=bounds, dst, y_scale=null, ncols_x=w_cols, nrows_x=w_rows, ncols_dst=total_rows, stride_row_x=w_cols/qk, ncols_y=total_rows, nrows_dst=ld_dst, nchannels_x=nchannels_y=n, stride_channel_x=expert_bytes/type_size, stride_channel_y=0, stride_channel_dst=0, nsamples=1,1, ncols_max=ncols_opt=max_rows}. It then switches over the types and calls mul_mat_q_case<T>. The upstream chain (third_party mmq.cuh) is:
1. mul_mat_q_case (1561): rows % 128 == 0, so fallback=false.
2. mul_mat_q_switch_J (1478): picks the smallest number of x-tiles over ncols_opt.
3. launch_mul_mat_q (1396-1475):
   - CUDA_SET_SHARED_MEMORY_LIMIT runs for BOTH mul_mat_q<T,J,false> and <T,J,true> (1409-1410).
   - When stream_k is on (it is on Ampere for IQ3_S, IQ4_NL and Q8_0), the grid is ntiles when tile efficiency is at least 90%, otherwise nsm (82 on a 3090). ntiles = ntx*nty*nchannels_y.
   - The fixup kernel runs when ntiles % grid != 0.

**Expert addressing in the kernel.** The static __global__ mul_mat_q (955) addresses expert z as `offset_x = fastdiv(wt,sample_ratio)*stride_sample_x + fastdiv(zt,channel_ratio)*stride_channel_x + it*I*stride_row_x` at three sites (1053, 1147, 1231). It passes (x, offset_x) to mul_mat_q_process_tile (874).

**Over-read.** K_vram = 256. The 640-wide down rows are 20 blocks of qk=32, read 8 per iteration, so the last row reads 4 blocks past the matrix: 72 B for IQ4_NL, 136 B for Q8_0. Gate/up (IQ3_S, qk=256, 10 blocks, 1 per iteration) reads nothing past the end.

## Approach

**Why not idea B (uniform stride, no MMQ change).** Idea B cannot be bit-identical, and the layout it needs is not the default:
- nchannels sets llama.cpp's stream-k kbc partition and grid (`kbc = blockIdx.x * (nchannels * ntx * nty * bpn) / gridDim.x`, and grid = nsm below 90% tile efficiency). Empty experts count toward that partition, so changing nchannels moves the k-split points and the fixup sums.
- The default native cache is profile-ordered and interleaves layers (generate.cpp:4485-4499). With no profile and per-layer slots off, it is even a UNIFORM cache (generate.cpp:4578-4579, `xcache.open(..., max_blob)`, off_ empty).
- kv-grow can move a different layer's blob into a slot.
- lcm padding costs 3.35% per IQ4_NL slot and 6.4% per Q8_0 slot.

**Design "B-tab" (exact). Only the base address of each expert's matrix changes; the launch geometry and the arithmetic do not.**

1. **Same geometry.** Groups of 16 in `order` order, n = ngx, the same bounds, ids, max_rows, Xq/Hq, types, J, grid and fixup. The table index is the group position q, not a compact index.

2. **Pointer table inside strata_mmq.** Strata adds a tracked copy of llama.cpp's outer scheduler: the mul_mat_q kernel body, launch_mul_mat_q and switch_J, fallback=false only. It differs from upstream only in these places:
   - At the three offset_x sites, the per-expert base comes from a 16-entry pointer table passed by value (`xz = tab.p[fastdiv(zt, channel_ratio)]`), and stride_channel_x drops out of offset_x.
   - The launcher drops the `CUDA_SET_SHARED_MEMORY_LIMIT(mul_mat_q<T,J,true>)` line, a documented deviation that keeps the fallback=true kernels from being instantiated.
   - process_tile, load_tiles, vec_dot, write_back and mul_mat_q_stream_k_fixup are upstream's own templates from the same mmq.cuh. They are compiled in the same strata_mmq target with the same flags.

3. **Which bytes a resident expert reads.** For a resident expert whose slot has guard room:
   - gate/up pointer = device_slot(s) (gate|up are contiguous: up_off = gu_half, native_expert.cpp:73-74);
   - down pointer = device_slot(s) + down_off;
   - no gather.

   Streamed experts, peer (P2P) experts, ring-slot experts, and resident experts without guard room or with an unaligned pointer are gathered as today. They go to compact positions c = 0..ns-1 of grp_gu/grp_d, and their table entries point there. The zero tail sits after position ns.

4. **Down over-read.** Each guarded slot keeps at least 256 B of room after its blob (the guard). One prep_group launch per group replaces today's two tail memsets. It zeroes the over-read bytes (72 or 136 B) after each direct blob in the group and also the 4096 B tails. So whatever wrote the slot slack before (loans, kv chunks, VMM grow, cross-layer relocation, a uniform slot's previous occupant) cannot leak into the result.

5. **The guard enters the cache through a setter, not a signature change.**
   - The SYCL engine compiles against the shared include/strata/core/expert_cache.hpp (sycl/CMakeLists.txt:93-95) and defines the 4-argument open_sized out of line (sycl/src/core/expert_cache.cpp:704). The 4-argument declaration therefore stays byte-identical.
   - A new inline `set_slot_guard(g)` is set once before opening, like set_segment_bytes, and is not reset by close(). The CUDA/HIP open_sized reads it. The SYCL open_sized ignores it, which is safe because slot_tail then finds no room and the expert is gathered.
   - Uniform caches (open) get no guard: their stride stays blob_ = max_blob, because resident_plan and the bytes=0 fill callers assume slot*blob_.
   - slot_tail is computed through slot_offset(), so it is valid on uniform caches. There, IQ4_NL-down layers go direct through the 819,200 B of max_blob slack, and Q8_0-down layers (room == blob) fall back to the gather.

6. **Opt-in through env STRATA_MMQ_DIRECT:**
   - 0 (default): today's code, byte for byte.
   - 1: guard layout plus direct MMQ.
   - 2: guard layout plus today's gather path, the bitwise A/B reference.

   The feature is CUDA only: under HIP, without STRATA_MMQ_TAB, or when the configure fingerprint mismatches, tab_built() is false and direct_mode() is forced to 0, so there is no guard and no layout change.

7. **Unchanged paths.** Decode, verify, the fused walk (2925-2935), the peer branches (3393, 3470), the helper stage, the non-MMQ dequant path and slot index semantics are unchanged.

## Changes

### /media/raid0/StrataQ38FN/Strata/include/strata/prefill/moe_mmq.hpp — before struct Product; Product; end of namespace
- Move `constexpr int kGatherGroupMax = 16;` above Product.
- Add `struct WeightTab { const void* p[kGatherGroupMax] = {}; };` and the Product field `const WeightTab* tab = nullptr;`. When it is set, expert z < n <= 16 starts at tab->p[z], and w/expert_bytes are not used for addressing.
- Declarations:
  - `bool tab_built();`
  - `bool tab_fits(int type, int64_t w_rows);`, true when tab_built() && fits(type) && w_rows%128==0 && an instance exists for the type;
  - `size_t overread_bytes(int type, int64_t cols);`, the bytes past a matrix that the last row's final K_vram iteration reads: ((bpi - bpn%bpi) % bpi) * type_size, with bpi = MMQ_ITER_K/qk and bpn = cols/qk;
  - `constexpr size_t kSlotGuard = 256;`
  - `int direct_mode();`, which reads STRATA_MMQ_DIRECT once (0/1/2) and forces 0 when !tab_built();
  - `size_t direct_slot_guard();`, which returns kSlotGuard when direct_mode()!=0, else 0, and makes no CUDA call;
  - `struct GuardTab { uint8_t* p[kGatherGroupMax] = {}; int n = 0; };`
  - `void prep_group(const GuardTab& g, size_t guard_bytes, void* tail_gu, void* tail_d, size_t tail_bytes, void* stream);`
- SYCL note: sycl/src/prefill/{moe_mmq.dp.cpp, gemm.dp.cpp, prefill.cpp} include this header. The additions are a default-initialized field and declarations that SYCL never ODR-uses, so SYCL compiles unchanged. sycl's Context::run ignores p.tab, and nothing in sycl sets it.

### /media/raid0/StrataQ38FN/Strata/src/prefill/moe_mmq_tab.hpp (new, tracked) — declaration-only header
- `struct mmq_x_tab { const char* p[16]; };`
- `template<ggml_type T> bool mul_mat_q_tab_case(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);`, a declaration only, with no body.
- For each tab type, under the same #ifdefs as Context::run: `extern template bool mul_mat_q_tab_case<GGML_TYPE_X>(...);`.
- moe_mmq.cu includes ONLY this header, so the tab kernels are never implicitly instantiated in moe_mmq.cu.

### /media/raid0/StrataQ38FN/Strata/src/prefill/moe_mmq_tab.cuh (new, tracked) — definitions, included only by the per-type instance TUs
- Include mmq.cuh and moe_mmq_tab.hpp.
- `template<ggml_type type,int J> __launch_bounds__(ggml_cuda_mmq_get_nthreads(type,J,false), ggml_cuda_mmq_get_occupancy(type,J,false)) static __global__ void mul_mat_q_tab(const mmq_x_tab tab, const int* y, ...)`. The parameter list is mul_mat_q's (mmq.cuh:956-961) minus `x`, with fallback fixed to false. The body is copied verbatim from mmq.cuh:963-1237. At each of the three offset_x sites (1053, 1147, 1231) it uses `const char* xz = tab.p[fastdiv(zt, channel_ratio)]; const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + it*I*stride_row_x;` and passes xz in place of x to mul_mat_q_process_tile<type,J,false,fixup>.
- launch_mul_mat_q_tab<type,J>: a copy of mmq.cuh:1396-1475 with these differences:
  - one CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_tab<type,J>)) line, with the fallback=true line deliberately dropped;
  - both the tiling and the stream-k branches launch mul_mat_q_tab;
  - upstream's `mul_mat_q_stream_k_fixup<type,J,false>` is called unchanged (it never reads x);
  - the same ctx.pool tmp_fixup and the same grid rule.
- mul_mat_q_tab_switch_J<type>: a copy of mmq.cuh:1478-1557 (fallback=false).
- `template<ggml_type T> bool mul_mat_q_tab_case(ctx, a, tab, s) { if (a.nrows_x % 128) return false; mul_mat_q_tab_switch_J<T>(ctx, a, tab, s); return true; }`.
- static_asserts as a second net behind the configure fingerprint: std::is_same on decltype(&mul_mat_q_process_tile<GGML_TYPE_Q8_0,8,false,false>) and on &mul_mat_q_stream_k_fixup<GGML_TYPE_Q8_0,8,false> against the exact function-pointer types the copy calls.
- A header comment names the llama.cpp commit(s) and the mirrored line ranges, and lists the two deliberate deviations.

### /media/raid0/StrataQ38FN/Strata/src/prefill/mmq_tab_instance.cu.in (new, tracked) — template for per-type TUs
`#include "moe_mmq_tab.cuh"` plus `template bool mul_mat_q_tab_case<GGML_TYPE_@T@>(ggml_backend_cuda_context&, const mmq_args&, const mmq_x_tab&, cudaStream_t);`. Each type gets its own TU, so nvcc builds them in parallel and MSVC objects stay small.

### /media/raid0/StrataQ38FN/Strata/src/prefill/moe_mmq.cu — Context::run (:192-226); new functions after gather_strata_q2
- Context::run builds mmq_args exactly as today, except that stride_channel_x is 0 when p.tab is set. When p.tab is set:
  - require 0 < p.n <= kGatherGroupMax;
  - copy the tab into mmq_x_tab;
  - call run_tab(ctx, a, tab, s), which switches over the same types and #ifdefs and calls mul_mat_q_tab_case<T>;
  - if run_tab returns false, print and exit(1). This cannot happen once the caller has checked tab_fits().
- Under `#if !defined(STRATA_MMQ_TAB) || defined(__HIPCC__)`: tab_built() is false, and run_tab prints and exits (unreachable).
- prep_group_kernel: grid (ceil(max(guard, tail)/256), g.n + 2), 256 threads.
  - y < n: zero guard_bytes bytes at g.p[y], with byte stores.
  - y == n / n+1: zero tail_gu / tail_d (null skips). It uses uint4 stores only when the host checked ((uintptr_t)ptr | tail_bytes) % 16 == 0, passed as a flag, and byte stores otherwise. IQ3_S/IQ4_NL/Q8_0 sizes (1,408,000 / 921,600 / 1,740,800) are 16-aligned, but the check covers other types.
  - The host skips the launch when g.n == 0 and both tails are null.
- Implement overread_bytes (pure arithmetic on ggml_blck_size/ggml_type_size and MMQ_ITER_K), tab_fits, tab_built, direct_mode and direct_slot_guard.

### /media/raid0/StrataQ38FN/Strata/CMakeLists.txt — CUDA strata_mmq block (~1251-1295); HIP branch untouched
- Add `option(STRATA_MMQ_TAB "prompt MMQ reads resident experts in place" ON)`.
- Fingerprint gate instead of a name probe:
  - file(READ) ${_strata_mmq}/mmq.cuh;
  - extract with string(FIND/SUBSTRING) the regions the copy depends on: from `static __device__ __forceinline__ void mul_mat_q_process_tile` through the end of mul_mat_q_case, plus the `struct mmq_args` block;
  - normalize CRLF to LF and take string(SHA256);
  - compare against `_strata_mmq_tab_known` (a list of hashes recorded for every llama.cpp tree the copy was verified against, starting with the vendored third_party tree and the fetched 3cf03257 once verified).
  - On a mismatch: message(STATUS "STRATA_MMQ_TAB off: mmq.cuh differs from the mirrored scheduler") and set the option OFF.
  - The probe compiles nothing, so configure stays fast and a foreign STRATA_GGML_DIR never breaks the build.
- When ON:
  - foreach _t in _strata_mmq_cuda_types: string(TOUPPER), configure_file(src/prefill/mmq_tab_instance.cu.in ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu @ONLY), then target_sources(strata_mmq PRIVATE ...). Being in the same target keeps -use_fast_math, the arch list and /bigobj.
  - target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TAB=1).
  - Add the CUDA-only test prefill_mmq_tab_test (strata_mmq, ggml-base) with add_test.
- HIP and SYCL: nothing is added.

### /media/raid0/StrataQ38FN/Strata/include/strata/core/expert_cache.hpp — public inline members next to set_segment_bytes (:121), slot_offset (:210); private members (:221-241)
- open/open_sized declarations unchanged (SYCL's out-of-line 4-argument definition keeps matching).
- New inline members, so SYCL needs no new definitions:
  - `void set_slot_guard(int64_t g) { guard_req_ = g > 0 ? g : 0; }` (configuration, like seg_req_; not reset by close());
  - `int64_t slot_guard() const { return off_.empty() ? 0 : guard_req_; }`;
  - `static int64_t slot_stride(int64_t blob, int64_t guard = 0) { return (blob + guard + 255) / 256 * 256; }`;
  - `uint64_t slot_room(int32_t s) const { return slot_offset(s + 1) - slot_offset(s); }`;
  - `uint8_t* slot_tail(int32_t s, uint64_t blob, uint64_t need) const { if (s < 0 || s >= slots_ || base_ == nullptr) return nullptr; return slot_room(s) >= blob + need ? base_ + slot_offset(s) + blob : nullptr; }`. It is safe on uniform caches (slot_offset handles an empty off_), and base_ is the single VA range for plain, VMM and segmented caches (device_slot uses base_ too, expert_cache.cpp:658-668).
- New private member `int64_t guard_req_ = 0;`.

### /media/raid0/StrataQ38FN/Strata/src/core/expert_cache.cpp — open_sized (:561-593)
- `off[i+1] = off[i] + (uint64_t) slot_stride(slot_bytes[i], guard_req_);`
- blob_ stays max(slot_bytes), unpadded, so fill_slot*/verify_slot (bytes=0 means blob_) and the HIP ensure_blocking_staging(blob_) never read a host blob past its end.
- open() (uniform) is unchanged.
- sycl/src/core/expert_cache.cpp is unchanged: it ignores guard_req_, so its slots have no guard and slot_tail returns null for full slots (and SYCL has no direct path anyway).

### /media/raid0/StrataQ38FN/Strata/include/strata/prefill/prefill.hpp + /media/raid0/StrataQ38FN/Strata/src/prefill/prefill.cpp (stub namespace :55-70) — Prefill public statics; no-MMQ stubs
- Add `static size_t Prefill::slot_guard_bytes();`, defined in src/prefill/prefill.cpp as mmq::direct_slot_guard() (0 in the no-MMQ build), so generate.cpp needs no MMQ header.
- The SYCL prefill.cpp shares prefill.hpp but never calls it, so no definition is needed there.
- Add no-MMQ stubs: tab_built false, tab_fits false, overread_bytes 0, direct_mode 0, direct_slot_guard 0, prep_group no-op.

### /media/raid0/StrataQ38FN/Strata/src/prefill/prefill.cpp — MmqPlan / mmq_plan() (:915-950)
- Add `std::vector<char> direct; std::vector<uint32_t> d_over;`.
- Layer l gets direct = 1 only when all of these hold:
  - lay.native && direct_mode()==1;
  - tab_fits(gt,1280) && tab_fits(dt,N);
  - overread_bytes(gt,N)==0 && overread_bytes(dt,640) <= kSlotGuard;
  - f.up_off == matrix_bytes(gt,1280,N)/2 && f.down_off == 2*f.up_off;
  - matrix_bytes(dt,N,640) == f.bytes - f.down_off.
- In that case d_over[l] = overread_bytes(dt,640): 72 for IQ4_NL, 136 for Q8_0.
- Log once: `prompt MMQ reads resident experts in place (STRATA_MMQ_DIRECT=1): X of L layers`.

### /media/raid0/StrataQ38FN/Strata/src/prefill/prefill.cpp — per-layer MoE walk: flush (:3601-3625), compute (:3628-3673), call sites :3710 and :3748
- `const bool direct = use_mmq && lay.native && m.cache && mp.direct[l];`
- Per-layer group state: `mmq::WeightTab tgu, td; mmq::GuardTab gd; int ns = 0;`
- Signature: `compute(j, blob_dev, slot, int32_t cslot)`. Call sites:
  - 3710 passes `host_res[l*NE+e]`;
  - streamed entries pass -1;
  - 3748 passes `r0 ? host_res[...] : -1` (peer pointers are always gathered).
- When !direct, the existing body runs byte for byte.
- When direct, in the use_mmq branch:
```
q = j % 16; last = q+1 == 16 || j+1 == order.size();
uint8_t* gp = (cslot >= 0 && ((uintptr_t) blob_dev | f.down_off) % 16 == 0)
              ? m.cache->slot_tail(cslot, f.bytes, mp.d_over[l]) : nullptr;
if (gp) { tgu.p[q] = blob_dev; td.p[q] = blob_dev + f.down_off; gd.p[gd.n++] = gp; ++stats_.experts_direct; }
else {
  c = ns++; tgu.p[q] = grp_gu + c*mmq_gub; td.p[q] = grp_d + c*mmq_db;
  if (group_gather) { gg.blob[c] = blob_dev; gg.n = c+1; if (slot >= 0) gg_slots[gg_nslots++] = slot; }
  else gather_native(blob_dev, ..., grp_gu + c*mmq_gub, grp_d + c*mmq_db, m.cs);
}
if (slot >= 0 && !group_gather) { record used[slot]; used_of[slot] = slot; }
if (!last) return true;
if (group_gather) { flush(); gg = GatherGroup{}; }   // runs even when the last entry was direct; flush is a no-op when ns == gg.first
```
- flush() is unchanged. It already gathers gg.blob[i] into grp_gu + i*mmq_gub for i in [gg.first, gg.n), so a mid-group flush from release_to keeps working with compact indices.
- At the group end, the two memsets are replaced by `mmq::prep_group(gd, mp.d_over[l], ns ? grp_gu + ns*mmq_gub : nullptr, ns ? grp_d + ns*mmq_db : nullptr, MMQ_TAIL, m.cs)`.
- gu and dn are built exactly as today plus `.tab = &tgu` / `.tab = &td`. Then tgu, td, gd and ns are reset.
- give_back/gg_nslots logic is unchanged: direct entries add no ring slots.
- Add stats_.experts_direct (prompt stats line).

### /media/raid0/StrataQ38FN/Strata/src/program/generate.cpp — :2982, :3369, :4468, :4494, :4547-4548, :4560, :4737; before the open loops (:4523 area, :4745)
- Define once, near the native_pack definition: `const int64_t slot_guard = native_pack ? (int64_t) strata::prefill::Prefill::slot_guard_bytes() : 0;`
- Replace `(b + 255) / 256 * 256` with `ExpertCache::slot_stride(b, slot_guard)` at all seven sizing sites (2982 split-skip-if-fits pairs_bytes, 3369 split placement cost(l), 4468 per-layer sum, 4494 profile sizing, 4547/4548 shrink_to, 4560 cache_bytes, 4737 stage sizing).
- `xcache.set_slot_guard(slot_guard)` next to set_segment_bytes (4523), before the retry loop. It survives close() between retries.
- `st.cache.set_slot_guard(slot_guard)` before the stage open loop (4745).
- The uniform xcache.open (4579) is unchanged.
- When slot_guard > 0, log `+256 B guard per sized slot (STRATA_MMQ_DIRECT)`.
- Unchanged sites:
  - 4821 slots_est (uses bytes()/slots);
  - peer_experts.cpp:154 and remote_experts.cpp:171 (guard 0);
  - all sycl/ files.

### /media/raid0/StrataQ38FN/Strata/tests/cuda/prefill_mmq_tab_test.cu (new); README.md; CHANGELOG.md — new test; docs
- The test is described in test_plan.
- README env table: STRATA_MMQ_DIRECT 0/1/2 (CUDA only; 1 = resident experts read in place, 2 = guard layout with the gather, for A/B).
- README CMake option table: STRATA_MMQ_TAB.
- CHANGELOG entry plus version bump in the implementation commit.


## Slot-size assumption audit

- **include/strata/core/expert_cache.hpp:95 open_sized declaration; sycl/src/core/expert_cache.cpp:704 out-of-line definition (shared header via sycl/CMakeLists.txt:93-95)**: 4-argument signature shared by the CUDA/HIP and SYCL engines — ok: True — Kept unchanged. The guard comes from the inline set_slot_guard / guard_req_. SYCL ignores it, so it gets no guard and slot_tail returns null, which is safe.
- **src/core/expert_cache.cpp:561-593 open_sized**: slot stride = round256(blob); blob_ = max(slot_bytes) — ok: True — Stride becomes slot_stride(blob, guard_req_). blob_ stays unpadded, so bytes=0 fills and HIP staging never over-read a host blob.
- **src/core/expert_cache.cpp:469 open (uniform) + generate.cpp:4578-4579 (native pack, no profile, not per-layer)**: slot stride = blob_ = max_blob; off_ empty — ok: True — No guard is added there. slot_tail goes through slot_offset(), so it does not index an empty off_. IQ4_NL-down layers have 819,200 B of slack and go direct. Q8_0-down layers (room == blob == max_blob) fail the room check and are gathered. resident_plan's slot*blob addressing is unchanged.
- **expert_cache.cpp fill_slot/fill_slot_blocking/fill_slot_queued/verify_slot (:670-776)**: copy/compare min(bytes, blob_) at device_slot — ok: True — The guard is never written or compared. Its content does not matter because prep_group zeroes the needed bytes before every use.
- **expert_cache.hpp slot_offset/slot_end/bytes/full_bytes/slots_within/bytes_of; device_slot (:658-668)**: offset-based via off_ (or s*blob_) — ok: True — All derive from off_, so the guard is counted consistently. slot_room(slots_-1) uses off_[slots_], which exists.
- **expert_cache.cpp shrink/grow (#533 --vram-elastic), segmented/VMM**: live slots wholly inside mapped segments; grown memory undefined — ok: True — The guard is inside the slot, so it is mapped whenever the slot is live. base_ is one VA range. Grown garbage is zeroed per group.
- **layer_slot_range / admit / layer cursors (expert_cache.cpp:624-655)**: index-based — ok: True — Indices are unchanged. B-tab needs no per-layer contiguity.
- **generate.cpp:2982 --split-skip-if-fits pairs_bytes**: slot costs round256(blob) — ok: False — Switch to slot_stride(b, slot_guard). The undercount is 256 B per pair.
- **generate.cpp:3369 split placement cost(l)**: slot costs round256(blob) — ok: False — Switch to slot_stride. Otherwise placement predicts that more fits than the guarded stage open at 4747 actually holds.
- **generate.cpp:4468 per-layer sum, :4494 profile sizing, :4547-4548 shrink_to, :4560 cache_bytes**: slot costs round256(blob) — ok: False — Switch to slot_stride(b, slot_guard), so that the budget, shrink_to arithmetic and xcache.bytes() agree.
- **generate.cpp:4737 stage sizing + :4747 stage open_sized**: slot costs round256(blob) — ok: False — Switch to slot_stride. Call st.cache.set_slot_guard(slot_guard) before the open loop, because stages run the prompt MMQ on their own cache.
- **generate.cpp:4568-4625 auto retry loop (open, close, shrink)**: budget in bytes; cudaMemset(xcache.bytes()) — ok: True — guard_req_ survives close() like seg_req_. The loop works on bytes once the formulas above use slot_stride.
- **generate.cpp:4821 slots_est (loan estimate)**: avg = bytes()/slots — ok: True — Offset-derived, so it includes the guard automatically.
- **generate.cpp:4655-4700 profile fill + verify_slot**: fills blob_bytes(layer) — ok: True — Unchanged.
- **generate.cpp:7498-7545 / :11662-11765 adaptive swaps (adapt_fence, fenced_slots, dma_batch cp_bytes)**: copy blob_bytes(layer) to device_slot after adapt_fence drains the stages' streams — ok: True — The fence waits for all queued stream work, so moving the resident read from gather time to group end on the same m.cs is still covered. Swaps never write the guard.
- **generate.cpp:5945-5955 kv-grow relocation, :6040-6056 refill**: an expert may move into any slot with room >= its blob — ok: True — slot_tail checks room >= blob + d_over per expert; otherwise the expert is gathered. Stale bytes after a smaller blob are zeroed by prep_group before use.
- **generate.cpp:5708-5726 slots_from_bytes/bytes_from_slots; loans (:6255-6270, :6462, :6591, :11277); kv chunks (:5923-6033)**: byte ranges from slot_offset(first) — ok: True — Offset-based. Lent slots are not resident, so prep_group never writes there. A resident slot's guard lies before off(first).
- **src/kernels/cuda/verify_kernels.cu:1059,:1132 resident_plan; src/core/verify.cpp:671-673; src/core/expert_source.cpp:3181**: slot address = base + slot_off[slot] (or slot*blob when uniform) — ok: True — They use the offsets table copied after open, or blob_ on uniform caches (unchanged). They read blob bytes only.
- **src/prefill/prefill.cpp:2925-2935 fused walk; :3393, :3470 peer branches; helper stage**: blob pointer = device_slot / slot_ptr — ok: True — Untouched.
- **src/prefill/prefill.cpp:3601-3625 flush / gather_native_group; :3636-3641 early return**: group index q == gather position — ok: True — In direct mode the gather position is compact (c) while the table uses q. flush already uses gg.first..gg.n, so it works unchanged. The early return moves after the direct/gather placement, and the end-of-group flush runs even when the last entry was direct.
- **src/core/peer_experts.cpp:154; src/core/remote_experts.cpp:171**: own cache via 4-argument open_sized — ok: True — They never call set_slot_guard, so they stay at guard 0. Their experts are always gathered (cslot = -1).
- **prefill.cpp grp_gu/grp_d (:1209-1210) + MMQ_TAIL**: <=16 gathered experts plus a 4096 B zero tail — ok: True — Still used for gathered experts at compact positions. The tail is zeroed after position ns by prep_group, with an alignment-checked store path.
- **sycl/src/prefill/{moe_mmq.dp.cpp, gemm.dp.cpp, prefill.cpp} including the shared moe_mmq.hpp/prefill.hpp**: Product layout; Prefill statics — ok: True — Only a default-null field and unused declarations are added, and nothing in sycl sets or calls them. The SYCL build is added to the build matrix.

## Exactness

1. **The bytes are the same.**
   - A resident native slot holds the exact GGUF slices: gate at +0, up at +up_off = gu_half, down at +down_off = 2*up_off (checked per layer in mmq_plan).
   - copy16/copy1 today copy gate|up and down without converting anything.
   - The rows [0,1280) of gate/up and [0,N) of down that MMQ reads for expert z are therefore byte-identical; only their base address differs.
   - Direct pointers are 16-byte aligned (checked), like the group buffer. Alignment does not change the results anyway.

2. **The launch is the same.** n = ngx, bounds, ids, total_rows, max_rows (ncols_opt/ncols_max), Xq/Hq, w_rows, w_cols and type are untouched. Therefore all of the following are identical:
   - J (switch_J);
   - ntx, nty and ntzw = nchannels_y = n;
   - ntiles, the grid (the 90% rule on nsm) and kbc = blockIdx.x * (n*ntx*nty*bpn) / gridDim.x;
   - fixup_needed and the fixup kernel's order.

   Every output tile is split at the same k-blocks and summed in the same order. stride_channel_x is set to 0 in tab mode but is no longer read for addressing. channel_ratio = n/n = 1 as today.

3. **The arithmetic is the same.**
   - The copy differs from upstream mul_mat_q only in an integer/pointer expression (xz = tab.p[zt], with the channel term removed from offset_x).
   - process_tile, load_tiles, vec_dot, write_back and mul_mat_q_stream_k_fixup are upstream's own templates from the same mmq.cuh, compiled in the same target with the same flags (-use_fast_math, arch list).
   - Float contraction is expression-local inside those templates, so the generated float sequences are expected to match. The bitwise unit test proves this per type and J.
   - The configure fingerprint plus the static_asserts keep the copy from compiling against a different scheduler or tile signature.

4. **The over-read cannot change a result.** Only the down product reads past a matrix: 4 blocks after the last row (72 or 136 B, computed by overread_bytes and gated at <= 256).
   - Today those bytes are the next gathered expert's down blocks or the zeroed MMQ_TAIL. In direct mode they are the guard, zeroed by prep_group on m.cs before the MMQ launch. A gathered entry's over-read lands on the next compact entry or on the tail zeroed at ns.
   - Either way the bytes are finite. They meet q8_1 activation blocks for k >= 640, whose quants and scale are 0, so each product term is ±0.
   - A float accumulator that starts at +0 cannot become -0 under round-to-nearest addition or FMA, so adding ±0 leaves it bitwise unchanged.
   - Finiteness is all that is required. The zeroing makes it hold no matter who wrote the slack before (loans, kv chunks, VMM grow, relocation, a uniform slot's previous occupant).

5. **Gate/up reads nothing past its matrix.** overread(gt,N) == 0 is required per layer.

6. **Gathered entries are unchanged.** Streamed, peer and unguarded experts are gathered exactly as today, only at compact positions.

7. **What the A/B reference must be.**
   - The bitwise reference is STRATA_MMQ_DIRECT=2, which has the same slot layout. Mode 0 has a different slot cost and therefore possibly a different residency.
   - The engine A/B must also pin the CPU share (STRATA_PREFILL_CPU_SHARE=0, or a fixed value). The measured share cpu_share_now (prefill.cpp:3027) is timing-driven, and mode 1 changes GPU timing by design.
   - The A/B also needs no adaptive swaps or relocations, or must compare only the requests before the first one.

## VRAM impact

**With STRATA_MMQ_DIRECT=1 or 2 (CUDA only):**
- Every sized slot (profile-ordered or per-layer primary cache, and each layer-split stage cache) gains +256 B. Both blob sizes are already multiples of 256 (2,329,600 = 9100*256; 3,148,800 = 12300*256), so the full 256 B is added per slot: +0.011% for IQ4_NL-down slots, +0.008% for Q8_0-down slots.
- That is about 2.3 MiB for 9,332 slots. Auto sizing may admit about one slot fewer per ~9,100.
- Uniform caches (native pack without a profile) gain nothing. Their Q8_0-down layers simply stay on the gather.

**Prompt buffers are unchanged:**
- grp_gu/grp_d remain for gathered experts.
- The pointer table (128 B) and guard table (136 B) are passed by value as kernel parameters.
- tmp_fixup comes from the same ctx.pool as today.

**Default (0):** no change at all, including on HIP and SYCL.

**Compared with idea B's lcm stride:** that costs +78,080 B per IQ4_NL slot (+3.35%) and +202,240 B per Q8_0 slot (+6.4%), roughly 140 fewer resident experts per 4,105 slots.

## Test plan

- Unit test, bitwise. New tests/cuda/prefill_mmq_tab_test.cu (CUDA only, exit 77 below sm_80), for IQ3_S gate/up 1280x2560, IQ4_NL and Q8_0 down 2560x640, and every other compiled tab type.
- Lay out random valid expert blobs at scattered 256-aligned offsets in one buffer. After each blob, fill the guard with 0x7C/0xFF bytes (fp16 Inf/NaN patterns).
- Run prep_group, then Context::run with tab.
- memcmp against the same experts gathered contiguously, with a zeroed MMQ_TAIL, through today's path.
- Sweep:
  - ngx 1..16;
  - rows per expert {0,1,7,8,9,31,64,100,512,1000}, mixed;
  - cases with tile efficiency above and below 90% (stream-k fixup active);
  - max_rows values that select every J from 8 to 128.
- Negative control: skip prep_group with the NaN-pattern guards and assert that non-finite values appear in the down output. This proves the guard handling is load-bearing.
- Mixed-group unit case: direct and gathered entries interleaved in one group (compact c != q), including a mid-group flush (gg.first > 0) and a group whose last entry is direct. Bitwise equal to the all-gathered result, for both gather_native and gather_native_group.
- Engine A/B, bitwise: STRATA_MMQ_DIRECT=2 vs 1, with STRATA_PREFILL_CPU_SHARE=0 (and again with a fixed value such as 0.2) pinned in both runs.
- Same --expert-cache/profile and adaptive tier off, or compare only requests served before the first swap.
- Prompts of 17, 128, 511, 1000, 1023, 1024 and 3000 tokens on the UD-IQ4_XS pack (all 48 layers, including the 5 Q8_0-down layers).
- Compare the post-prefill hidden state/logits dump, or a greedy 64-token decode with logprobs, byte for byte.
- Cache variants:
  1. profile-ordered (default);
  2. --expert-cache-per-layer;
  3. a uniform cache (native pack, no --expert-profile; Q8_0 layers must report gathered);
  4. a layer split over 4x3090 (stage caches guarded).
- Robustness runs, finite-output checks and bitwise only before the first event, with compute-sanitizer memcheck where feasible:
- --vram-elastic after a shrink/grow cycle;
- --kv-grow relocation active;
- adaptive tier swapping during serve;
- peer experts present (they must count as gathered);
- pipelined batch mode with overlapping prompts (concurrent prep_group writes of zeros over zeros).
- STRATA_MMQ_DIRECT=0 vs 2 with a fixed explicit slot count, to confirm that the guard layout alone changes nothing but byte offsets and counts.
- compute-sanitizer --tool memcheck on a short prompt with DIRECT=1, plus racecheck on prep_group.
- Build matrix:
- Linux CUDA with STRATA_GGML_DIR=third_party/llama.cpp, and with the fetched 3cf03257 pin (fingerprint either matches, or the tab is OFF with a message);
- Windows MSVC+nvcc (per-type tab TUs, /bigobj, CRLF-normalized fingerprint);
- HIP gfx11/gfx906 (no tab sources, tab_built false, no layout change);
- STRATA_ENABLE_SYCL (shared expert_cache.hpp/moe_mmq.hpp/prefill.hpp compile unchanged);
- a modified mmq.cuh that must configure with STRATA_MMQ_TAB OFF.
- Existing tests: expert_cache_segmented_test, expert_cache_per_layer_test, tests/hip/expert_cache_staging, resident_plan_parity, prefill_fused_iq_test, prefill_mmq_kquant_test, prefill_fused_moe_test.
- Perf, measured first, on 4x RTX 3090 with prompts of 128, 256, 512 and 1000 tokens:
- nsys GPU time of copy16_kernel/copy1_kernel/copy16_group_kernel;
- stats_.experts_resident vs experts_streamed vs experts_direct;
- the STRATA_PREFILL_TIMING phases (kPfDequant, kPfGemmGU, kPfGemmD, kPfWaitCopy);
- prompt tok/s, DIRECT=2 vs 1.

Expect the gather kernel time for resident experts to fall to about 0, GemmGU/GemmD to stay flat, and the gain to match the measured resident gather kernel time.

## Expected gain

**Unverified estimate; measure first.**

**What is removed.** Per resident expert, the gather's DRAM traffic goes away: about 2.33 MB read plus 2.33 MB written on IQ4_NL-down layers, or 3.15 + 3.15 MB on Q8_0-down layers. At about 800-850 GB/s effective on a 3090, that is roughly 5.5-6 µs (IQ4_NL) or 7.5 µs (Q8_0) of copy kernel per expert, plus launch overhead.
- In the <1024 walk, the per-expert launches and the two memsets per group become one prep_group launch per group.
- The ≥1024 walk also stops copying its resident experts in copy16_group_kernel.

**What is not removed.**
- kPfDequant (the ~31% figure) is not gather kernel time. PfTimer charges the gap between two marks to the earlier phase (prefill.cpp:1878-1880), so host stalls from stage_one/stager waits and launch gaps also land there. Those host stalls stay.
- Streamed and peer experts keep their gathers.
- Residency is partial in the logged configs: 9,332 slots / 21 GiB on the GPU against 34.4 GiB of experts in RAM (logs/E-A0-live.txt). A profile-ordered cache holds the hot experts, so the fraction of routed entries that are resident is higher than the byte fraction, but it has not been measured.

**How to state the gain.**
- Realistic: speedup ≈ 1 / (1 − resident_gather_kernel_ms / prefill_ms), using nsys copy-kernel time and the measured experts_resident/(resident+streamed+peer) ratio for 128-1000-token prompts.
- Back-of-envelope: about 48 layers × 200-250 resident entries × ~6 µs ≈ 60-70 ms per short-prompt chunk. Against ~400 ms for a 512-token prompt that suggests roughly 10-17% less prefill time (about 1.1-1.2x).
- Upper bound: 1/(1−0.31) = 1.45x, reachable only if all of kPfDequant were resident gather kernel time.
- MMQ GEMM time should not change: same tiles, same bytes, read from DRAM either way.

## Alternative: patching vendored mmq

Three options were compared.

**(A) Idea B: uniform stride, no MMQ change.** x = device_slot(range_lo), z = slot − lo, nchannels = range size. Rejected.
- Not exact: nchannels sets the stream-k kbc partition and the grid (nsm below 90% efficiency), and empty experts count.
- Slow on small groups.
- Needs per-layer contiguous, equal-stride slots. Those exist only with --expert-cache-per-layer (off by default). The default profile cache interleaves layers, the no-profile cache is uniform, and kv-grow relocates across layers.
- Costs 3.4% (IQ4_NL) or 6.4% (Q8_0) of cache VRAM in lcm padding.

**(B) Patch the vendored mmq.cuh.** Add a per-z table to mmq_args and mul_mat_q at the three offset_x sites.
- Smallest diff, and exact.
- But third_party/llama.cpp is gitignored, so the patch would not live in the Strata repo. The default and CI/Windows builds fetch 3cf03257 and would need a patches/ file plus a FetchContent PATCH_COMMAND (git apply, CRLF on Windows). Every user STRATA_GGML_DIR checkout would need the same patch. A failed apply breaks the build, rather than turning the feature off.
- Moderate benefit (about 30 lines), high plumbing cost and fragility.

**(C) Chosen: a Strata-side copy of only the outer scheduler.** The mul_mat_q stream-k body (~280 lines), launch_mul_mat_q (fallback=true shared-memory line dropped) and switch_J, in a tracked moe_mmq_tab.cuh, instantiated per type in strata_mmq. It calls upstream's process_tile, tile loaders, dot products, write_back and fixup from the same header.
- Exact.
- No third-party edits.
- Gated by a configure-time SHA256 fingerprint of the mirrored mmq.cuh regions, plus compile-time static_asserts. A different llama.cpp turns the feature OFF instead of failing.
- Costs: about 320 copied lines to re-sync and re-fingerprint when the pin moves (the bitwise test catches drift), and extra nvcc time for one fallback=false instantiation per (type, J), spread over per-type TUs.

## Open risks

- Fetched pin: the mirrored regions of the vendored third_party mmq.cuh (which has y_scale/NVFP4 code) may differ from 3cf03257. The copy is verified and fingerprinted against the vendored tree first. The fetched build then gets STRATA_MMQ_TAB OFF until its hash is verified and added, or a second mirrored variant is shipped.
- Compiler equivalence: process_tile is __forceinline__d into a different outer kernel. Contraction is expression-local, so identical float code is expected, but only the bitwise unit test across all types and J proves it. Keep both kernels in one target with the same flags.
- The resident-slot read moves from gather time to group end, still on m.cs within the same layer. adapt_fence (generate.cpp:7503-7511) drains the stages' streams before swap copies, which covers this. kv-grow relocation and --vram-elastic shrink must likewise fence on stream completion; this needs review and the sanitizer runs.
- prep_group writes zeros into live slots' slack. The audit found no other user of slot slack: lent slots and kv chunks are never resident, and decode/verify read only blob bytes. With overlapping prompts in pipelined batch mode, two streams can rewrite the same guard with the same zero value, which is a benign same-value race. Confirm with racecheck.
- Uniform caches (native pack, no profile): Q8_0-down layers stay gathered because they have no guard room. Padding the uniform stride would also need resident_plan/verify (slot*blob_) and the bytes=0 fill callers to change, so it is out of scope.
- Engine-level bitwise A/B needs a pinned CPU share (STRATA_PREFILL_CPU_SHARE) and no adaptive swaps. Otherwise timing-driven placement differences (prefill.cpp:3027) show up as CPU/GPU rounding differences that are not MMQ's.
- The gain is uncertain: kPfDequant includes host stalls, and only resident gathers go away. Peer and helper-stage prompt paths are deliberately unchanged; they could later point at peer slots.
- Kernel parameter tables are indexed dynamically (as copy16_group_kernel already does with ga.blob[q]). This may add a tiny per-block local-memory cost. Watch GemmGU/GemmD for regressions.
- HIP stays on the gather until tab TUs are added to the HIP branch and validated on gfx11/gfx906. SYCL stays without the feature.
- The relayed user request ('comitte aber vorher schön und push und readme und changelog', i.e. polish, then commit and push, and update README and CHANGELOG) was not carried out in this step. The computed task was read-only (no edits, builds or GPU jobs) and produced only this spec, so there is nothing in Strata to commit yet. The README/CHANGELOG entries for STRATA_MMQ_DIRECT and STRATA_MMQ_TAB, the version bump, the commit and the push belong to the implementation step on branch g3-overlap-spike.

## Critiques of the first draft

### Verdict: needs_changes
- **The spec breaks the SYCL build. The claim that the 'default argument keeps it compiling' is wrong, because the SYCL engine shares the header.** — evidence: sycl/CMakeLists.txt:93-95 puts ${STRATA_ROOT}/include on the include path, and sycl/include/strata/core has no expert_cache.hpp, so the SYCL engine compiles against include/strata/core/expert_cache.hpp. sycl/src/core/expert_cache.cpp:704 defines `ExpertCache::open_sized(const std::vector<int64_t>&, int64_t, int64_t, std::string&)` out of line. Once the header declares a fifth parameter `int64_t guard_bytes = 0`, that definition matches no declaration and the STRATA_ENABLE_SYCL build fails to compile. A default argument only helps callers, not out-of-line definitions. — fix: Keep the existing 4-argument open_sized declaration unchanged. Add a separate overload `open_sized(const std::vector<int64_t>&, int64_t, int64_t, std::string&, int64_t guard_bytes)`, or a setter `set_slot_guard(int64_t)` called before open_sized, and make the 4-argument version forward with guard 0 in src/core. Alternatively, update sycl/src/core/expert_cache.cpp in the same change. Put guard_, slot_room and slot_tail in the header as inline members so the SYCL .cpp needs no definition for them.
- **The bitwise engine A/B (STRATA_MMQ_DIRECT=1 vs 2) as planned will not be bitwise even if the MMQ change is exact, because expert placement depends on timing.** — evidence: prefill.cpp:3027 sets `m.cpu_share_now = clamp(m.cpu_g_ms / (m.cpu_c_ms + m.cpu_g_ms), ...)` from measured GPU and CPU wall time. cpu_share_env() returns -1 (measured) by default (prefill.cpp:172-175), and `stats_.cpu_share = cpu_share_env() >= 0 ? env : m.cpu_share_now`. Mode 1 changes GPU time by design, so a different set of experts goes to the CPU pool, and CPU and GPU rounding differ. Adaptive tier swaps (generate.cpp:7470-7480) are driven by heat and gain, and their timing relative to requests also differs. The test plan asks for a bitwise comparison 'with the adaptive tier swapping during serve', which cannot hold. — fix: Run the engine bitwise A/B with the CPU share pinned explicitly (STRATA_PREFILL_CPU_SHARE=0 or a fixed value), adaptive swaps off (or only compare outputs before the first swap), and a fixed slot list. Keep the swap, kv-grow and vram-elastic runs as compute-sanitizer and finite-output checks, plus a bitwise comparison only on prompts that run before any swap or relocation.
- **The CMake probe does not deliver the promise that 'a foreign llama.cpp checkout never breaks the build'.** — evidence: The spec probes only for the names mul_mat_q_process_tile, ggml_cuda_mmq_config and mul_mat_q_stream_k_fixup. The copied scheduler is coupled to the exact parameter lists of mul_mat_q, process_tile and the fixup kernel, and to mmq_args. For example, the checked-in third_party/llama.cpp mmq.cuh:955-961 already carries a y_scale argument and NVFP4-specific offset_y_scale code that older pins lack. A checkout with the same names but different signatures passes the probe and then fails in nvcc. In addition, the verbatim launch_mul_mat_q (mmq.cuh:1413-1414) runs CUDA_SET_SHARED_MEMORY_LIMIT on both mul_mat_q<type,J,false> and <type,J,true>, so a verbatim copy instantiates fallback=true kernels as well. That contradicts 'only fallback=false is instantiated' and doubles the compile cost. — fix: Replace the name probe with a check_cuda_source_compiles/try_compile of a tiny TU that includes mmq.cuh and calls mul_mat_q_process_tile<GGML_TYPE_Q8_0,8,false,false> and mul_mat_q_stream_k_fixup with the exact argument list the copy uses. Alternatively, hash the mirrored line range of mmq.cuh (the spec names the commit and lines) and compare it with a recorded value. On a mismatch, set STRATA_MMQ_TAB OFF with a message. In the copied launcher, drop the fallback=true CUDA_SET_SHARED_MEMORY_LIMIT line and state that this deviation is intended.
- **The gain estimate (1.35-1.4x prompt throughput) is not supported, because kPfDequant is not gather-kernel time.** — evidence: PfTimer (prefill.cpp:1878-1880) charges 'the time between two consecutive marks ... to the phase of the first, so a gap where the GPU waits (for the host ...) lands on the phase that was waiting'. compute() marks kPfDequant first (prefill.cpp:3629), and in the <1024 walk the host then runs stage_one between experts (prefill.cpp:3699-3703), which includes the blocking m.stager->wait for non-pinned blobs and the H2D issue. Host launch gaps and stager stalls therefore count toward the 31%. Also, only resident experts lose their gather; streamed experts (prefill.cpp:3712-3714) keep theirs. Residency is far from complete in the logged configs: 'expert cache 9332 slots, 21.03 GiB' against 34.4-37.9 GiB of experts in RAM (logs/E-A0-live.txt, E-F0-combo.txt). — fix: Before you claim a speedup, measure copy16_kernel/copy1_kernel GPU time with nsys (or count events around the gathers only) on the target 4x3090 config, together with stats_.experts_resident vs experts_streamed for 128/256/512/1000-token prompts. State the expected gain as (resident gather kernel time) / (prefill time), scaled by the measured resident fraction. Report the 31%-based 1.45x only as an upper bound.
- missed: src/program/generate.cpp:2982: --split-skip-if-fits pairs_bytes uses ((blob+255)/256*256). Switch it to ExpertCache::slot_stride(b, slot_guard); it undercounts by 256 B per pair, which the 1096 MiB margin absorbs.
- missed: src/program/generate.cpp:3369: the split-placement cost(l) lambda uses the same round-256 formula. Switch it to slot_stride.
- missed: src/program/generate.cpp:4737: the stage sizing formula, cited in the spec as ':4727'. Same change.
- missed: sycl/src/core/expert_cache.cpp:704: the out-of-line open_sized definition shares include/strata/core/expert_cache.hpp (sycl/CMakeLists.txt:93-95). A signature change breaks it.
- missed: sycl/src/prefill/moe_mmq.dp.cpp and sycl/src/prefill/prefill.cpp compile against the shared include/strata/prefill/moe_mmq.hpp and prefill.hpp. The new declarations and the Product::tab field are harmless only while nothing in sycl calls or sets them. Say so, or have the sycl Context::run reject p.tab.
- missed: src/prefill/prefill.cpp:3027: measured cpu_share_now makes the engine A/B timing-dependent. Pin STRATA_PREFILL_CPU_SHARE for bitwise runs.
- missed: third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1413-1414: in the 'verbatim' launcher copy, CUDA_SET_SHARED_MEMORY_LIMIT instantiates fallback=true too. Drop that line in the copy.
- missed: src/prefill/prefill.cpp:3640-3647 (compute, group_gather branch): the early `return true` / flush() / `gg = GatherGroup{}` sequence has to be restructured. A direct entry must still reach the group-end check, and the end-of-group flush/reset must run even when the last entry was direct (ns may be 0, in which case flush is a no-op).
### Verdict: needs_changes
- **The SYCL build shares include/strata/core/expert_cache.hpp, so changing open_sized's signature breaks it. The spec says 'sycl untouched; the default argument keeps it compiling if it ever shares the header', but SYCL already shares that header.** — evidence: sycl/include/strata/core/ has no expert_cache.hpp. sycl/CMakeLists.txt:93-95 puts ${STRATA_ROOT}/include on the include path, and sycl/src/core/expert_cache.cpp:8 includes "strata/core/expert_cache.hpp". At sycl/src/core/expert_cache.cpp:704 SYCL defines `bool ExpertCache::open_sized(const std::vector<int64_t>&, int64_t, int64_t, std::string&)` out of line. If the shared header declares only `open_sized(..., std::string& err, int64_t guard_bytes = 0)`, that 4-parameter definition matches no declaration, which is a hard compile error (a default argument does not make a 4-parameter out-of-line definition match a 5-parameter member). — fix: Leave the 4-parameter open_sized declaration as it is. Add the guard either through a setter called before open, like the existing set_segment_bytes (for example `void set_slot_guard(int64_t g)`, read in open_sized and reset in close()), or as a separate overload that only src/core defines. Define slot_stride, slot_room, slot_tail and guard_bytes inline in the header so that sycl/src/core/expert_cache.cpp links without changes. Add the SYCL build to the build matrix in the test plan.
- **slot_tail as specified (`base_ + off_[s] + blob`) indexes an empty off_ on uniform caches. Native packs reach uniform caches, so this is undefined behaviour or a crash in direct mode.** — evidence: generate.cpp:4578-4580 calls `sized_slots.empty() ? xcache.open(o.expert_cache, ..., max_blob) : open_sized(...)`. sized_slots stays empty for a native pack whenever there is no profile and --expert-cache-per-layer is off (4459/4485 branches), so the native prompt path runs on a uniform cache whose off_ is empty. expert_cache.hpp:210 shows slot_offset() already handles that case (`off_.empty() ? s*blob_ : off_[s]`), and device_slot does the same. With STRATA_MMQ_DIRECT=1, mp.direct[l]=1 and the resident walk calls slot_tail(cslot, ...) → off_[cslot] on an empty std::vector. The spec's audit wrongly lists ExpertCache::open (uniform) as 'Q2_0 pack only'. — fix: Define slot_tail with slot_offset(s): `return slot_room(s) >= blob + need ? base_ + slot_offset(s) + blob : nullptr;`. This is then safe on uniform caches: IQ4_NL-layer experts use the max_blob slack and Q8_0-layer experts fall back to the gather. Optionally also pass the guard to the uniform open, using slot_stride(max_blob, guard) as the blob size, so the Q8_0-down layers go direct there too. Add a uniform-cache case (native pack, no --expert-profile) to the engine A/B.
- missed: sycl/src/core/expert_cache.cpp:704 out-of-line ExpertCache::open_sized against the SHARED include/strata/core/expert_cache.hpp (signature change breaks the SYCL build)
- missed: src/program/generate.cpp:4579 uniform xcache.open(..., max_blob) used for NATIVE packs when sized_slots is empty (no profile, not per-layer): off_ empty (slot_tail must use slot_offset), no guard -> Q8_0-down layers never direct there
- missed: src/program/generate.cpp:2982 --split-skip-if-fits pairs_bytes uses (b+255)/256*256 (should be ExpertCache::slot_stride(b, slot_guard))
- missed: src/program/generate.cpp:3369 layer-split placement cost lambda uses (b+255)/256*256 (should be slot_stride(b, slot_guard), else stage placement over-predicts fits vs. the guarded stage open at :4737)
- missed: src/prefill/moe_mmq.cu: needs only a declaration of mul_mat_q_tab_case<T> / mmq_x_tab (or explicit `extern template` for each type), not moe_mmq_tab.cuh's body; otherwise every type's tab kernels are implicitly instantiated a second time in moe_mmq.cu
- missed: launch_mul_mat_q 'verbatim' copy: upstream's CUDA_SET_SHARED_MEMORY_LIMIT names both mul_mat_q<type,J,false> and <type,J,true>. The tab copy must reference only mul_mat_q_tab<type,J,false>, or it instantiates the fallback=true kernels the spec says are excluded
- missed: prep_group tail zeroing with uint4 stores assumes grp_gu+ns*mmq_gub and grp_d+ns*mmq_db are 16-byte aligned. This holds for IQ3_S/IQ4_NL/Q8_0 here (1,408,000 / 921,600 / 1,740,800), but needs either an assert or the existing cudaMemsetAsync-equivalent byte path