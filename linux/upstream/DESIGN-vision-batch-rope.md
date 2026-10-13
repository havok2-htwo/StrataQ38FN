# Design: image and text slots sharing the batch (per-slot M-RoPE positions)

Status: design only, no code changes. Line numbers refer to `HEAD` = `ee67a078` (branch `g3-143`) in `/media/raid0/StrataQ38FN/Strata-143`. The interim refusal that another session has added to `src/program/generate.cpp` (uncommitted, about 26 lines near the GENI parse) shifts the later lines of that file by about +23 in the working tree.

## 1. Recommendation

Use **option (c): one table region per slot, addressed through the position value.** It is option (b)'s memory layout without changing any kernel.

- On each device the table grows from `[cells][3]` to `[(1 + B) * stride][3]`. Region 0 is the main sessions' region and behaves exactly as today's table. Region `1 + b` belongs to batch slot `b`.
- Every rope kernel already reads the value a caller passes as a **row index** into the table (`mrope.hpp:51-53`: `tab[pos * 3 + pair % 3]`). So a slot row passes `mrope_base(slot) + cell` instead of `cell`, and reads its own region. No kernel signature, kernel body or graph argument type changes.
- Region 0 is rebuilt at every admission, as today (`generate.cpp:11883-11970`). At `copy_to_slot` the main region is copied into the slot's region, but only when the request has pictures or the slot's region is not the identity. Text-only traffic never pays for this copy.
- Without `--vision` every base is 0 and the change is a no-op. With `--vision` and only text traffic, every region is the identity and the kernels return the same integers, so the floating-point results are identical.

Cost:
- VRAM: about 51 MiB per GPU at the production settings (262K context, 16 slots), about 204 MiB over the 4 cards. An optional `--vision-slots K` brings this down to (2 + K) regions.
- Code: about 2 days of host-side work, plus a GPU validation window.

Option (a), a per-row rope offset with the table bypassed in decode, was rejected; §4 gives the reasons. In short, it changes 3 indexer kernels, needs a mechanism to select the table per launch site that has to reach the stage threads, and leaves an edge case open where an image falls in the prompt's last block.

## 2. How positions flow today (findings)

### 2.1 How the table pointer is read

Each launcher reads the table pointer on the host at launch time (`mrope_table()`) and passes it as a kernel argument:

| Launcher | Read at |
|---|---|
| native rope | `native_rope.cu:212,215,237,241` |
| rope_neox | `rope.cu:127` |
| indexer_key_append | `qsa.cu:736` |
| native indexer | `native_qsa_indexer.cu:259,263,285` |
| prefill rope | `prefill/kernels.cu:2197,2201` |

- **Every CUDA graph therefore bakes in the pointer**, while its contents are read at replay. `mrope.hpp:13-14` says the same.
- The registry is one entry per device (`native_rope.cu:127-138`).
- One table is allocated per device: CUDA0 at `generate.cpp:3511-3527`, each later stage at `3911-3922`.
- At every GEN/GENI admission the table is rebuilt and uploaded synchronously to all devices (`generate.cpp:11883-11970`). The image layout is `put(i+j, p, p+y, p+x)`, the text tail is `p+(c-n)`, and a text request after an image resets the table to the identity.

### 2.2 The decode windows of active slots run between an image prompt's chunks

- `read_part` interleaves `batch_step()` between chunks (`generate.cpp:13196-13199`, `in_prompt_read`). A queued window also runs between request lines (`11456`: `piped ? pump(true) : batch_step()`).
- With `STRATA_PIPE_OVERLAP` a prompt segment is read on its own thread while `pump` runs groups (`12903`, `13125`).
- Production has `parallel: 16` on a 4-way split, which gives pipelined groups (`PGroup`, `generate.cpp:10875`, `G.pos[t] = sl.p` at `10999`).

So "swap the table only during the image prompt's chunks" is not safe under any design in which decode graphs read the same table.

### 2.3 Where the rope position and the KV cell are already separate, and where they are fused

| Path | Rope positions | KV cell, attention length, QSA step | Fused? |
|---|---|---|---|
| Verifier, batch rows (`stage_batch`, `verify.cpp:2832-2873`) | `h_pos_` → device `pos_` (`[T][NH]`, `[T][NKV]`, `[T][IQ]`, `verify.cpp:628,931-934`). Read only by the rope calls at `1270,1275,1283,1284,1369,1373,1378,1384,1391,1398` | `h_step_` (`qsa_step_fill`, `2867`) → `step_`: kv_append (`1302-1341`), block scores/top-k/attn (`1422-1463`) | **Indexer only:** `native_qsa_indexer_append(..., step_+kStepPos, pos_base=0)` (`1355-1357`) derives the block's rope position from the cell (`pos_base + R*b`, `native_qsa_indexer.cu:92`). The `dec_batch` path at `1348` is the same but main-only (`!batch_rec_`) |
| Verifier, batch commit graph (`verify.cpp:2655-2760`) | n/a | `commitb_` cells | **Indexer re-append** `2744-2747`: `pos_base = 0`, rope from the cell |
| Verifier, solo windows (`stage_inputs`, `2117-2129`) and solo commit (`2084-2093`) | `h_pos_ = pos0 + t` | `h_step_` | Indexer as above. **Main sessions only** |
| MTP drafter (`mtp.cpp`) | `h_pos_` / `pos_` (`1222,1268,1304,1463`), read by rope only (`746-821`) | `h_step_` / `step_` | Separate. Each `put(row, cell)` writes both |
| Session token path (`session.cpp:188`, `layer.cpp:1058`) | `st.host_pos = pos_base + pos` → `st.pos_dev` | `st.step` | **Fused:** the legacy `indexer_key_append` reads `pos_dev` as the cell (`qsa.cu` kernel: `pos = *pos_dev; slot = pos % r; b = pos / r`) and `layer.cpp:1106`. **Main sessions only** (`pos_base = 0`, `generate.cpp:15033`) |
| Prompt path (`prefill.cpp:1629,2848,2911,2914,2924,2930`) | scalar `p0`/`c0` | the same `p0` | Fused, but **main sessions only** |

A slot row never runs the session-token or prefill paths. Slots are filled by `copy_to_slot` (`generate.cpp:10104`), and BYIELD only copies text into a slot (`13163-13169`, `req_imgs.empty()`). In slot decode, the only place where the cell and the rope position are fused is the **indexer's pooled-key rotation**, which uses `pos_base + R*b`.

### 2.4 Positions are not deltas inside the prompt

Text after the last image has `rotary(c) = c + (p_end - n)`. Image cells have three different positions. Text between images has its own offset. In decode, every row is text after the last image (`generate.cpp:11941`: the prompt cannot end in an image), so a single delta would be correct for the Q/K rows. It would not be correct for the indexer block that decode completes when that block's first cell, `4*floor(n/4)`, lies at or before the last image cell.

## 3. Chosen design (c): slot regions folded into the position value

### 3.1 Table layout

```
stride  = round_up(max_context + 64, 64)    // a multiple of R = 4: native indexer requires pos_base % R == 0
                                              // (native_qsa_indexer.cu:245)
regions = 1 + B                             // B = o.batch as parsed (an upper bound on the slots that fit)
table   = int32 [regions * stride][3]       // one per device, as today: d_mrope (CUDA0), st.mrope (stages)
base(main)   = 0
base(slot b) = (1 + b) * stride
```

At startup every region is the identity, uploaded once from the existing `mrope_host` identity vector, which stays one region long. The pointer is still set before any graph capture (`3526`, `3921`), only the allocation is larger.

Indices stay well inside `int`: 17 × 262208 × 3 ≈ 13.4M. Rotary values stay at or below the cell, so `RopeTab` lookups (`rope_tab_cs`) and the `cos_tab` reads (`rope.cu:109`) stay in range.

### 3.2 Where each session's base lives

- Add `int32_t mrope_base = 0;` to `SessionState` (`include/strata/core/session.hpp:31`), documented as the table row of this session's cell 0 (`--vision`), with 0 for the main sessions and for runs without a table.
- Set `bslot_ss[k][b]->mrope_base = base(b)` on **every stage**, right after the slot sessions are carved (`generate.cpp:4853-4880`). Each stage's verifier stages its own `h_pos_`.
- `MtpDrafter::load(..., SessionState& ss, ...)` (`mtp.cpp:244`) copies `ss.mrope_base` into a new member `rope_base_`. Slot drafters are carved from `bslot_ss.back()[b]` (`generate.cpp:4945-4956`); the main drafter's base is 0.
- Assert at startup that every base is 0 unless `o.vision`. Without a table the passed value is the rotary position itself.

### 3.3 Staging changes (the complete list)

| # | File:line | Change |
|---|---|---|
| 1 | `src/core/verify.cpp:2866-2872` (`stage_batch`) | `const int32_t rp = (int32_t) (slots_[rows[t]]->mrope_base + pos[t]);` for the `h_pos_` Q rows, the `pk` KV-head rows and the `pi` indexer-query rows. `h_step_` keeps `pos[t]` (the cell). This one change covers the serial windows (`run_slot_rows`, `2925-2930`, which recurses over the stages at `2956`/`3020`) and the pipelined groups (`batch_launch`, `3156-3162`). |
| 2 | `src/core/verify.cpp:1355-1357` (per-row indexer append, batch rows, `STRATA_BATCH_SELF_COMMIT` and `STRATA_NO_BATCH_KV`) | `pos_base = slot_ss(t).mrope_base` instead of `0`. This is a constant of the captured layout: `brow_` is set at capture (`2663`) and graphs are keyed by `bkey(rows,S,hbase)`. Main windows still pass 0. |
| 3 | `src/core/verify.cpp:2744-2747` (batch commit graph re-append) | `pos_base = slots_[rows[first]]->mrope_base`, again a layout constant. |
| 4 | `src/core/mtp.cpp:1222, 1268, 1304, 1463` (the four `put`/record writers of `h_pos_`/`ps`) | `rope_base_ + cell` in `h_pos_` only; `h_step_` keeps `cell`. `draft_first` (`1525`) delegates to `draft` and is covered. Slot drafters never prefill (their K/V is copied from `mtp` in `copy_to_slot`, `10124-10140`), but 1222/1268 get the base too for uniformity. |
| 5 | `src/program/generate.cpp:3511-3527`, `3911-3922` | Allocate `regions * stride * 3` int32 per device and fill every region with the identity. `mrope_host` stays one region; `upload_mrope` (`11949-11959`) still writes region 0 only. |
| 6 | `src/program/generate.cpp:10104` (`copy_to_slot`, which covers both the admission at `14561` and BYIELD at `13169`) | After the K/V copy: if `!mrope_identity` (region 0 holds an image layout) **or** `slot_mrope_img[b]` is set, `cudaMemcpy` D2D region 0 → region `1+b` (the **whole** stride, so that the decode tail `p+(c-n)` comes along) on CUDA0 and on every `st->mrope`. Then set `slot_mrope_img[b] = !mrope_identity`. Keep `slot_mrope_img` as a vector outside `BSlot`, because `sl = BSlot{}` (`14572`) resets per-request fields and this flag describes the region's contents. Order it the way the K/V copy is ordered (that path already synchronises each device, `10079-10101`). |
| 7 | `src/program/generate.cpp`, the interim refusal in the working tree (around 11880-11905) | Remove it: image and text admissions may interleave again. |
| 8 | `include/strata/kernels/mrope.hpp:9-14` | Comment only: the value a caller passes is a table **row** (`mrope_base + cell`); main sessions use base 0, batch slots use their own region. |

The following are **unchanged**, either because they serve the main sessions only and use base 0, or because they are kernels:

- `stage_inputs` (`verify.cpp:2117-2129`) and the solo commit (`2084-2093`); `pl_launch` (`3392`).
- The session token path (`session.cpp:188`, `layer.cpp:1058,1103-1106`).
- The whole prompt path (`prefill.cpp:1629`, the E-9 `draft_kv` from `STRATA_SPLIT_MTP_BATCH`, and `prefill.cpp:2848,2911,2914,2924,2930`).
- `read_windows` prompt windows, which run on the main sessions.
- All kernels: `native_rope.cu:61-62,112-113`, `rope.cu:109`, `qsa.cu:238`, `native_qsa_indexer.cu:99-100,133-134`, `prefill/kernels.cu:1638-1641`.
- Our patch paths:
  - `STRATA_QSA_SLOTS` (`verify.cpp:1404-1438`) reads only `step_`.
  - `STRATA_BATCH_SELF_COMMIT` is covered by #2.
  - `STRATA_SPLIT_MTP_BATCH` runs on the main drafter.
  - `STRATA_ADMIT_MERGE_FIRST` is already off for image requests (`generate.cpp`, `merge_from`: `!req_imgs.empty()`).
  - `STRATA_SLOT_COPY_DIRECT` goes through `copy_to_slot`, covered by #6.

**HIP:** both the gfx906 compat build and the RDNA `STRATA_ENABLE_HIP` build compile the same `src/kernels/cuda/*.cu` and `src/core/*.cpp` (`CMakeLists.txt:59-71`), so there is nothing extra to do.

**SYCL:** this is a separate converted tree with the same host structure (`sycl/src/program/generate.cpp:2864-2888,3321-3335,8484,9410-9513`, `sycl/src/core/verify.cpp` with `stage_batch`, and `sycl/src/core/mtp.cpp`). Rows 1-6 must be mirrored there. Its kernels (`*.dp.cpp`) stay unchanged. If the SYCL port is not done, SYCL needs to keep the interim refusal.

### 3.4 Optional: `--vision-slots K`

This reduces VRAM.
- Regions become main, one permanent identity region, and K image regions. Slots `b < K` get base `(2 + b) * stride`; slots `b >= K` share the identity region (base `stride`). Bases are still constant per slot, so the graph properties are kept.
- The engine answers `ERR` to a BGENI into a slot `>= K` and reports `INFO vision_slots=K`.
- The server's `pick_slot` (`serve/server.py:1595`) / `generate_batched` (`1290`, BGENI at `1439`) chooses slots `< K` for requests that carry embeddings.
- Default: K = B, the simplest setting.

### 3.5 Server

- Once the engine refusal is gone, BGENI already flows through `generate_batched` (`server.py:1439`); no change is needed beyond optional `--vision-slots` routing.
- The `_vision_down()` gate in the parallel branch (`server.py:3797`) only decides whether the engine must be (re)started under the fifo; it is not a position concern and stays.

## 4. Options considered

**(a) Per-row rope offset with an identity or absent table in decode, and the image table only while the prompt is read.** Rejected:
1. The indexer derives the pooled key's rotation from the cell (`native_qsa_indexer.cu:92,133`, `qsa.cu:238`). A per-row delta needs new arguments and arithmetic in `append`, `append_first`, `append_blocks`/`pooled_value`, `indexer_key_append`, and in both commit re-appends. That means kernel changes and parity work for byte-identity; upstream's own note (`native_rope.cu:42-44`) shows that even a code path skipped at run time moved results.
2. Decode graphs must not read the image table, but the prompt path must. The pointer is read at launch inside launchers that both sides share: `native_rope_apply` is used by the MTP prefill graphs (`mtp.cpp:1213-1239`) and by the drafter rounds, and the solo verify windows also read the prompt (`read_windows`). This would need a launch-site override that also reaches the persistent hand-off threads (`STRATA_HANDOFF_PIPE`) and the overlap read thread. A thread-local scope is fragile.
3. Edge case: when an image cell lies in the prompt's last incomplete block (its first cell is `4*floor(n/4)`), the block that decode completes is rotated at an image cell's (t,h,w), which no single delta reproduces. This needs a refusal rule.
4. Advantage: no VRAM cost. It is the "natural" upstream model if the kernels are ever reworked.

**(b) Explicit per-slot tables with a per-row slot-index array.** It has the same VRAM as (c), but every rope kernel and launcher (6 kernels plus the launchers) gains a per-row slot array and a stride, and the batch graphs gain new device buffers. (c) gets the same effect by folding the slot index into the position value the kernels already read as a row index.

**(c) Region base folded into the position value.** This is the recommended option: no kernel changes, graph-stable, exact for every cell including image cells and the edge in (a)3, and isolated to image slots if a call site is missed (see §8).

## 5. Rope position vs KV cell

The `pos` arrays given to `run_slot_rows`/`batch_launch` (`generate.cpp:10212,10221,10999`) are cells. Inside `stage_batch` they feed both `h_step_` (cell: KV write, `n_kv`, `n_bid`, width) and `h_pos_` (rope).

The design splits them only at that point (#1): `h_step_` stays the cell, `h_pos_` becomes base + cell. The slot loop and `PGroup` do not change.

- The one place in slot decode where a cell feeds the rope is the indexer, and #2/#3 handle it through the existing `pos_base` argument.
- `batch_rows_error` and the `pos + 1 > max_cells` check (`verify.cpp:2846-2851`) keep checking cells.
- The fused paths (session token, prefill, legacy indexer) serve the main sessions only, whose base is 0.

**Side effect on metadata.** `idx_block_pos` now records `base + 4b` for blocks completed in a slot (`native_qsa_indexer.cu:126`, `*block_pos = rope_pos`). No kernel reads it; it is only saved and copied in checkpoints and parked conversations (`conversation_state.cpp:181,198,229`). Session files refuse `--batch` (`generate.cpp`, "session files do not support --batch"). This is noted under risks.

## 6. VRAM

| Setting | Per GPU | 4 GPUs |
|---|---|---|
| Today (one region) | 262208 × 12 B ≈ 3.0 MiB | 12 MiB |
| (c), B = 16 (17 regions) | ≈ 51 MiB | ≈ 204 MiB |
| (c), `--vision-slots 4` (6 regions) | ≈ 18 MiB | ≈ 72 MiB |

The table is allocated before the sessions and the expert cache (`auto`), so the cache absorbs the difference: well under 1% of its slots. The engine should log the table size beside the "slot sessions" line.

## 7. CUDA graphs

- **No recapture and no new graph arguments.** The table pointer is the same allocation, only larger. `pos_`/`step_` are the same device buffers.
- The indexer `pos_base` in the batch window and commit graphs (#2, #3) is a per-slot constant, and the layout key (`bkey(rows,S,hbase)`) already includes the slots. The pipelined layouts use `rows[t] = base + t` (`verify.cpp:3159`).
- Solo and main graphs keep `pos_base = 0`.
- MTP round and step graphs read `pos_` staged per call.
- Text and image slots share the same graphs.
- Pad rows (inactive slots in a padded layout, `G.pos = 0`) read their own slot's region row 0, which is always (0,0,0); their output is already discarded.

## 8. Text-only byte-identity

- **Without `--vision`:** every base is 0, so every staged value and every `pos_base` is unchanged. The computation is identical by construction, and no kernel is rebuilt with different code.
- **With `--vision`, text only:**
  - Every region holds the identity, `mrope_pos(tab, base + c, k) = c`. The kernels see the same integers and run unchanged code, so their floating-point results are bit-identical.
  - A text request admitted into a slot with an identity region skips the copy in #6, so admission timing is unchanged.
  - Regions are only rewritten by image traffic.
  - The only difference is the `block_pos` metadata described in §5.
- **Isolation:** a call site missed in §3.3 can only give wrong positions to slots whose region is not the identity, that is image slots. Text slots stay correct even with a bug in the new code.

## 9. Test plan

**CPU (allowed now):**
1. New `tests/program/mrope_layout_test.cpp`, with the layout builder (`generate.cpp:11887-11943`) and the region math moved into a small header such as `include/strata/program/mrope_layout.hpp`. It checks:
   - text-only gives the identity;
   - one image, two images, and text between images match mtmd (t = p, h = p+y, w = p+x, tail `p+(c-n)`);
   - `stride % 64 == 0`;
   - the bases are distinct and do not overlap;
   - an identity region gives `row(base + c) = c`;
   - the "prompt cannot end in an image" error is kept.
2. Extend `tests/core/batch_rows_test.cpp` with a host-side `stage_batch` staging check, if `stage_batch` can be split into a pure helper: `h_step_` = cell and `h_pos_` = base + cell for every head group (q, kv, idx).
3. Build the CUDA target, and the SYCL target if ported. Run the existing CPU unit tests.

**GPU (maintenance window, never on the live production cards):**
4. Kernel parity (unchanged kernels, as a sanity check): `rope_parity`, `qsa_parity`, `qsa_slots_parity`, `verify_batch_parity`.
5. **Bit-exactness of text.** Run `tools/batch_test.py --batch 16 --dump` with the old and the new build, with and without `--vision`, on the production config (layer split 12,25,37, two pipelined groups, the production env flags). Solo and batch tokens must be identical across the builds, and every slot must still equal its solo run.
6. **Image slot equals solo.** Extend `batch_test.py` with `--images`, which calls the vision encoder (`strata-vision-cuda`, as `serve` does) to write the SVE1 file. Compare greedy GENI solo against BGENI with the other slots idle and against BGENI beside 15 text slots; the tokens must be identical. This is the same assertion the tool already makes for text.
7. **Mixed-batch scenarios:**
   - (i) an image admitted while text slots decode (its chunks interleaved with windows: `--prefill 512` and a long prompt so that `il_parts > 0`);
   - (ii) a text admission while image slots decode;
   - (iii) two different images in two slots at once;
   - (iv) an image slot ends, then the same slot takes text (the region reset);
   - (v) BYIELD around an image read;
   - (vi) conversation-cache follow-ups (`linux/vision_check.py` cache and repeat tests);
   - (vii) `--batch-mtp` with a single group, for the slot drafters;
   - (viii) `STRATA_PIPE_OVERLAP=1`.
   The text slots' tokens must equal their solo runs and the image slots must equal theirs.
8. **Negative control:** a debug env that skips the region copy in #6 must make test 6/7 FAIL. This proves the tests can detect the bug.
9. `linux/vision_check.py --load 16`: text throughput and the longest stall. Compare against text-only to confirm that the copy in #6 adds no measurable cost.
10. Check the VRAM and expert-cache slot counts in the startup log against the previous build.

## 10. Effort

| Item | Estimate |
|---|---|
| `session.hpp`, `verify.cpp` (#1-#3), `mtp.cpp`/`mtp.hpp` (#4) | 0.5 day (~30 lines) |
| `generate.cpp` (#5-#7, plus optional `--vision-slots`), `mrope.hpp` comment | 0.5-1 day (~80-120 lines) |
| CPU unit tests and the `batch_test.py --images` extension | 0.5-1 day |
| SYCL mirror | 0.5 day |
| Server `--vision-slots` routing (optional) | 0.25 day |
| GPU validation (§9 items 4-10) | 1 day in a maintenance window |

Total: about 3-4 days. Option (a) would be about 5-7 days because of the kernel changes and parity work.

## 11. Risks

1. **A missed staging site** gives silently wrong text for **image slots only** (see §8). Mitigation: the call-site list in §3.3, plus `grep h_pos_|pos_base|put(` in review, plus the negative control in test 8.
2. **A future kernel that derives a rope position from a cell** in slot decode, as the indexer does, would ignore the base. Mitigation: the `mrope.hpp` comment states the rule; upstream review.
3. **VRAM:** about 51 MiB per GPU at B = 16 and 262K context. Mitigation: `--vision-slots K`.
4. **Region copy ordering vs in-flight windows:** handled by putting the copy inside `copy_to_slot`, after its per-device synchronisation and before `draft_first` (`generate.cpp:14582`) and before the slot's first window.
5. **`idx_block_pos` holds base + 4b** for blocks completed in a slot. It is write-only metadata today, but any future reader must subtract the base. Parked conversations and the tier keep these bytes, which is harmless.
6. **Bases without a table:** a non-zero base without `--vision` would rotate by the wrong angle. Mitigation: the startup assert in §3.2.
7. **The SYCL tree diverges** if it is not ported. It must keep the interim refusal until it is.
8. **Upstream acceptance:** the meaning of "the position a caller passes" changes from cell to table row. The upstream PR should present it that way and include the CPU layout test.
9. **The existing synchronous upload at admission** (`generate.cpp:11949-11959`, `cudaDeviceSynchronize` on every device) stays as it is. Region 0 no longer conflicts with slot windows, so it could later become an asynchronous copy ordered on the prompt stream; that is a separate optimisation.