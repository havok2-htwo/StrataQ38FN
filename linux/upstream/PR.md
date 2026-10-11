## batch: more than 8 slots on a layer split, and a device-to-device slot copy

Two independent commits on top of fb58e0d. Measurements and context are in #<issue>.

### 1. More than 8 slots with a layer split (8 rows per pipeline group)
- `--batch` above `kVerifyMaxT` is accepted when the slots run in pipeline groups. Each group's window holds `batch / groups` rows, at most 8, so the limit becomes 8 x stages. One group or no split keeps the 8-slot cap. If the groups resolved would put more than 8 rows in one window, the count is cut back with a warning.
- The verifiers are initialized with `min(batch, kVerifyMaxT)` rows.
- The inter-stage hand-off holds one row per slot. Groups write at `hbase = group * rows`, so with more than 8 slots the old 8-row buffer overflowed and raised an illegal memory access.
- Adds `tools/batch_test16.py` and `tools/batch_test32.py`: `batch_test.py` with 16 and 32 questions.

Verified on 4x RTX 3090 with UD-IQ4_XS and `--layer-split 12,24,36 --batch-groups 4 --pcie-frac 0 --adapt-every 1000000`: 16/16 and 32/32 slots are identical to solo.

### 2. Device-to-device slot copy
- `conversation_checkpoint_copy` and `conversation_kv_copy` write the same bytes and targets as save + restore, without the host image. They queue `cudaMemcpyAsync` on the legacy stream.
- `refresh_residency()` is factored out of `conversation_kv_restore` and shared.
- `copy_to_slot`, and `copy_from_slot` without a checkpoint, use the new path: one sync per device before and after, and queued copies are drained on error. `STRATA_SLOT_COPY_DIRECT=0` switches back to the host path.

Verified: `batch_test16.py` 16/16 identical, and `batch_interleave_test.py` identical for A, B, C and the next turn from its slot. Admission copy dropped from 65.6 to 34.0 ms on average on a 4-card split.

Not tested: HIP and Windows builds. The code only uses `cudaMemcpyAsync` and `cudaDeviceSynchronize`, which hip_compat maps. The SYCL port keeps its own copy of the batch path and is untouched.
