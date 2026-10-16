# Low-level levers on g3-1441 (2026-10-16)

Base: g3-1441 f17f3eec (= production engine strata-g3-1441a). Each directory is one branch (`ll-<name>`), applied in
any order; merged as g3-1441-ll. Every lever is opt-in; with the switches off the SASS of every kernel object equals
build-1441 (CUDA 13.0, MMQ_KQUANTS/Q6K_EXPERTS as that build).

| dir | switch | what |
|---|---|---|
| qsastream | `STRATA_QSA_SLOTS_STREAM=1` (with `STRATA_QSA_SLOTS=1`) | multi-slot QSA launches also with a streamed KV cache (production: --kv-resident < max-context) |
| vbatch | `STRATA_ATTN_VBATCH=1` (=4) | QSA decode attention loads V 8 (4) cells at a time, same fmaf order |
| hcpf | `STRATA_GR_DOWN_PF=2\|3`, `STRATA_GR_NORM_FAST=1` | deeper weight prefetch in the staged HC down read; register norm on CUDA; HC_Q8 uploads only the stage's layers |
| earlysample | `STRATA_EARLY_SAMPLE=1` | the last stage samples right behind its graph |

Analysis: `../../upstream/ANALYSIS-kernel-efficiency.md`. Window scripts: `../../window-ll.sh`, `../../bisect-ll.sh`.
