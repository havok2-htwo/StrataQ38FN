# Third-party notices

This repository is a set of scripts, configs, measurements and documentation around
[Strata](https://github.com/Niko1221/Strata). It does not contain a copy of Strata itself (the engine is cloned
separately into `Strata/`, which is not tracked), but it does contain material derived from third-party projects:

| What | Where | Origin | License |
|---|---|---|---|
| Engine patches (diffs against Strata source, with context lines) | `linux/patches/**` | Strata, © 2026 Niko1221 and the Strata contributors | MIT (below) |
| Pointer-table MMQ scheduler, mirrored from `ggml/src/ggml-cuda/mmq.cuh` (llama.cpp 3cf03257) | inside `linux/patches/mmq-tab/`, `linux/patches/v0.1.42/`, `linux/patches/v0.1.43/`, `linux/patches/v0.1.44.1/` (`src/prefill/moe_mmq_tab.cuh`) | llama.cpp / ggml, © 2023-2026 The ggml authors | MIT (below) |
| Evaluation questions (sample of MMLU-Pro and GSM8K test sets) | `linux/eval/data/sets.json` | TIGER-Lab/MMLU-Pro, openai/gsm8k | MIT (see the dataset cards) |

The vLLM patches in `linux/vllm/` are our own code that patches an installed vLLM at run time; vLLM itself
(Apache-2.0) is not included. Model weights are not part of this repository; each model's own license applies.

## Strata

```
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## llama.cpp / ggml

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
