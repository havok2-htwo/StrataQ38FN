"""Scan the bf16 PLE n-gram table of a Qwen4Exp checkpoint: global absmax and the log2|x| distribution,
to pick one FP8 (e4m3) scale for the whole table."""
import glob, json, sys, torch
from safetensors import safe_open
d = sys.argv[1]
amax = 0.0; hist = torch.zeros(80, dtype=torch.float64); n = 0; zeros = 0
for f in sorted(glob.glob(d + "/*.safetensors")):
    with safe_open(f, "pt") as fh:
        for k in fh.keys():
            if "ngram_embedding.shard_" not in k:
                continue
            t = fh.get_tensor(k).float()
            a = t.abs()
            amax = max(amax, a.max().item())
            nz = a[a > 0]
            zeros += a.numel() - nz.numel()
            hist += torch.histc(torch.log2(nz[::97]), bins=80, min=-40, max=0).double()
            n += a.numel()
    print(f, "absmax so far", amax, flush=True)
print(json.dumps({"absmax": amax, "elements": n, "zeros": zeros,
                  "log2_hist_-40_0": hist.tolist()}))
