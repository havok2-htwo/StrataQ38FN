"""Experiment patch for vLLM 0.31 (only active with QWEN4_PLE_FP8_SCALE set): run Qwen4Exp (Qwen3.8-Flash-Next) from a
compressed-tensors W4A16 checkpoint whose PLE n-gram table is bf16 (102 GB).

0.31 keeps the table in pinned host memory itself (EngramConfig.cpu_offload, UVA lookup with prefetch), but
  - picks the table format from the quant config and raises NotImplementedError for compressed-tensors, and
  - would pin it as bf16 (102 GB) through torch's caching host allocator, which rounds each 12.8 GB TP shard up
    to 16 GiB.
Here the table is FP8 (e4m3) with one global scale (absmax / 448, from ple_scan.py): ~51 GB of host RAM, pinned at
its exact size with cudaHostRegister, each checkpoint shard cast on the GPU as it is loaded.
"""
import importlib.abc
import importlib.machinery
import os
import sys

_SCALE = os.environ.get("QWEN4_PLE_FP8_SCALE")
_COMMON = "vllm.models.qwen4_exp.common.ngram_embedding"
_PLE = "vllm.models.qwen4_exp.common.ple"
_NVIDIA = "vllm.models.qwen4_exp.nvidia.ngram_embedding"


def _say(msg):
    print(f"[ple-fp8 patch 0.31] {msg} (pid {os.getpid()})", file=sys.stderr, flush=True)


def _patch_common(mod):
    import torch

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        return mod.Qwen4ExpPLEFp8EmbeddingMethod()

    mod.Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(from_quant_config)

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        cpu = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu")
        nbytes = cpu.numel() * cpu.element_size()
        err = torch.cuda.cudart().cudaHostRegister(cpu.data_ptr(), nbytes, 3)   # portable | mapped
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
        _say(f"PLE table pinned exactly: {nbytes / 2**30:.2f} GiB {dtype}")
        return cpu

    mod.Qwen4ExpPLEPinnedHostEmbedding.allocate_embedding_weight = allocate_embedding_weight

    # The pinned lookup is a Triton row copy; Triton on sm_86 cannot load fp8e4nv.  It only moves bytes (masked
    # rows become 0, which is +0.0 in e4m3 too), so run it on uint8 views of the table and the output.
    fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
    orig_lookup = mod.Qwen4ExpPLEPinnedHostEmbedding._lookup

    def _lookup(self, input_ids, output=None):
        if self.weight.dtype not in fp8_types:
            return orig_lookup(self, input_ids, output)
        if output is None:
            output = torch.empty((*input_ids.shape, self.embedding_dim), dtype=self.weight.dtype,
                                 device=input_ids.device)
        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            mod._lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight.view(torch.uint8), flat_ids, output.view(torch.uint8), self.embedding_dim,
                self.shard_indices.org_vocab_start_index, self.shard_indices.org_vocab_end_index,
                BLOCK_D=self._block_d)
        return output

    mod.Qwen4ExpPLEPinnedHostEmbedding._lookup = _lookup
    _say("FP8 PLE method + exact host pinning + byte-wise pinned lookup")


def _patch_ple(mod):
    import torch
    scale = float(_SCALE)
    fp8 = torch.float8_e4m3fn
    orig = mod.PLEVocabParallelEmbedding.weight_loader

    def weight_loader(self, param, loaded_weight, checkpoint_start=None):
        if checkpoint_start is None or param.dtype != fp8:
            return orig(self, param, loaded_weight, checkpoint_start)
        ov = mod.compute_ple_shard_overlap(
            checkpoint_start=checkpoint_start, checkpoint_rows=loaded_weight.shape[0],
            tp_start=self.shard_indices.org_vocab_start_index, tp_end=self.shard_indices.org_vocab_end_index)
        if ov is None:
            return
        src = loaded_weight.narrow(0, ov.source_start, ov.row_count)
        dev = torch.device("cuda", torch.cuda.current_device())
        q = (src.to(dev, torch.float32) / scale).clamp_(-448.0, 448.0).to(fp8)
        with torch.no_grad():
            param.data.narrow(0, ov.destination_start, ov.row_count).copy_(q)

    mod.PLEVocabParallelEmbedding.weight_loader = weight_loader


def _patch_nvidia(mod):
    import torch
    scale = float(_SCALE)
    orig = mod.Qwen4ExpNGramEmbedding.load_weights

    def load_weights(self, weights):
        loaded = orig(self, weights)
        emb = self.ngram_embedding
        if getattr(emb, "weight_scale", None) is not None:
            with torch.no_grad():
                emb.weight_scale.fill_(scale)
            loaded.add("ngram_embedding.weight_scale")
        return loaded

    mod.Qwen4ExpNGramEmbedding.load_weights = load_weights


_HOOKS = {_COMMON: _patch_common, _PLE: _patch_ple, _NVIDIA: _patch_nvidia}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn):
        self.inner, self.fn = inner, fn

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.fn(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fn = _HOOKS.get(name)
        if fn is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, fn)
        return spec


if _SCALE:
    sys.meta_path.insert(0, _Finder())
