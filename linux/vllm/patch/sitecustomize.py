"""Experiment patch (only active with QWEN4_PLE_FP8_SCALE set): load Qwen4Exp's bf16 PLE n-gram table as FP8 (e4m3)
with one global scale, so that with `--cpu-offload-params ngram_embedding` the table takes ~51 GB of pinned host RAM
(12.8 GB per GPU at TP=4) instead of ~102 GB.

vLLM 0.29.0 has an FP8 PLE embedding method, but selects it only for an Fp8Config checkpoint; the local checkpoint is
compressed-tensors W4A16 with a bf16 table.  This hooks the module import (also in the spawned TP workers, which
inherit PYTHONPATH and the env var) and
  - selects Qwen4ExpPLEFp8EmbeddingMethod for the table,
  - casts each checkpoint shard to FP8 as it is loaded (x / scale, clamped to +-448),
  - fills the global weight_scale and reports it loaded.
"""
import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = "vllm.models.qwen4_exp.nvidia.ple_layer"
_UVA = "vllm.model_executor.offloader.uva"
_SCALE = os.environ.get("QWEN4_PLE_FP8_SCALE")


def _patch(mod):
    import torch
    from vllm.models.qwen4_exp.common import ple as ple_common

    scale = float(_SCALE)
    fp8 = torch.float8_e4m3fn

    def quant_method(quant_config, prefix):
        return mod.Qwen4ExpPLEFp8EmbeddingMethod()

    mod._get_ple_embedding_quant_method = quant_method

    orig_loader = ple_common.PLEVocabParallelEmbedding.weight_loader

    def weight_loader(self, param, loaded_weight, checkpoint_start=None):
        if checkpoint_start is None or param.dtype != fp8:
            return orig_loader(self, param, loaded_weight, checkpoint_start)
        ov = ple_common.compute_ple_shard_overlap(
            checkpoint_start=checkpoint_start, checkpoint_rows=loaded_weight.shape[0],
            tp_start=self.shard_indices.org_vocab_start_index, tp_end=self.shard_indices.org_vocab_end_index)
        if ov is None:
            return
        src = loaded_weight.narrow(0, ov.source_start, ov.row_count)
        dev = torch.device("cuda", torch.cuda.current_device())
        q = (src.to(dev, torch.float32) / scale).clamp_(-448.0, 448.0).to(fp8)
        with torch.no_grad():
            param.narrow(0, ov.destination_start, ov.row_count).copy_(q)

    ple_common.PLEVocabParallelEmbedding.weight_loader = weight_loader

    orig_load = mod.Qwen4ExpNGramEmbedding.load_weights

    def load_weights(self, weights):
        loaded = orig_load(self, weights)
        emb = self.ngram_embedding
        if getattr(emb, "weight_scale", None) is not None:
            with torch.no_grad():
                emb.weight_scale.fill_(scale)
            loaded.add("ngram_embedding.weight_scale")
        return loaded

    mod.Qwen4ExpNGramEmbedding.load_weights = load_weights
    print(f"[ple-fp8 patch] active in pid {os.getpid()}: PLE table as FP8, scale {scale}", file=sys.stderr, flush=True)


def _patch_uva(mod):
    """Pin offloaded parameters in place and at their exact size: torch's pinned caching allocator rounds a
    12.8 GB table up to 16 GiB, and `.to("cpu").pin_memory()` keeps a second, unpinned copy while it pins -
    4 workers x (12.9 + 16) GB at once ran the 128 GB box out of memory.  Here: one plain CPU tensor of the exact
    size, registered with cudaHostRegister (portable | mapped), filled straight from the GPU."""
    import torch

    def pin_exact(gpu):
        cpu = torch.empty(gpu.shape, dtype=gpu.dtype, device="cpu")
        nbytes = cpu.numel() * cpu.element_size()
        err = torch.cuda.cudart().cudaHostRegister(cpu.data_ptr(), nbytes, 3)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
        cpu.copy_(gpu)
        return cpu

    def _maybe_offload_to_cpu(self, module, prefix=""):
        if (params := next(module.parameters(), None)) is None:
            return module
        if params.device == torch.device("cpu") or self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                break
            if p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False):
                continue
            if self.cpu_offload_params and not any(f".{q}." in f".{prefix}{name}." for q in self.cpu_offload_params):
                continue
            if not self.uva_offloading:
                raise RuntimeError("exact-pin patch needs UVA")
            cpu = pin_exact(p.data)
            p.data = mod.get_accelerator_view_from_cpu_tensor(cpu)
            p._vllm_is_uva_offloaded = True
            p._g3_pinned_host = cpu                 # keep the registered host tensor alive
            self.cpu_offload_bytes += cpu.numel() * cpu.element_size()
            torch.cuda.empty_cache()
        return module

    mod.UVAOffloader._maybe_offload_to_cpu = _maybe_offload_to_cpu
    print(f"[ple-fp8 patch] exact host pinning active in pid {os.getpid()}", file=sys.stderr, flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn):
        self.inner = inner
        self.fn = fn

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.fn(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fn = {_TARGET: _patch, _UVA: _patch_uva}.get(name)
        if fn is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, fn)
        return spec


if _SCALE:
    sys.meta_path.insert(0, _Finder())
