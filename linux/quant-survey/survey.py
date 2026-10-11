"""Survey all Qwen3.8-Flash-Next derivatives on the Hub: format, quantization, size, experts, PLE dtype, MTP.
Writes survey.jsonl (one line per repo)."""
import concurrent.futures as cf, json, sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
repos = json.load(open("repos.json"))


def one(rid):
    out = {"id": rid}
    try:
        info = api.model_info(rid, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        out["err"] = repr(e)[:120]
        return out
    files = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
    out["downloads"] = info.downloads or 0
    out["likes"] = info.likes or 0
    out["gguf_gb"] = round(sum(sz for f, sz in files if f.endswith(".gguf")) / 1e9, 1)
    out["st_gb"] = round(sum(sz for f, sz in files if f.endswith(".safetensors")) / 1e9, 1)
    out["gguf_files"] = sorted({f.split("/")[0] for f, _ in files if f.endswith(".gguf")})[:12]
    out["base"] = [t for t in (info.tags or []) if t.startswith("base_model:")][:3]
    if out["st_gb"] > 0 and any(f == "config.json" for f, _ in files):
        try:
            cfg = json.load(open(hf_hub_download(rid, "config.json")))
            tc = cfg.get("text_config", cfg)
            q = cfg.get("quantization_config") or tc.get("quantization_config") or {}
            out["arch"] = (cfg.get("architectures") or [None])[0]
            out["quant_method"] = q.get("quant_method")
            out["quant_format"] = q.get("format") or q.get("quant_algo")
            grp = (q.get("config_groups") or {}).get("group_0", {}).get("weights", {}) if q.get("config_groups") else {}
            out["bits"] = grp.get("num_bits") or q.get("bits") or q.get("w_bit")
            out["group_size"] = grp.get("group_size") or q.get("group_size")
            out["num_experts"] = tc.get("num_experts")
            out["experts_per_tok"] = tc.get("num_experts_per_tok")
            out["layers"] = tc.get("num_hidden_layers")
            out["ple_dtype"] = tc.get("ple_embedding_dtype")
            out["mtp_layers"] = tc.get("mtp_num_hidden_layers")
            out["ignore"] = (q.get("ignore") or q.get("modules_to_not_convert") or [])[:6]
        except Exception as e:  # noqa: BLE001
            out["cfg_err"] = repr(e)[:120]
    return out


with cf.ThreadPoolExecutor(16) as ex, open("survey.jsonl", "w") as f:
    for i, r in enumerate(ex.map(one, repos)):
        f.write(json.dumps(r) + "\n")
        if i % 100 == 0:
            print(i, file=sys.stderr, flush=True)
