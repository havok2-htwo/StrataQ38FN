"""No engine start: imports what batch_groups_cache_test.py imports from STRATA_REPO, checks the signatures it calls,
and prints the engine command line + the CKPT/tier env each scratch config would give."""
import inspect, json, os, sys
from pathlib import Path
REPO = Path(os.environ["STRATA_REPO"])
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO))
import strata_tokenizer as ST
from serve.server import StrataEngine, child_env, engine_args
print("engine_args", inspect.signature(engine_args), "child_env", inspect.signature(child_env))
print("StrataEngine.__init__", inspect.signature(StrataEngine.__init__))
print("StrataEngine.generate", inspect.signature(StrataEngine.generate))
assert hasattr(ST, "Tokenizer")
for p in sys.argv[1:]:
    cfg = json.load(open(p))
    extra = os.environ.get("STRATA_EXTRA_ARGS_" + Path(p).stem.split("-")[0], "").split()
    args = engine_args(cfg) + extra
    env = child_env(cfg)
    print(f"\n{Path(p).name}: {cfg['exe']} --serve {' '.join(args)}")
    print("  CKPT_MOVE=", env.get("STRATA_SLOT_CKPT_MOVE"), " CKPT_POOL=", env.get("STRATA_CKPT_POOL"),
          " TIER=", env.get("STRATA_CONV_TIER_DIR"), " CUDA_VISIBLE_DEVICES=", env.get("CUDA_VISIBLE_DEVICES"),
          " log=", cfg["log"])
    for flag in ("--batch", "--batch-groups", "--conversation-cache-mib", "--conversation-cache-slots", "--prompt-cache"):
        n = args.count(flag)
        if n > 1:
            print(f"  DUPLICATE {flag} x{n}")
