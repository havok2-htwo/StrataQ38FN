#!/usr/bin/env python3
"""Writes the scratch configs for run_ckpt_move_check.sh from the production config (read only).
One config per variant x CKPT_MOVE setting, each with its own log and its own empty conversation tier dir."""
import json, os, sys
from pathlib import Path

D = Path(__file__).resolve().parent
PROD = Path("/media/raid0/StrataQ38FN/linux/configs/production.json").resolve()
base = json.load(open(PROD))
CACHE_ARGS = ("--conversation-cache-mib", "--conversation-cache-slots", "--prompt-cache")


def strip(args, names):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in names:
            skip = True
            continue
        out.append(a)
    return out


# variant: (keep "parallel": 16 -> --batch 16 with auto groups (4x4), strip the cache args (the extras give them))
VARIANTS = {
    "U":    (False, True),    # upstream's test as written: --batch 2 --batch-groups 2 + small cache (extras)
    "B16":  (True,  False),   # production exactly: --batch 16, groups auto 4x4, 128 GiB / 448 conversation cache
    "B16P": (True,  True),    # --batch 16 with upstream's small cache (prompt-cache 4, 4 GiB, 3 conversations)
}
for v, (keep_parallel, strip_cache) in VARIANTS.items():
    for move in ("on", "off"):
        tag = f"{v}-move{move}"
        cfg = json.loads(json.dumps(base))
        if strip_cache:
            cfg["args"] = strip(cfg["args"], CACHE_ARGS)
        if not keep_parallel:
            cfg.pop("parallel", None)       # no --batch 16 before the extras' --batch 2
        cfg["log"] = str(D / "logs" / f"{tag}.log")
        tier = D / "tier" / tag
        tier.mkdir(parents=True, exist_ok=True)
        cfg["env"]["STRATA_CONV_TIER_DIR"] = str(tier)
        if move == "on":
            cfg["env"]["STRATA_SLOT_CKPT_MOVE"] = "1"
        else:
            cfg["env"].pop("STRATA_SLOT_CKPT_MOVE", None)
        cfg.pop("port", None)
        (D / "configs").mkdir(exist_ok=True)
        json.dump(cfg, open(D / "configs" / f"{tag}.json", "w"), indent=1)
        print(D / "configs" / f"{tag}.json")
(D / "logs").mkdir(exist_ok=True)
