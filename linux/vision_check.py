#!/usr/bin/env python3
"""Vision checks against a running Strata (or the G3 proxy in front of it).

  vision_check.py [--base http://127.0.0.1:8090/v1] [--load 16] [--out results/vision/run.json]

1. basic:     each test picture alone; the reply must name what is in it (colour/shape/text).
2. cache:     the same text around two DIFFERENT pictures, back to back and as follow-up turns (conversation cache,
              slot parking): each reply must describe its own picture, never the other one.  All image placeholders
              are the same token id, so a cache keyed on token ids alone would hand picture A's state to picture B.
3. repeat:    the same picture twice: greedy replies must be identical (encoder cache + KV reuse).
4. load:      N text users decoding while pictures arrive: text throughput and the text users' longest stall.
"""
import argparse, base64, io, json, sys, threading, time, urllib.request

from PIL import Image, ImageDraw, ImageFont


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def picture(kind, size=(800, 600)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    w, h = size
    if kind == "red-square":
        d.rectangle([w * .2, h * .2, w * .6, h * .7], fill=(220, 20, 20))
    elif kind == "blue-circle":
        d.ellipse([w * .3, h * .15, w * .75, h * .75], fill=(20, 50, 220))
    elif kind == "green-triangle":
        d.polygon([(w * .5, h * .1), (w * .85, h * .85), (w * .15, h * .85)], fill=(20, 160, 40))
    elif kind.startswith("text:"):
        d.text((w * .1, h * .4), kind[5:], fill="black", font=font(int(h / 7)))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# what a correct reply must contain (any of), and what it must not (a sign of the other picture's state)
EXPECT = {"red-square": (["red", "rot"], ["blue", "green", "circle", "triangle"]),
          "blue-circle": (["blue", "blau"], ["red", "green", "square", "triangle"]),
          "green-triangle": (["green", "grün"], ["red", "blue", "circle", "square"]),
          "text:STRATA 4711": (["4711"], []),
          "text:KAFFEE 0815": (["0815"], ["4711"])}


def post(base, body, timeout=600, key="dummy"):
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            j = json.load(r)
        return j, time.time() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.time() - t0, f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}"
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, repr(e)


def ask(base, model, messages, max_tokens=60):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    j, dt, err = post(base, body)
    text = j["choices"][0]["message"].get("content") or "" if j else ""
    return text, dt, err, (j or {}).get("usage")


def user(text, img):
    return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": img}}]}


Q_SHAPE = "Answer in English in one short sentence: what shape and colour is in this picture?"
Q_TEXT = "Answer in one short line: which text is written in this picture? Copy it exactly."


def judge(kind, text):
    want, bad = EXPECT[kind]
    t = text.lower()
    ok = any(w in t for w in want) and not any(b in t for b in bad)
    return ok


def run(a):
    model = a.model or json.load(urllib.request.urlopen(a.base + "/models"))["data"][0]["id"]
    res = {"base": a.base, "model": model, "checks": []}

    def rec(name, kind, text, dt, err, usage=None, ok=None):
        ok = (err is None and judge(kind, text)) if ok is None else ok
        res["checks"].append({"check": name, "kind": kind, "ok": ok, "s": round(dt, 2), "reply": text[:200],
                              "error": err, "usage": usage})
        print(f"{'OK  ' if ok else 'FAIL'} {name:28s} {kind:18s} {dt:6.1f}s  {err or text[:110]!r}", flush=True)

    pics = {k: picture(k) for k in EXPECT}
    # 1. basic
    for k in EXPECT:
        q = Q_TEXT if k.startswith("text:") else Q_SHAPE
        t, dt, err, u = ask(a.base, model, [user(q, pics[k])])
        rec("basic", k, t, dt, err, u)
    # 1b. a large picture (more image tokens than the cap)
    big = picture("green-triangle", (2400, 1800))
    t, dt, err, u = ask(a.base, model, [user(Q_SHAPE, big)])
    rec("basic-large-2400x1800", "green-triangle", t, dt, err, u)
    # 2. cache: same prefix, different pictures, alternating, then follow-up turns of both conversations
    sys_msg = {"role": "system", "content": "You are a precise visual assistant. " + "Context. " * 300}
    convs = {}
    for round_ in range(2):
        for k in ("red-square", "blue-circle"):
            msgs = [sys_msg, user(Q_SHAPE, pics[k])]
            t, dt, err, u = ask(a.base, model, msgs)
            rec(f"cache-alternate-r{round_}", k, t, dt, err, u)
            convs[k] = msgs + [{"role": "assistant", "content": t}]
    for k in ("red-square", "blue-circle"):
        msgs = convs[k] + [{"role": "user", "content": "Repeat only the colour of the shape in the picture, one word."}]
        t, dt, err, u = ask(a.base, model, msgs, 10)
        rec("cache-followup", k, t, dt, err, u)
    # two pictures in one request, order swapped
    for order in (("red-square", "green-triangle"), ("green-triangle", "red-square")):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Two pictures follow. Name the colour of the shape in the FIRST picture only, one word."},
            {"type": "image_url", "image_url": {"url": pics[order[0]]}},
            {"type": "image_url", "image_url": {"url": pics[order[1]]}}]}]
        t, dt, err, u = ask(a.base, model, msgs, 10)
        want = EXPECT[order[0]][0]
        rec("two-pictures-order", order[0], t, dt, err, u, ok=err is None and any(w in t.lower() for w in want))
    # 3. repeat determinism
    r1 = ask(a.base, model, [user(Q_SHAPE, pics["blue-circle"])])
    r2 = ask(a.base, model, [user(Q_SHAPE, pics["blue-circle"])])
    rec("repeat-identical", "blue-circle", r2[0], r2[1], r2[2], r2[3], ok=r1[2] is None and r1[0] == r2[0])
    # 4. load
    if a.load:
        res["load"] = load(a, model, pics)
    print(f"\n{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])} checks passed")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if all(c["ok"] for c in res["checks"]) else 1


def load(a, model, pics):
    """N streaming text users for `secs`; pictures arrive every few seconds.  Reports text tok/s and the longest gap
    between two streamed tokens of any text user (the stall a picture causes)."""
    stop = time.time() + a.secs
    gaps, toks, img = [], [0], []
    lock = threading.Lock()

    def text_user(i):
        while time.time() < stop:
            body = {"model": model, "max_tokens": 400, "temperature": 0.7, "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": f"User {i}: write a long story about ship number {i}."}]}
            req = urllib.request.Request(a.base + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "Authorization": "Bearer dummy"})
            last = None
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    for line in r:
                        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                            continue
                        now = time.time()
                        if last is not None:
                            with lock:
                                gaps.append(now - last)
                        last = now
                        with lock:
                            toks[0] += 1
            except Exception as e:  # noqa: BLE001
                print("text user error", e, flush=True)

    def image_user():
        time.sleep(15)
        kinds = list(EXPECT)
        n = 0
        while time.time() < stop - 30:
            k = kinds[n % len(kinds)]
            q = Q_TEXT if k.startswith("text:") else Q_SHAPE
            t, dt, err, _ = ask(a.base, model, [user(q, pics[k] if n % 3 else picture(k, (1600, 1200)))])
            img.append({"kind": k, "s": round(dt, 2), "ok": err is None and judge(k, t), "error": err})
            print(f"  image {k:18s} {dt:6.1f}s {'ok' if img[-1]['ok'] else 'FAIL ' + str(err or t[:80])}", flush=True)
            n += 1
            time.sleep(a.image_every)

    th = [threading.Thread(target=text_user, args=(i,)) for i in range(a.load)] + [threading.Thread(target=image_user)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    el = time.time() - t0
    gaps.sort()
    out = {"users": a.load, "secs": round(el, 1), "text_tok_s": round(toks[0] / el, 1),
           "gap_p50": round(gaps[len(gaps) // 2], 3) if gaps else None,
           "gap_p99": round(gaps[int(len(gaps) * .99)], 3) if gaps else None,
           "gap_max": round(gaps[-1], 2) if gaps else None, "images": img}
    print(json.dumps({k: v for k, v in out.items() if k != "images"}), flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8090/v1")
    p.add_argument("--model")
    p.add_argument("--load", type=int, default=0)
    p.add_argument("--secs", type=int, default=180)
    p.add_argument("--image-every", type=float, default=8)
    p.add_argument("--out")
    sys.exit(run(p.parse_args()))
