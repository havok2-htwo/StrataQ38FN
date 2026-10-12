import json, sys, threading, time, urllib.request
URL = "http://127.0.0.1:8090/v1/chat/completions"
SYS = "You are a meticulous assistant for a hardware lab. " + " ".join(
    f"Rule {i}: always state units, cite the measurement method, and keep answers factual and concise." for i in range(60))
HIST = [{"role": "system", "content": SYS}, {"role": "user", "content": "Explain Ohm's law briefly."},
        {"role": "assistant", "content": "Ohm's law states that the current I through a conductor equals the voltage V "
         "across it divided by its resistance R: I = V / R, with V in volts, I in amperes and R in ohms. " * 6}]
def ask(q, n, out, i):
    msgs = q if isinstance(q, list) else [{"role": "user", "content": q}]
    body = {"model": "x", "messages": HIST + msgs,
            "max_tokens": n, "temperature": 0}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900)
    j = json.loads(r.read())
    out[i] = (time.time() - t, j["choices"][0]["finish_reason"], j.get("usage"), j.get("timings"))
o = {}
ask("Say hello.", 16, o, 0); print("warm", o[0], flush=True)
qs = [f"Write a long, detailed essay about topic number {k}: the history of measuring voltage, current and power." for k in range(6)]
th = []
for k, q in enumerate(qs):
    th.append(threading.Thread(target=ask, args=(q, 700, o, k + 1))); th[-1].start(); time.sleep(0.4)
def nxt(k, tag):   # the next turn of chat k (it extends that chat's turn checkpoint, which sits in k's running slot)
    return [{"role": "user", "content": qs[k]}, {"role": "assistant", "content": "A short answer."},
            {"role": "user", "content": f"{tag}: and what is an ohm?"}]
time.sleep(1.0)
print("control (before the injected window)", time.time(), flush=True)
th.append(threading.Thread(target=ask, args=(nxt(1, "control"), 64, o, 90))); th[-1].start()
time.sleep(8.0)
print("late (after the injected window)", time.time(), flush=True)
for k, i in ((0, 98), (2, 99)):
    th.append(threading.Thread(target=ask, args=(nxt(k, f"late{k}"), 64, o, i))); th[-1].start(); time.sleep(1.0)
for t in th: t.join()
for k in sorted(o): print(k, o[k], flush=True)
