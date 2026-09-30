#!/usr/bin/env python3
"""The Ollama embed request carries options.num_gpu: 0 by default, the configured value when set.

Pins GPT's review on membot #3: "add a small payload-level test that the Ollama request carries
`options.num_gpu == 0` by default and the configured override when set."

No network. urllib.request.urlopen is replaced before _embed_via_ollama runs, and OLLAMA_HOST
points at a closed port, so a missed patch fails loudly instead of reaching a live ollama or a
live membot. The env var is read at import, so every environment runs in a fresh interpreter.

Run from membot/:  python tests/test_ollama_embed_payload.py
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)

failures = []


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if detail and not ok else ""))


PROBE = r"""
import io, json, sys, urllib.request
sys.path.insert(0, %r)
import membot_server as m

captured = []

class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False

def fake_urlopen(req, timeout=None):
    captured.append({"url": req.full_url, "body": json.loads(req.data)})
    return _Resp(json.dumps({"embeddings": [[0.0, 1.0, 2.0]]}).encode())

urllib.request.urlopen = fake_urlopen
vec = m._embed_via_ollama("hello")
print(json.dumps({"captured": captured, "vec_len": int(vec.shape[0])}))
""" % _MEMBOT_DIR


def probe(**env_overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MEMBOT_") and k != "OLLAMA_HOST"}
    env["OLLAMA_HOST"] = "http://127.0.0.1:9"  # discard port: never a live server
    env.update(env_overrides)
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True,
                       cwd=_MEMBOT_DIR, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"probe failed rc={r.returncode}\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# 1. Default: CPU.
out = probe()
check("default: exactly one request sent", len(out["captured"]) == 1, out)
body = out["captured"][0]["body"]
check("default: options.num_gpu == 0",
      body.get("options", {}).get("num_gpu") == 0, body)
check("default: request goes to /api/embed on OLLAMA_HOST",
      out["captured"][0]["url"] == "http://127.0.0.1:9/api/embed", out["captured"][0]["url"])
check("default: model and input still carried",
      body.get("model") and body.get("input") == "hello", body)
check("default: embedding parsed", out["vec_len"] == 3, out)

# 2. Override: a host whose accelerator is genuinely free.
out = probe(MEMBOT_OLLAMA_EMBED_NUM_GPU="99")
body = out["captured"][0]["body"]
check("override 99: options.num_gpu == 99",
      body.get("options", {}).get("num_gpu") == 99, body)
check("override 99: sent as an int, not a string",
      isinstance(body.get("options", {}).get("num_gpu"), int), body)

# 3. Explicit 0 is the same as the default.
out = probe(MEMBOT_OLLAMA_EMBED_NUM_GPU="0")
check("explicit 0: options.num_gpu == 0",
      out["captured"][0]["body"].get("options", {}).get("num_gpu") == 0, out)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
