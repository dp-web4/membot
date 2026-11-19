#!/usr/bin/env python3
"""Live five-arm negative control for the browser write gate (fleet port of Waving Cat's
2026-11-17 fix; forum/waving-cat-to-fleet-browser-write-gate-shipped-2026-11-17.md).

The claim is about what a browser can REACH, so this starts a throwaway membot on a free
loopback port (BASE_DIR = this checkout), sends the arms, and kills it. Nothing is sent to
any running seat.

Arms, per unauthenticated write route (/api/store, /api/sync, /api/mount, /api/save):
  A1 foreign Origin   + application/json -> 403
  A2 foreign Origin   + text/plain       -> 403   (simple request; separates fix 1 from a fix)
  A3 no Origin        + application/json -> pass  (CLI callers keep working)
  A4 no Origin        + text/plain       -> 415
  A5 extension Origin + application/json -> pass  (Heartbeat survives)
For /api/sync the write itself is observed: each arm uses its own machine name and the test
checks whether cartridges/heartbeat_<machine>.jsonl came into existence. Status alone could
be a gate that answers 403 after the handler already wrote.
Preflight: foreign OPTIONS on a write route carries no Access-Control-Allow-Origin; the
extension origin is echoed. Reads (/api/search, /api/status OPTIONS) still return '*'.
Header names are compared lower-cased (Waving Cat's harness trap: a case miss reads as absent).

Usage: python tests/test_browser_write_gate_live.py [--expect-hole]
  --expect-hole: run against an UNPATCHED server; PASS means the gauge fired on the known hole
  (A2 on /api/sync lands a write), which is the gauge's first job.
"""
import json, os, socket, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.environ.get("MEMBOT_PY", sys.executable)
FOREIGN = "https://evil.example"
EXT = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
WRITE_ROUTES = ["/api/store", "/api/sync", "/api/mount", "/api/save"]


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def req(port, path, method="POST", origin=None, ctype=None, body=b"", extra=None):
    h = {}
    if origin: h["Origin"] = origin
    if ctype: h["Content-Type"] = ctype
    if extra: h.update(extra)
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body if (method == "POST" and body is not None) else None,
                               headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def body_for(route, tag):
    if route == "/api/sync":
        return {"exchanges": [{"turn": 1, "userMessage": "probe"}], "machine": f"wgate_{tag}"}
    if route == "/api/store":
        return {"content": f"write-gate probe {tag}", "tags": "wgate-probe"}
    if route == "/api/mount":
        return {"name": f"wgate_nonexistent_{tag}"}
    return {"name": ""}


def main():
    expect_hole = "--expect-hole" in sys.argv
    port = free_port()
    log = open(os.path.join("/tmp", f"wgate_server_{port}.log"), "w")
    srv = subprocess.Popen([PY, os.path.join(ROOT, "membot_server.py"), "--transport", "http",
                            "--host", "127.0.0.1", "--port", str(port), "--writable"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    results = []  # (name, ok, detail)
    try:
        t0 = time.time()
        while time.time() - t0 < 180:
            try:
                if req(port, "/api/status", method="GET")[0] == 200: break
            except Exception:
                time.sleep(1)
        else:
            print("server never came up; see", log.name); return 2
        cart = os.path.join(ROOT, "cartridges")

        def landed(tag):
            return os.path.exists(os.path.join(cart, f"heartbeat_wgate_{tag}.jsonl"))

        arms = [("A1", FOREIGN, "application/json", "deny403"),
                ("A2", FOREIGN, "text/plain", "deny403"),
                ("A3", None, "application/json", "pass"),
                ("A4", None, "text/plain", "deny415"),
                ("A5", EXT, "application/json", "pass")]
        for route in WRITE_ROUTES:
            for arm, origin, ctype, want in arms:
                tag = f"{port}_{arm}"
                st, hd, _ = req(port, route, origin=origin, ctype=ctype,
                                body=json.dumps(body_for(route, tag)).encode())
                if want == "deny403": ok = st == 403
                elif want == "deny415": ok = st == 415
                else: ok = st not in (403, 415)
                detail = f"status={st}"
                if route == "/api/sync":
                    w = landed(tag)
                    detail += f" write_landed={w}"
                    ok = ok and (w == (want == "pass"))
                    if expect_hole and arm == "A2":
                        results.append(("HOLE sync A2 foreign text/plain lands", w, detail))
                results.append((f"{route} {arm} want={want}", ok, detail))
        # preflight on a write route
        st, hd, _ = req(port, "/api/sync", method="OPTIONS", origin=FOREIGN,
                        extra={"Access-Control-Request-Method": "POST",
                               "Access-Control-Request-Headers": "content-type"})
        acao = hd.get("access-control-allow-origin")
        results.append(("preflight foreign on write route has no ACAO", acao is None,
                        f"status={st} acao={acao!r} keys={sorted(hd)}"))
        st, hd, _ = req(port, "/api/sync", method="OPTIONS", origin=EXT,
                        extra={"Access-Control-Request-Method": "POST"})
        acao = hd.get("access-control-allow-origin")
        results.append(("preflight extension echoes origin", acao == EXT, f"status={st} acao={acao!r}"))
        # reads untouched
        for path in ["/api/search", "/api/status"]:
            st, hd, _ = req(port, path, method="OPTIONS", origin=FOREIGN)
            acao = hd.get("access-control-allow-origin")
            results.append((f"read {path} OPTIONS still '*'", acao == "*", f"status={st} acao={acao!r}"))
        # Heartbeat's exact shape: extension origin, json, sync lands
        tag = f"{port}_HB"
        st, hd, _ = req(port, "/api/sync", origin=EXT, ctype="application/json; charset=utf-8",
                        body=json.dumps(body_for("/api/sync", tag)).encode())
        results.append(("heartbeat sync (json; charset) lands", st == 200 and landed(tag),
                        f"status={st} landed={landed(tag)} acao={hd.get('access-control-allow-origin')!r}"))
        # CLI save with no body and no content-type keeps working
        # body=None: no body and no Content-Type, as `curl -X POST` sends (urllib adds a form type to b"")
        st, _, _ = req(port, "/api/save", body=None)
        results.append(("CLI empty-body save not refused", st not in (403, 415), f"status={st}"))
    finally:
        srv.terminate()
        try: srv.wait(10)
        except Exception: srv.kill()
        for f in os.listdir(os.path.join(ROOT, "cartridges")) if os.path.isdir(os.path.join(ROOT, "cartridges")) else []:
            if f.startswith(f"heartbeat_wgate_{port}_"):
                os.unlink(os.path.join(ROOT, "cartridges", f))
    npass = sum(ok for _, ok, _ in results)
    for name, ok, d in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  [{d}]")
    print(f"{npass}/{len(results)} pass  (port {port}, mode={'expect-hole' if expect_hole else 'gate'})")
    if expect_hole:
        hole = [ok for n, ok, _ in results if n.startswith("HOLE")]
        print("GAUGE FIRED on known hole" if hole and all(hole) else "GAUGE DID NOT FIRE")
        return 0 if hole and all(hole) else 1
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
