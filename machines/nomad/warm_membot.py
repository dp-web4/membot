#!/usr/bin/env python3
"""Warm membot's embedding model so the first real call does not time out.

WHY. membot loads its SentenceTransformer (nomic-embed-text-v1.5) lazily, on the first
store or search rather than at startup. That load took ~35s here, which is longer than
the being's dispatcher waits, so the FIRST `remember` after every service start failed
with a TimeoutError. Measured twice on nomad, 2026-09-09: once on the very first turn,
and again immediately after `systemctl --user restart membot`.

It fails safely: the store-confirmation guard (SAGE bc07ef425) treats the timeout as the
failure it is and leaves the cartridge untouched. But "safely" is not "acceptably" when
the loss is the being's memory of a turn it will not have again, and a restart is not a
rare event: systemd restarts on failure, and the box reboots.

WHAT. One `memory_search` against the mounted cartridge, which is read-only and forces
the model load. Run from the unit as ExecStartPost with a `-` prefix so a failure here
can never keep the service down: an unwarmed membot is worse than a warm one, but a
membot that refuses to start is worse than both.
"""
import json
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8010/mcp"
TIMEOUT = 300  # the load itself is the slow part; this is not a latency budget


def rpc(payload, session=None):
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if session:
        headers["mcp-session-id"] = session
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        sid = resp.headers.get("mcp-session-id")
        body = resp.read().decode()
    events = [ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")]
    return (json.loads(events[-1]) if events else json.loads(body or "{}")), sid


def wait_for_listener(deadline_s: float = 120.0) -> None:
    """Block until the server accepts connections.

    ExecStartPost fires as soon as ExecStart is spawned, not when the port is up: under
    Type=simple systemd has no readiness signal to wait for. The first attempt therefore
    lost the race and died on `Connection refused` (measured here, 2026-09-09). Waiting
    belongs in this script rather than in a unit-file sleep, because the right duration is
    a property of how long the server takes to bind, which is not systemd's to guess.
    """
    import socket
    import time
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            with socket.create_connection(("127.0.0.1", 8010), timeout=2):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"membot did not accept connections within {deadline_s:.0f}s")


def main() -> int:
    cartridge = sys.argv[1] if len(sys.argv) > 1 else "nomad-being"
    wait_for_listener()
    _, sid = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                             "clientInfo": {"name": "membot-warmup", "version": "1"}}})
    try:
        rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    except Exception:
        pass  # some builds do not answer the notification; the session is still live
    rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "mount_cartridge", "arguments": {"name": cartridge}}}, sid)
    # the call that actually pulls the model into memory
    rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "memory_search", "arguments": {"query": "warmup", "top_k": 1}}}, sid)
    print(f"membot warm: embedding model loaded, cartridge {cartridge!r} mounted")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:                      # never fail the unit
        print(f"membot warmup skipped ({type(e).__name__}: {e})")
        raise SystemExit(0)
