# Nomad (Windows WSL2 laptop, RTX 4060, 16GB)

Membot here exists for one consumer: **`nomad-being`**, the SAGE raising instance
`nomad-gemma4-e2b`, whose `remember` and `recall` verbs dispatch to this server under
hestia governance. It is not a general service on this box.

Standing it up was the last step in giving that being a working effector: the shared law
already allowed `remember`, and the act failed at execution with
`Connection refused` until this ran.

## Why a machine directory

Three adaptations, none of them hardware limits:

| | Nomad | Standard |
|---|---|---|
| service scope | **user** unit (`systemctl --user`) | system unit |
| bind | **127.0.0.1** | `0.0.0.0` (the server default) |
| port | **8010** | 8000 |

- **User unit, not system.** Everything else on this box that the fleet drives runs under
  the per-user systemd instance (`hestia.service`, `hub-watch.service`,
  `hestia-deploy.timer`). A system unit would need root and would not share their
  lifecycle. Note the WSL consequence: if `systemctl --user` reports
  `Failed to connect to bus`, the user runtime directory has been torn down and the fix
  is `sudo systemctl restart user@1000`, not anything about membot.
- **Loopback bind.** The default `0.0.0.0` would expose a **writable** memory store on
  the tailnet. Nothing off-box should write a being's cartridge.
- **Port 8010** is what SAGE's `HestiaF1aDispatcher` defaults to
  (`membot_endpoint` in `sage/gateway/hestia_dispatch.py`).

`sentence-transformers` works here, so unlike Sprout there is no ollama embedding
backend. First use downloads and loads `nomic-ai/nomic-embed-text-v1.5` from
HuggingFace, which took about 35 seconds and **timed out the being's first
`remember`**. The retry after the model was warm succeeded. If you are watching a first
turn on a cold server, expect that.

## Install

    install -m 0644 machines/nomad/membot-nomad.service ~/.config/systemd/user/membot.service
    systemctl --user daemon-reload
    systemctl --user enable --now membot.service
    journalctl --user -u membot -f

## Verify

    curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8010/mcp   # not 000
    cat cartridges/nomad-being.cart_manifest.json                        # count, fingerprint

## Checking the being's memory is real

From `legion-claude-to-fleet-a-soft-failure-emptied-a-beings-memory-check-yours-2026-09-09.md`.
membot reports some failures as **ordinary text** rather than a JSON-RPC error, and a
client that reads those as success can then save an empty cartridge over a populated
one. legion-being lost 223 memories that way while every beat reported `ok`.

So do not trust a `remember` that returned successfully. Check the store:

    cat cartridges/nomad-being.cart_manifest.json    # "count": 0 is the alarm
    ls -l cartridges/nomad-being.cart.npz            # ~1KB is an empty cart

A healthy write moves both: after `nomad-being`'s first governed `remember`, count went
0 -> 1, the fingerprint changed to match the value the store returned, and the cart grew
from 1053 to 4347 bytes.

**The dispatcher-side fix for the wipe is SAGE `c62fadf0b`, which is NOT on SAGE main as
of 2026-09-09** (it is in PR #56, `legion/mission-artifact`). Until that merges, treat
unattended writing as unsafe: run the being's governed turns attended and check the
manifest after each one. The rule Legion drew from the incident holds regardless: never
save a partial run, abort instead, because the save is the destructive act.
