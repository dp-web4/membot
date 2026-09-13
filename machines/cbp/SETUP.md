# CBP (Windows WSL2 desktop, RTX 2060 SUPER 8GB, 32GB)

Two membot processes run here, and both embed on the **CPU**.

| process | port | bind | unit / launcher | consumers |
|---|---|---|---|---|
| shared | 8000 (+ REST bridge 8001) | 0.0.0.0 (unchanged) | user units `membot.service`, `membot-bridge.service` | claude-code, kimi-code, codex seat hooks (each in its own session/cartridge) |
| being | 8010 | 127.0.0.1 | `SAGE/sage/scripts/cbp_being_membot.sh` (cron `@reboot`) | `cbp-being`, cartridge `cbp-being` |

## Why CPU: the being takes priority for the GPU

dp, 2026-09-13: "we've moved membot to cpu on other machines, we should do that here also"
and "the being takes priority".

The card belongs to the SAGE being (`qwen3.8-distill:4b`, about 6 GB of the 8 GB at 16k
context, loaded with `keep_alive -1`). Windows holds another 2 to 2.5 GB. With
`MEMBOT_EMBED_BACKEND` unset, `auto` resolves to ollama whenever `nomic-embed-text` is in
ollama's tag list, and that model could not load beside the being. Measured 2026-09-13:
an ollama embed request hung past 60 s, every membot store timed out (Codex's queued
verification record, the being's own `remember`).

The fix is environment only, no source change:

    MEMBOT_EMBED_BACKEND=st      # sentence-transformers, nomic-ai/nomic-embed-text-v1.5, in-process
    CUDA_VISIBLE_DEVICES=        # torch never sees the card, even if CUDA becomes visible later

For the shared server these are systemd drop-ins, so the base units are untouched:

    ~/.config/systemd/user/membot.service.d/cpu-embed.conf
    ~/.config/systemd/user/membot-bridge.service.d/cpu-embed.conf
    systemctl --user daemon-reload && systemctl --user restart membot membot-bridge

The REST bridge imports `membot_server` in-process and embeds on its own, so it needs the
same environment.

Measured after the change: first embed 18.5 s (model load), then 0.06 s; nothing of
membot's on the GPU.

## Warm-up

The shared server loads the model lazily. `POST /api/embed {"texts":["warm"]}` loads it
without mounting any cartridge, which matters on a server other members' hooks share.
The being's launcher runs `machines/nomad/warm_membot.py cbp-being` after start (that
script dials 127.0.0.1:8010, which is also the being's port here).

## Caveat: older cartridges on the shared server

Until 2026-09-13 the shared server embedded through ollama. The top-level
`machines/README.md` warns that ollama and sentence-transformers vectors are not
numerically identical even for the same model, so recall over memories stored before
this date may rank worse than over new ones. The documented remedy is to re-embed from
the stored text; it has not been done here.

## Verify

    systemctl --user show membot -p Environment          # MEMBOT_EMBED_BACKEND=st CUDA_VISIBLE_DEVICES=
    curl -s -m 60 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/embed -H 'Content-Type: application/json' -d '{"texts":["x"]}'
    ss -ltn | grep 8010                                   # 127.0.0.1:8010
    cat cartridges/cbp-being.cart_manifest.json           # count moves after a remember
