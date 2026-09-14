# Membot on Thor (Jetson AGX Thor Developer Kit)

Thor is the fleet's synthesis pool machine: 122GB unified CPU/GPU memory, ARM64 (aarch64), NVIDIA Thor GPU. Membot runs here with standard configuration using SentenceTransformer embeddings loaded in CPU memory.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Thor (Jetson AGX Thor 122GB)              │
│                                             │
│  SNARC (Claude Code hooks)                  │
│    └── membot-bridge → Membot REST API      │
│                                             │
│  Membot (:8000)                             │
│    └── SentenceTransformer (CPU)            │
│        └── nomic-embed-text-v1.5 (768-dim)  │
│                                             │
│  SAGE instances (various ports)             │
│    └── MemoryCartridgeIRP → Membot REST API │
│                                             │
│  Autonomous tracks (systemd)                │
│    ├── thor-sage (00:00, 06:00, 12:00, 18:00)
│    ├── thor-gnosis (every 6h)               │
│    ├── thor-policy (every 6h)               │
│    └── supervisor (03:30)                   │
└─────────────────────────────────────────────┘
```

**Configuration**: Standard setup with SentenceTransformer. Cartridges are compatible with Legion, CBP, McNugget, and Nomad (all use SentenceTransformer backend).

## Prerequisites

- **Python 3.12** with venv support (`python3.12-venv`)
- **122GB RAM** (plenty of headroom for embeddings and SAGE instances)
- **GLIBC 2.39** (supports latest dependencies)

## Installation

```bash
cd ~/ai-workspace
git clone https://github.com/dp-web4/membot.git
cd membot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

This installs:
- `mcp[cli]` - FastMCP server framework
- `sentence-transformers` - Nomic embeddings (~2GB model)
- `torch` - PyTorch backend
- `numpy` - Numerical operations
- `einops` - Tensor operations

## Server Configuration

Thor runs membot as a **systemd user service** (`~/.config/systemd/user/membot.service`, enabled + lingering), upgraded 2026-09-13 to current main:

```bash
systemctl --user status membot        # state
systemctl --user restart membot       # after code/venv changes
```

Unit configuration:
- **ExecStart**: `.venv/bin/python membot_server.py --transport http --host 127.0.0.1 --port 8000 --writable --mount thor-memory`
- **Environment**: `MEMBOT_EMBED_BACKEND=auto` (prefers Ollama `nomic-embed-text`, ~2GB less RAM; falls back to SentenceTransformer in the venv)
- **Port**: 8000 (loopback HTTP)
- **Writable mode**: enabled (store + save operations allowed)
- **Auto-mount**: thor-memory cartridge (engram dual-write recall)

**Post-restart gotcha (observed 2026-09-13):** the startup `--mount` attaches the cart before the embed backend finishes warming — status shows `hamming-only` and searches skip embeddings until you re-`POST /api/mount` once (embeddings then attach, mode flips to `hamming+embedding`). One remount after boot is the current warm-up ritual. Hamming-only recall still works in the meantime, just weaker ranking.

## Multi-cartridge

Current membot has two coexisting layers:

- **Multi-cart pool** — MCP tools only (`multi_mount` / `multi_search` / `multi_list`, spec `docs/RFC/multi-cart-query-spec.md`); one process holds many carts and queries across them with `scope_mode` ranking. Verified on thor: `tests/test_multi_cart.py` (standalone, all 9 pass).
- **REST layer** (`/api/mount`, `/api/search`, …) — one cart per **session slot**. Passing a distinct `session_id` in the request body mounts that cart in its own slot without displacing the `default` session's cart. Verified: thor-memory on `default` + attention-is-all-you-need on `kimi-test`, both searchable simultaneously.

REST consumers coexist via session slots: engram/kimi-style clients that pass their own `session_id` never displace each other. (The kimi-memory adapter currently does not pass one — it shares `default`; see shared-context/kimi-memory.)

## Verification

```bash
# Server running?
ps aux | grep "[m]embot_server"

# Health check
curl -s http://localhost:8000/api/status | python3 -m json.tool

# Expected output:
# {
#     "status": "ok",
#     "cartridge": null,
#     "memories": 0,
#     "gpu": false,
#     "hamming": false,
#     "session_id": "default",
#     "read_only": false
# }

# Run fleet test
cd ~/ai-workspace/membot
python3 tests/fleet/test_membot_health.py TestMembotHealth.test_server_responding

# Mount and search
curl -s -X POST http://localhost:8000/api/mount -H 'Content-Type: application/json' \
  -d '{"name":"sage"}'
curl -s -X POST http://localhost:8000/api/search -H 'Content-Type: application/json' \
  -d '{"query":"consciousness","top_k":3}'
```

## Cartridge Compatibility

**Cartridges ARE portable to/from other SentenceTransformer machines.**

Thor uses the same embedding backend as Legion, CBP, McNugget, and Nomad (SentenceTransformer with `nomic-embed-text-v1.5`). Cartridges can be shared freely between these machines.

**⚠️ NOT compatible with Sprout**: Sprout uses Ollama backend which produces different numerical vectors despite same model architecture.

**Rules**:
- Cartridges built on Thor can be used on Legion, CBP, McNugget, Nomad
- Cartridges from those machines work on Thor
- Do NOT use Sprout cartridges on Thor (different embedding backend)
- Share cartridges by copying `.npz` files to `~/.snarc/membot/cartridges/`

## Memory Budget

| Component | RAM |
|-----------|-----|
| OS + buffers | ~10GB |
| SentenceTransformer model | ~2GB |
| Membot server (no cartridge) | ~200MB |
| Membot (with medium cartridge) | ~500MB |
| Claude Code session | ~500MB |
| SAGE instances (multiple) | ~2GB total |
| Autonomous tracks | ~1GB |
| **Total working set** | **~16GB** |
| **Available for experiments** | **~106GB** |

Thor's 122GB RAM provides massive headroom for:
- Large cartridges (millions of entries)
- Multiple simultaneous Claude sessions
- GPU lattice physics (future: can enable neuromorphic recall)
- Heavy autonomous workloads

## Integration Status

**✓ Working**:
- Membot HTTP server running on port 8000 (systemd user unit, current main as of 2026-09-13)
- REST API responding (`/api/status`, `/api/search`, `/api/store`, `/api/mount` with per-session slots)
- Ollama embedding backend active (`nomic-embed-text`, hamming+embedding mode after warm remount)
- thor-memory cartridge mounted (240 memories, integrity verified)
- Multi-cart layer verified (`tests/test_multi_cart.py`, 9/9)
- Fleet health suite green on thor (9 passed, 1 sprout-only skip)

**⏳ Pending**:
- SAGE IRP integration testing
- Dual-write experiment data collection
- Startup `--mount` embedding attachment (currently needs one manual remount after boot)

## Experiment Participation

Thor participates in the SNARC/Membot dual-write experiment:
- **Experiment log**: `~/.snarc/membot/experiment_log.jsonl`
- **Data collection**: Automatic when SNARC hooks fire
- **Comparison metrics**: FTS5 vs embedding-based recall
- **Timeline**: 2 weeks data collection, analysis in week 3

## Autonomous Track Integration

Thor runs 4 autonomous tracks via systemd:
- **thor-sage** (00:00, 06:00, 12:00, 18:00) - SAGE consciousness research
- **thor-gnosis** (every 6h) - Philosophical exploration
- **thor-policy** (every 6h) - Policy training
- **supervisor** (03:30 daily) - Fleet maintenance

All tracks use Claude Code which triggers SNARC hooks. Membot automatically captures:
- Pre-compact: conversation turns stored with embeddings
- Session-end: dream patterns stored in cartridge
- Session-start: dual-search (FTS5 + embeddings) for context

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on port 8000 | Server not running | Start manually: `.venv/bin/python membot_server.py ...` |
| Embedding timeout | First load initializing | Wait ~30s for model download, subsequent calls <100ms |
| `sentence-transformers not found` | Venv not activated | Use `.venv/bin/python` not bare `python` |
| Search returns empty | Cartridge not mounted | POST to `/api/mount` first |
| Memory growing unbounded | Large cartridge + no GC | Set `max_entries` limit in mount call |

## Future Enhancements

**GPU Lattice Recall**: Thor's NVIDIA GPU can run neuromorphic lattice physics for content-addressable memory with noise tolerance. Requires:
- Compiling `lattice_cuda_v7.so` (CUDA kernel)
- Training Hebbian weights on cartridge
- Enabling `--train` flag in cartridge builder

This is optional — embedding-only search works fine without it.

**Systemd Service**: ✅ Done (2026-08-29 unit, `membot.service`); embedding-only search works fine without the lattice.

---

**Setup Date**: 2026-03-26 · **Last verified current**: 2026-09-13 (kimi-code: fastmcp 4 reinstall, systemd restart, multi-cart + session-slot verification, fleet health green)
**Membot Version**: main @ 6688021 (multi-cart + membox + federate layers present)
**Status**: Operational, current, experiment-ready
