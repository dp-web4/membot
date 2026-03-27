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

Thor runs membot as a background process (not systemd service yet). Started manually via:

```bash
cd ~/ai-workspace/membot
.venv/bin/python membot_server.py --transport http --port 8000 --writable --mount ~/.snarc/membot/cartridges/sage > membot.log 2>&1 &
```

Configuration:
- **Port**: 8000 (standard HTTP)
- **Writable mode**: enabled (store + save operations allowed)
- **Transport**: HTTP REST API
- **Auto-mount**: sage cartridge (shared with SAGE instances)

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
- Membot HTTP server running on port 8000
- REST API responding (`/api/status`, `/api/search`, `/api/store`)
- SentenceTransformer embeddings loaded (768-dim)
- SNARC rebuilt with membot-bridge (dual-write ready)
- Fleet health tests passing

**⏳ Pending**:
- Systemd service file (currently manual start)
- SAGE IRP integration testing
- Dual-write experiment data collection
- Cartridge auto-mount on restart

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

**Systemd Service**: Convert to proper systemd service like Sprout's `membot-sprout.service` for auto-start on boot and proper resource management.

---

**Setup Date**: 2026-03-26
**SNARC Version**: 0.3.0 (membot-bridge integrated)
**Membot Version**: Latest (REST API)
**Status**: Operational, experiment-ready
