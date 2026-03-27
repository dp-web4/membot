# Membot on Sprout (Jetson Orin Nano 8GB)

Sprout is the fleet's resource-constrained stress test: 8GB unified CPU/GPU memory, ARM64 (aarch64), JetPack 6.x (GLIBC < 2.38). Membot runs here with adaptations that avoid loading the ~2GB SentenceTransformer model.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Sprout (Jetson Orin Nano 8GB)              │
│                                             │
│  Ollama (always-on)                         │
│    ├── qwen3.5:0.8b  ← SAGE daemon LLM     │
│    └── nomic-embed-text ← Membot embeddings │
│                                             │
│  SAGE daemon (:8750)                        │
│    └── MemoryCartridgeIRP → Membot REST API │
│                                             │
│  Membot (:8000)                             │
│    └── embed_text() → Ollama /api/embed     │
│                                             │
│  SNARC (Claude Code hooks)                  │
│    └── membot-bridge → Membot REST API      │
└─────────────────────────────────────────────┘
```

**Key difference from other machines**: Membot uses Ollama for embeddings instead of SentenceTransformer. This saves ~2GB RAM but means cartridges are **not cross-compatible** with machines running SentenceTransformer (see Cartridge Compatibility below).

## Prerequisites

- **Ollama** running with `nomic-embed-text` model pulled:
  ```bash
  ollama pull nomic-embed-text
  curl -s http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"test"}' | python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))"
  # Should print: 768
  ```

- **Python deps** (no sentence-transformers!):
  ```bash
  pip install "mcp[cli]>=1.0.0" fastmcp numpy einops
  # Do NOT install sentence-transformers on Sprout
  ```

- **SAGE daemon** running on port 8750 (optional, for IRP integration)

## Installation

```bash
cd ~/ai-workspace
git clone https://github.com/dp-web4/membot.git
cd membot
pip install -r requirements.txt  # Only installs mcp, numpy, sentence-transformers, einops
# sentence-transformers will fail to import at runtime — that's fine, Ollama takes over
```

## Server Patches (Sprout-Specific)

Two changes to `membot_server.py` make it work on resource-constrained hardware:

### 1. Ollama Embedding Backend

The embedding section was extended to support Ollama as an alternative to SentenceTransformer. Controlled by environment variable:

```
MEMBOT_EMBED_BACKEND=ollama    # Use Ollama (default on Sprout)
MEMBOT_EMBED_BACKEND=st        # Use SentenceTransformer (default elsewhere)
MEMBOT_EMBED_BACKEND=auto      # Prefer Ollama if available, fall back to ST
```

When set to `auto`, the server checks if Ollama is running and has `nomic-embed-text` before deciding. This means the same code works on all machines — Sprout gets Ollama automatically, others get SentenceTransformer.

### 2. FastMCP 3.x Compatibility

The REST API handlers used `.fn()` to call MCP tool functions (FastMCP 2.x pattern). Sprout has FastMCP 3.x where `@mcp.tool()` returns plain functions. Fix: `getattr(fn, 'fn', fn)` — works on both versions.

## Systemd Service

```bash
# Already installed at /etc/systemd/system/membot-sprout.service
sudo systemctl enable membot-sprout
sudo systemctl start membot-sprout
```

Service configuration:
- **MemoryMax**: 2G (hard ceiling — prevents swap thrashing)
- **After**: ollama.service (ensures Ollama is up for embeddings)
- **Writable mode**: enabled (store + save allowed)
- **Port**: 8000

## Verification

```bash
# Service running?
systemctl status membot-sprout

# Health check
curl -s http://localhost:8000/api/status | python3 -m json.tool

# Mount and search
curl -s -X POST http://localhost:8000/api/mount -H 'Content-Type: application/json' \
  -d '{"name":"sage-sprout"}'
curl -s -X POST http://localhost:8000/api/search -H 'Content-Type: application/json' \
  -d '{"query":"test","top_k":3}'

# Memory footprint (should be < 200MB without cartridge loaded)
systemctl status membot-sprout | grep Memory
```

## Cartridge Compatibility

**Cartridges are NOT portable between embedding backends.**

Ollama's `nomic-embed-text` and SentenceTransformer's `nomic-embed-text-v1.5` produce the same model architecture and dimensionality (768-dim) but different numerical vectors due to different inference paths (quantization, tokenizer implementation). This was discovered in the original devlog:

> "Same model weights != same vectors (different inference paths produce subtle drift)"

**Rules**:
- Cartridges built/stored on Sprout (Ollama) can only be searched on Sprout (Ollama)
- Cartridges built on other machines (SentenceTransformer) can only be searched on those machines
- If you need to share content across machines, share the raw text and re-embed on each machine
- The `cartridge_builder.py` uses SentenceTransformer — do NOT use it on Sprout for cartridges you'll search with the Ollama backend

## Memory Budget

| Component | RAM |
|-----------|-----|
| Ollama (idle, model unloaded) | ~200MB |
| Ollama (qwen3.5:0.8b loaded) | ~1.2GB |
| Ollama (nomic-embed-text loaded) | ~300MB |
| SAGE daemon | ~100MB |
| Membot server (no cartridge) | ~70MB |
| Membot (with small cartridge) | ~80MB |
| Claude Code session | ~200MB |
| **Total working set** | **~2.2GB** |
| **System + buffers** | **~3GB** |
| **Available for work** | **~2.2GB** |

Ollama swaps models in/out of GPU memory on demand — embedding requests briefly load `nomic-embed-text`, then it gets evicted when `qwen3.5:0.8b` is needed next. This is slower than keeping both loaded but prevents OOM.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `membot not reachable` in SAGE logs | Service not running | `sudo systemctl start membot-sprout` |
| Embedding timeout | Ollama loading model | First embed after model swap takes ~5s. Subsequent: <100ms |
| Search returns wrong results | Mixed embedding backends | Rebuild cartridge on same machine you'll search on |
| `MemoryMax exceeded` | Cartridge too large | Reduce max_entries or use smaller cartridge |
| Port 8000 already in use | Stale process | `pkill -f membot_server && sudo systemctl restart membot-sprout` |
