# Machine-Specific Deployments

Membot runs across a fleet of machines with different hardware. Each subdirectory here contains setup docs, systemd service files, and patch notes for machines that need adaptations beyond the standard install.

## When Do You Need a Machine Directory?

If the standard setup (`pip install -r requirements.txt && python membot_server.py`) works on your machine, you don't need one. Create a directory when:

- The machine can't run `sentence-transformers` (RAM, architecture, or dependency issues)
- You need a custom systemd/launchd service file
- You've applied patches to `membot_server.py` for compatibility
- There are resource constraints that affect configuration (MemoryMax, port conflicts, etc.)

## Current Machine Directories

| Machine | Why | Key Adaptation |
|---------|-----|----------------|
| `sprout/` | Jetson Orin Nano 8GB, ARM64, no GLIBC 2.38 | Ollama embedding backend instead of SentenceTransformer |

## Standard Setup (Most Machines)

```bash
pip install -r requirements.txt
python membot_server.py --transport http --port 8000 --writable
```

This loads SentenceTransformer for embeddings (~2GB RAM), which works on any x86_64 machine with 16+ GB RAM.

## Cartridge Compatibility Warning

Cartridges built with one embedding backend are **not compatible** with another. Ollama and SentenceTransformer produce different numerical vectors for the same text (different inference paths, quantization). If you need to share knowledge across machines with different backends, share the raw text and re-embed locally.
