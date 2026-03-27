# Sprout-Specific Patches to membot_server.py

These changes are applied directly to `membot_server.py` on Sprout. They're designed to work on all machines (auto-detection, version-safe getattr) so they can be merged upstream without breaking anything.

## 1. Ollama Embedding Backend (line ~255)

**What**: Extended the embedding section to support Ollama as an alternative to SentenceTransformer.

**Why**: SentenceTransformer loads ~2GB into RAM. On an 8GB Jetson already running SAGE + Ollama + Claude Code, that causes swap thrashing. Ollama already runs on Sprout and has `nomic-embed-text` — reusing it avoids loading a second model.

**How**: Added `MEMBOT_EMBED_BACKEND` env var with `auto`/`ollama`/`st` options. The `auto` mode checks Ollama availability at first embed call. Ollama embeddings use the standard `/api/embed` endpoint.

**Upstream-safe**: Yes. `auto` mode falls through to SentenceTransformer if Ollama isn't running. No behavior change on machines without Ollama.

## 2. FastMCP 3.x REST Handler Compatibility (lines ~611, ~782, ~828)

**What**: Changed `memory_store.fn(...)` to `getattr(memory_store, 'fn', memory_store)(...)` in REST API handlers.

**Why**: FastMCP 2.x wraps `@mcp.tool()` functions and exposes the original via `.fn`. FastMCP 3.x returns the plain function. Sprout installed 3.x from pip.

**How**: `getattr(fn, 'fn', fn)` works on both versions — tries `.fn` first, falls back to the function itself.

**Upstream-safe**: Yes. Works identically on FastMCP 2.x and 3.x.
