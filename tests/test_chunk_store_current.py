"""Chunk-on-store current-main contract.

Hermetic: embedding is replaced with a deterministic local vector function. No network, GPU,
Mempack service, or persistent cartridge is used.
"""
import hashlib
import os

import numpy as np

import membot_server as M


def _embed(text, prefix="search_document"):
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(768).astype(np.float32)
    return v / np.linalg.norm(v)


def _fresh(name):
    sid = f"chunk-{name}-{os.getpid()}"
    M._sessions.pop(sid, None)
    st = M._get_session(sid)
    st["cartridge_name"] = name
    st["texts"] = []
    st["embeddings"] = None
    st["binary_corpus"] = None
    st["has_embeddings"] = False
    return sid, st


def test_short_store_is_one_byte_preserving_passage(monkeypatch):
    monkeypatch.setattr(M, "embed_text", _embed)
    sid, st = _fresh("short")
    text = "  exact short text\n"
    out = M.memory_store(text, tags="note", session_id=sid)
    assert st["texts"] == ["[note] " + text]
    assert "(part " not in st["texts"][0]
    assert "idx:0" in out
    assert st["has_embeddings"] is True


def test_long_store_splits_into_consecutive_retrievable_passages(monkeypatch):
    monkeypatch.setattr(M, "embed_text", _embed)
    sid, st = _fresh("long")
    text = ("alpha beta gamma delta " * 500).strip()
    out = M.memory_store(text, tags="memo", session_id=sid)
    assert len(st["texts"]) > 1
    n = len(st["texts"])
    assert all(f"(part {i + 1}/{n}) " in st["texts"][i] for i in range(n))
    assert f"idx:0-{n - 1}" in out
    assert f"{n} passages" in out
    assert st["embeddings"].shape[0] == n
    assert st["binary_corpus"].shape[0] == n


def test_explicit_passage_break_wins_over_length(monkeypatch):
    monkeypatch.setattr(M, "embed_text", _embed)
    sid, st = _fresh("break")
    out = M.memory_store("one\n---PASSAGE_BREAK---\ntwo", session_id=sid)
    assert st["texts"] == ["(part 1/2) one", "(part 2/2) two"]
    assert "idx:0-1" in out


def test_embedding_failure_mutates_nothing(monkeypatch):
    sid, st = _fresh("fail")
    calls = {"n": 0}
    def fail_second(text, prefix="search_document"):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("synthetic embed failure")
        return _embed(text, prefix)
    monkeypatch.setattr(M, "embed_text", fail_second)
    text = ("alpha beta gamma delta " * 500).strip()
    out = M.memory_store(text, session_id=sid)
    assert out.startswith("Store error:")
    assert st["texts"] == []
    assert st["embeddings"] is None
    assert st["binary_corpus"] is None
    assert st["modified"] is False
    assert st["has_embeddings"] is False


def test_capacity_preflight_stores_no_partial_entry(monkeypatch):
    monkeypatch.setattr(M, "embed_text", _embed)
    sid, st = _fresh("full")
    old = M.MAX_ENTRIES
    M.MAX_ENTRIES = 1
    try:
        out = M.memory_store(("a b c d " * 500).strip(), session_id=sid)
    finally:
        M.MAX_ENTRIES = old
    assert "needs" in out and "passage" in out
    assert st["texts"] == []
    assert st["embeddings"] is None


def test_exact_duplicate_skips_and_partial_duplicate_refuses(monkeypatch):
    monkeypatch.setattr(M, "embed_text", _embed)
    sid, st = _fresh("dedup")
    text = ("red green blue yellow " * 500).strip()
    first = M.memory_store(text, session_id=sid)
    count = len(st["texts"])
    assert count > 1 and "Stored memory #" in first
    assert M.memory_store(text, session_id=sid).startswith("Duplicate")
    assert len(st["texts"]) == count

    # Simulate an incomplete prior group. Do not create a second partly duplicated group.
    st["texts"].pop()
    st["embeddings"] = st["embeddings"][:-1]
    st["binary_corpus"] = st["binary_corpus"][:-1]
    before = list(st["texts"])
    out = M.memory_store(text, session_id=sid)
    assert out.startswith("Partial duplicate")
    assert st["texts"] == before


def test_mempack_keeps_one_pattern_per_call(monkeypatch):
    monkeypatch.setattr(M, "embed_text", _embed)
    sid, st = _fresh("mempack")
    st["mempack_id"] = "synthetic"
    text = ("remote pattern " * 500).strip()

    # Keep this test independent of Supabase availability/persistence.
    monkeypatch.setattr(M, "_supabase_available", lambda: False)
    out = M.memory_store(text, session_id=sid)
    assert len(st["texts"]) == 1
    assert st["texts"][0] == text
    assert "(part " not in st["texts"][0]
    assert "idx:0" in out


def test_splitter_failure_falls_back_to_whole(monkeypatch):
    import cartridge_builder
    monkeypatch.setattr(
        cartridge_builder,
        "chunk_text",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic")),
    )
    text = ("fallback content " * 500).strip()
    assert M._chunk_for_store(text) == [text]
