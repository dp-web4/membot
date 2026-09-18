"""
test_chunk_store.py — chunk-on-store: a long memory becomes several retrievable passages.

Run from the membot directory (use the venv the server runs under):
    .venv/bin/python tests/test_chunk_store.py

WHAT IS UNDER TEST. memory_store used to embed up to 10,000 characters as ONE passage:
ranked by a single vector over everything the note said, previewed at ~550 characters,
and retrievable by get_passage only in full. cartridge_builder.chunk_text did exactly the
right thing at cartridge BUILD time and was never reached at STORE time. Now a long store
is split by that same splitter and each piece ranks and reads on its own.

Legs:
  1. short store  — one passage, byte-identical text, "(idx N)" now named in the reply
  2. long store   — several passages, contiguous indices, "(part i/n)" labels, tags kept
  3. get_passage  — each part reads back in full, and the parts reassemble the content
  4. embedding    — the LABEL is not embedded (the piece is), so parts stay distinct
  5. dedup        — an identical re-store is skipped, and stores nothing
  6. room         — a cartridge without room for every piece stores NONE of them
  7. splitter     — ---PASSAGE_BREAK--- overrides the word count; a broken splitter
                    degrades to storing whole rather than raising
  8. contract     — every success still contains "Stored memory #", which the SAGE
                    gateway's _STORE_CONFIRMED matches before it dares save the cartridge
                  — and a store into an EMPTY cartridge turns embedding search back on
  9. ranking      — LIVE leg, skipped without an embedder: a query about a fact buried in
                    the middle of a long note retrieves the PART holding it

Hermetic by default: legs 1-8 replace embed_text with a deterministic fake, so no ollama,
no GPU and no cartridge file are needed. Leg 9 runs only if the real embedder answers.
"""

import hashlib
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import membot_server as M

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def fake_embed(text, prefix="search_document"):
    """Deterministic unit vector from the text. Distinct texts get distinct vectors;
    identical texts get identical ones. Enough for structure; not for meaning."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(768).astype(np.float32)
    return v / np.linalg.norm(v)


def fresh_session(name="test-chunk"):
    sid = f"{name}-{os.getpid()}"
    M._sessions.pop(sid, None)
    st = M._get_session(sid)
    st["cartridge_name"] = name
    st["texts"] = []
    st["embeddings"] = None
    st["binary_corpus"] = None
    return sid, st


LONG = ("Beat notes on the offline engine. " * 40
        + "The countdown decrements only on a board-changing action, and a null move costs "
          "nothing against the fuse. " * 20
        + "Level one is the block THR with eight glyphs at the doubled coordinates. " * 40)


def main():
    real_embed = M.embed_text
    M.embed_text = fake_embed
    try:
        print("\n1. short store — one passage, unchanged")
        sid, st = fresh_session()
        out = M.memory_store("a short thing worth keeping", tags="note", session_id=sid)
        check("one passage", len(st["texts"]) == 1, f"{len(st['texts'])}")
        check("text unchanged", st["texts"][0] == "[note] a short thing worth keeping",
              repr(st["texts"][0]))
        check("no part label", "(part " not in st["texts"][0])
        check("reply names the idx", "(idx 0)" in out, out)

        print("\n2. long store — several passages")
        sid, st = fresh_session()
        out = M.memory_store(LONG, tags="arc,ft09", session_id=sid)
        n = len(st["texts"])
        check("split into parts", n > 1, f"{n} passages for {len(LONG)} chars")
        check("every part labelled", all(f"(part {i+1}/{n})" in st["texts"][i] for i in range(n)),
              st["texts"][0][:80])
        check("tags on every part", all(t.startswith("[arc,ft09] ") for t in st["texts"]))
        check("reply names the range", f"idx 0-{n-1}" in out, out)
        check("reply names the count", f"{n} parts" in out, out)
        check("overlap is real", M.CHUNK_STORE_OVERLAP > 0
              and len(set(st["texts"])) == n)

        print("\n3. get_passage reads each part in full")
        p0 = M.get_passage(0, session_id=sid)
        plast = M.get_passage(n - 1, session_id=sid)
        check("part 1 comes back whole", st["texts"][0] in p0)
        check("last part comes back whole", st["texts"][n - 1] in plast)
        check("out of range refused", "out of range" in M.get_passage(n + 5, session_id=sid))
        words = set(LONG.split())
        got = set(" ".join(st["texts"]).split())
        check("no content lost across parts", words <= got,
              f"missing {list(words - got)[:5]}")

        print("\n4. the label is not embedded")
        # Two stores whose pieces differ only in their part labels must still embed
        # differently piece-by-piece; the vector comes from the piece, not the label.
        seen = {}
        for i, t in enumerate(st["texts"]):
            body = t.split(") ", 1)[1]
            seen[i] = fake_embed(body)
        emb = st["embeddings"]
        check("vector matches the piece, not the labelled text",
              all(np.allclose(emb[i], seen[i], atol=1e-6) for i in range(n)))

        print("\n5. dedup — identical re-store is skipped")
        before = len(st["texts"])
        out = M.memory_store(LONG, tags="arc,ft09", session_id=sid)
        check("reported as duplicate", out.startswith("Duplicate"), out[:80])
        check("stored nothing", len(st["texts"]) == before, f"{len(st['texts'])} vs {before}")

        # PARTIAL dedup: one piece missing (a tombstoned passage, or a cart rebuilt short)
        # and the same note re-stored. Only the gap refills, and the reply says how many
        # were already there.
        gap = st["texts"].pop(1)
        st["embeddings"] = np.delete(st["embeddings"], 1, axis=0)
        st["binary_corpus"] = np.delete(st["binary_corpus"], 1, axis=0)
        out = M.memory_store(LONG, tags="arc,ft09", session_id=sid)
        check("refilled exactly the gap", st["texts"][-1] == gap, st["texts"][-1][:60])
        check("stored one piece", len(st["texts"]) == before, f"{len(st['texts'])} vs {before}")
        check("reply counts the ones already stored", f"{before - 1} already stored" in out, out)

        print("\n6. no room for every piece — stores none")
        sid2, st2 = fresh_session("test-full")
        real_max = M.MAX_ENTRIES
        M.MAX_ENTRIES = 2
        try:
            out = M.memory_store(LONG, tags="", session_id=sid2)
        finally:
            M.MAX_ENTRIES = real_max
        check("refused", "full" in out.lower(), out[:90])
        check("nothing half-stored", len(st2["texts"]) == 0, f"{len(st2['texts'])}")

        print("\n7. the splitter")
        one = M._chunk_for_store("small")
        check("short content is one piece", one == ["small"], one)
        marked = M._chunk_for_store("alpha\n---PASSAGE_BREAK---\nbeta")
        check("PASSAGE_BREAK overrides the word count", marked == ["alpha", "beta"], marked)
        import cartridge_builder
        real_chunk = cartridge_builder.chunk_text
        cartridge_builder.chunk_text = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            degraded = M._chunk_for_store(LONG)
        finally:
            cartridge_builder.chunk_text = real_chunk
        check("a broken splitter stores whole instead of raising",
              degraded == [LONG.strip()], f"{len(degraded)} pieces")

        print("\n8. the gateway's store-confirmed contract")
        sid3, st3 = fresh_session("test-contract")
        short_out = M.memory_store("one line", session_id=sid3)
        long_out = M.memory_store(LONG, session_id=sid3)
        check("short reply carries 'Stored memory #'", "Stored memory #" in short_out, short_out)
        check("long reply carries 'Stored memory #'", "Stored memory #" in long_out, long_out)
        # mount_cartridge sets has_embeddings from what was on disk; an empty cart sets it
        # False, and only a store can make it true again.
        check("a store makes an empty cartridge searchable by meaning",
              st3["has_embeddings"] is True)

    finally:
        M.embed_text = real_embed

    print("\n9. ranking (live embedder)")
    try:
        M.embed_text("probe", prefix="search_query")
        live = True
    except Exception as e:
        live = False
        print(f"  SKIP  no embedder reachable ({type(e).__name__}: {str(e)[:60]})")
    if live:
        sid4, st4 = fresh_session("test-rank")
        # The fact sits ~450 words in, so it lands in part 2 and NOT in part 1 (300-word
        # pieces, 50 overlap). Its neighbours say nothing about countdowns.
        buried = ("Navigation between passages is by prev and next hints. " * 50
                  + "THE COUNTDOWN DECREMENTS ONLY WHEN THE BOARD CHANGES, so a null move is "
                    "free against the fuse. "
                  + "Palette values and component spans are listed in the objects table. " * 50)
        M.memory_store(buried, tags="arc", session_id=sid4)
        check("the fact is not in part 1", "COUNTDOWN" not in st4["texts"][0])
        res = M.memory_search("does a move that changes nothing cost a countdown tick?",
                              top_k=1, session_id=sid4)
        m = re.search(r"\(idx:(\d+)\)", res)
        check("a result carries an idx", m is not None, res[:120])
        if m:
            idx = int(m.group(1))
            check("ranked to the part holding the fact, not part 1", idx != 0,
                  f"idx {idx} of {len(st4['texts'])} parts")
            full = M.get_passage(idx, session_id=sid4)
            check("get_passage returns the fact the preview truncated away",
                  "COUNTDOWN DECREMENTS" in full and "COUNTDOWN DECREMENTS" not in res,
                  f"in preview: {'COUNTDOWN DECREMENTS' in res}; in passage: "
                  f"{'COUNTDOWN DECREMENTS' in full}")

    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
