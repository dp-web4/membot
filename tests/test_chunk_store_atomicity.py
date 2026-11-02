"""Regression tests for the chunked-store review's logical-act invariants.

These tests use only synthetic session state and temporary cartridges. They do
not connect to an embedder, an MCP server, or a remote persistence service.
"""
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import membot_server as M


LONG = " ".join(f"word{i}" for i in range(850))


def fake_embed(text, prefix="search_document"):
    return np.full(768, len(text), dtype=np.float32)


class ChunkStoreAtomicityTests(unittest.TestCase):
    def setUp(self):
        self.sid = "chunk-atomicity-test"
        M._sessions.pop(self.sid, None)
        self.state = M._get_session(self.sid)
        self.state["cartridge_name"] = "test"
        self.config = patch.dict(M._server_config, {"read_only": False})
        self.config.start()
        self.addCleanup(self.config.stop)
        self.addCleanup(M._sessions.pop, self.sid, None)

    def test_failed_later_embedding_leaves_no_partial_state(self):
        calls = 0

        def fail_on_second(text, prefix="search_document"):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected embedding failure")
            return fake_embed(text, prefix)

        with patch.object(M, "embed_text", fail_on_second):
            result = M.memory_store(LONG, session_id=self.sid)
        self.assertIn("error", result.lower())
        self.assertEqual(calls, 2)
        self.assertEqual(self.state["texts"], [])
        self.assertIsNone(self.state["embeddings"])
        self.assertIsNone(self.state["binary_corpus"])
        self.assertFalse(self.state["modified"])

    def test_group_identity_survives_save_and_reload(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            result = M.memory_store(LONG, tags="test", session_id=self.sid)
            self.assertIn("Stored memory #", result)
            saved = M.save_cartridge(session_id=self.sid)
            self.assertTrue(saved.startswith("Saved"), saved)
            reloaded = M.load_cartridge_safe(os.path.join(temp, "cartridges", "test.cart.npz"))

        meta = reloaded["per_pattern_meta"]
        self.assertEqual(len(meta), len(reloaded["texts"]))
        group_ids = {part["memory_group_id"] for part in meta}
        self.assertEqual(len(group_ids), 1)
        self.assertIn(next(iter(group_ids)), result)
        self.assertEqual([part["memory_part"] for part in meta], list(range(1, len(meta) + 1)))
        self.assertTrue(all(part["memory_parts"] == len(meta) for part in meta))
        self.state["texts"] = list(reloaded["texts"])
        self.state["embeddings"] = reloaded["embeddings"]
        self.state["binary_corpus"] = (reloaded["embeddings"] > 0).astype(np.uint8)
        self.state["per_pattern_meta"] = meta
        with patch.object(M, "embed_text", fake_embed):
            duplicate = M.memory_store(LONG, tags="test", session_id=self.sid)
        self.assertTrue(duplicate.startswith("Duplicate"), duplicate)
        self.assertEqual(len(self.state["texts"]), len(meta))

    def test_remote_long_store_uses_legacy_single_passage(self):
        self.state["mempack_id"] = "synthetic"
        with patch.object(M, "embed_text", wraps=fake_embed) as embed, \
                patch.object(M, "_supabase_available", return_value=False), \
                patch.object(M, "_mempack_persist_new_pattern") as persist:
            result = M.memory_store(LONG, session_id=self.sid)
        self.assertIn("Stored memory #", result)
        self.assertEqual(self.state["texts"], [LONG])
        embed.assert_called_once()
        persist.assert_not_called()

    def test_remote_pattern_update_does_not_strip_existing_group_metadata(self):
        self.state["mempack_id"] = "synthetic"
        self.state["per_pattern_meta"] = [{"memory_group_id": "existing-group"}]
        with patch.object(M, "_mempack_persist_pattern_update") as persist:
            result = M.mempack_update_pattern_i("replacement", session_id=self.sid)
        self.assertIn("cannot preserve", result)
        persist.assert_not_called()

    def test_short_text_preserves_whitespace(self):
        with patch.object(M, "embed_text", fake_embed):
            M.memory_store("  a note\n", session_id=self.sid)
        self.assertEqual(self.state["texts"], ["  a note\n"])

    def test_store_refuses_a_preexisting_hamming_only_corpus(self):
        self.state["texts"] = ["existing"]
        self.state["binary_corpus"] = np.zeros((1, 768), dtype=np.uint8)
        with patch.object(M, "embed_text", wraps=fake_embed) as embed:
            result = M.memory_store(LONG, session_id=self.sid)
        self.assertIn("do not match", result)
        self.assertEqual(self.state["texts"], ["existing"])
        embed.assert_not_called()

    def test_switching_to_local_cart_clears_old_remote_identity(self):
        self.state["mempack_id"] = "old-remote"
        fake_cart = {"name": "local", "path": "/synthetic/local.cart.npz"}
        loaded = {"texts": [], "embeddings": np.array([]), "format": "npz"}
        with patch.object(M, "find_cartridges", return_value=[fake_cart]), \
                patch.object(M, "load_cartridge_safe", return_value=loaded), \
                patch.object(M, "verify_manifest", return_value=(True, "test")), \
                patch.object(M, "load_manifest", return_value=None), \
                patch.object(M.os.path, "exists", return_value=False):
            result = M.mount_cartridge("local", session_id=self.sid)
        self.assertIn("local", result)
        self.assertIsNone(self.state["mempack_id"])
        self.state["mempack_id"] = "synthetic"
        M.unmount(session_id=self.sid)
        self.assertIsNone(self.state["mempack_id"])

    def test_existing_metadata_is_preserved_when_storing(self):
        with patch.object(M, "embed_text", fake_embed):
            M.memory_store("prior", session_id=self.sid)
            self.state["per_pattern_meta"][0] = {"source": "prior evidence"}
            M.memory_store(LONG, session_id=self.sid)
        self.assertEqual(self.state["per_pattern_meta"][0], {"source": "prior evidence"})

    def test_startup_mount_preserves_groups_across_restart(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed), \
                patch.object(M, "_resolve_session_id", side_effect=lambda sid: sid or self.sid):
            first = M.memory_store(LONG, tags="test", session_id=self.sid)
            group_id = self.state["per_pattern_meta"][0]["memory_group_id"]
            self.assertIn(group_id, first)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))

            # New state, same disk: call the very helper used by --mount.
            M._sessions.pop(self.sid)
            mounted = M._auto_mount("test")
            self.assertIn("test", mounted)
            restarted = M._get_session(self.sid)
            self.assertEqual(restarted["per_pattern_meta"][0]["memory_group_id"], group_id)
            M.memory_store("another note", session_id=self.sid)
            duplicate = M.memory_store(LONG, tags="test", session_id=self.sid)
            self.assertTrue(duplicate.startswith("Duplicate"), duplicate)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            reloaded = M.load_cartridge_safe(os.path.join(temp, "cartridges", "test.cart.npz"))
            self.assertEqual(reloaded["per_pattern_meta"][0]["memory_group_id"], group_id)

    def test_startup_mount_refuses_a_failed_integrity_check(self):
        cart = {"name": "test", "path": "/synthetic/test.cart.npz"}
        data = {"texts": [], "embeddings": np.array([]), "format": "npz"}
        self.state["cartridge_name"] = None
        with patch.object(M, "find_cartridges", return_value=[cart]), \
                patch.object(M, "load_cartridge_safe", return_value=data), \
                patch.object(M, "verify_manifest", return_value=(False, "injected mismatch")), \
                patch.object(M, "_resolve_session_id", side_effect=lambda sid: sid or self.sid):
            result = M._auto_mount("test")
        self.assertIn("SECURITY", result)
        self.assertIsNone(self.state["cartridge_name"])

    def test_manifest_detects_group_identity_tampering(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store(LONG, session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            loaded = M.load_cartridge_safe(path)
            changed = [dict(entry) for entry in loaded["per_pattern_meta"]]
            changed[0]["memory_group_id"] = "tampered-group"
            # Only identity metadata changes; embeddings and passage count are identical.
            M.save_as_npz(path, loaded["embeddings"], loaded["texts"], changed)
            tampered = M.load_cartridge_safe(path)
            ok, detail = M.verify_manifest(path, tampered["embeddings"], len(tampered["texts"]))
            self.assertFalse(ok, detail)

    def test_two_sessions_must_not_silently_overwrite_each_others_groups(self):
        other_sid = "chunk-atomicity-other"
        self.addCleanup(M._sessions.pop, other_sid, None)
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            self.assertIn("Stored memory #", M.memory_store("seed", session_id=self.sid))
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            M._sessions.pop(self.sid)
            first_mount = M.mount_cartridge("test", session_id=self.sid)
            second_mount = M.mount_cartridge("test", session_id=other_sid)
            self.assertNotIn("SECURITY", first_mount)
            self.assertNotIn("SECURITY", second_mount)
            self.assertIn("Stored memory #", M.memory_store(LONG, tags="first", session_id=self.sid))
            self.assertIn("Stored memory #", M.memory_store(LONG, tags="second", session_id=other_sid))
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            second_save = M.save_cartridge(session_id=other_sid)
            loaded = M.load_cartridge_safe(os.path.join(temp, "cartridges", "test.cart.npz"))
            all_text = "\n".join(loaded["texts"])
            # Either fail the stale save visibly, or retain both complete groups.
            self.assertTrue(not second_save.startswith("Saved") or
                            ("[first]" in all_text and "[second]" in all_text), second_save)

    def test_overlapping_stores_cannot_publish_misaligned_groups(self):
        entered = threading.Event()
        release = threading.Event()
        responses = []

        def controlled_embed(text, prefix="search_document"):
            if threading.current_thread().name == "first-store" and not entered.is_set():
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out")
            return fake_embed(text, prefix)

        def store(text):
            responses.append(M.memory_store(text, session_id=self.sid))

        with patch.object(M, "embed_text", controlled_embed):
            first = threading.Thread(target=store, args=(LONG,), name="first-store")
            first.start()
            self.assertTrue(entered.wait(5))
            second = threading.Thread(target=store, args=(LONG + " new ending",), name="second-store")
            second.start()
            second.join(0.1)
            release.set()
            first.join(5)
            second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(responses), 2)
        self.assertTrue(all("Stored memory #" in response for response in responses), responses)
        count = len(self.state["texts"])
        self.assertEqual(count, len(self.state["embeddings"]))
        self.assertEqual(count, len(self.state["binary_corpus"]))
        self.assertEqual(count, len(self.state["per_pattern_meta"]))
        self.assertEqual(len({m["memory_group_id"] for m in self.state["per_pattern_meta"]}), 2)

    def test_save_waits_for_the_whole_store(self):
        entered = threading.Event()
        release = threading.Event()
        saved = []

        def controlled_embed(text, prefix="search_document"):
            if not entered.is_set():
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out")
            return fake_embed(text, prefix)

        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", controlled_embed):
            first = threading.Thread(target=M.memory_store, args=(LONG,), kwargs={"session_id": self.sid})
            first.start()
            self.assertTrue(entered.wait(5))
            saver = threading.Thread(target=lambda: saved.append(M.save_cartridge(session_id=self.sid)))
            saver.start()
            saver.join(0.1)
            self.assertTrue(saver.is_alive(), "save did not wait for the in-flight store")
            release.set()
            first.join(5)
            saver.join(5)
            self.assertFalse(first.is_alive())
            self.assertFalse(saver.is_alive())
            self.assertTrue(saved[0].startswith("Saved"), saved)
            reloaded = M.load_cartridge_safe(os.path.join(temp, "cartridges", "test.cart.npz"))
        self.assertEqual(len(reloaded["texts"]), len(self.state["texts"]))
        self.assertEqual(len(reloaded["per_pattern_meta"]), len(reloaded["texts"]))

    def test_partial_match_never_reuses_another_groups_pieces(self):
        with patch.object(M, "embed_text", fake_embed):
            M.memory_store(LONG, session_id=self.sid)
            original_count = len(self.state["texts"])
            self.assertGreater(original_count, 1)
            # An occupied address is not removed or renumbered: this models a
            # changed/tombstoned entry without pretending list.pop is a tombstone.
            self.state["texts"][1] = "[tombstone]"
            M.memory_store(LONG, session_id=self.sid)
        self.assertEqual(len(self.state["texts"]), original_count * 2)
        new_meta = self.state["per_pattern_meta"][original_count:]
        self.assertEqual(len({part["memory_group_id"] for part in new_meta}), 1)
        self.assertNotEqual(new_meta[0]["memory_group_id"],
                            self.state["per_pattern_meta"][0]["memory_group_id"])


if __name__ == "__main__":
    unittest.main()
