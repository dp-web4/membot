"""Regression tests for the chunked-store review's logical-act invariants.

These tests use only synthetic session state and temporary cartridges. They do
not connect to an embedder, an MCP server, or a remote persistence service.
"""
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
import zipfile
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
        loaded = {"texts": [], "embeddings": np.array([]), "format": "npz"}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(M, "find_cartridges", return_value=[{"name": "local", "path": os.path.join(temp, "local.cart.npz")}]), \
                patch.object(M, "load_cartridge_safe", return_value=loaded), \
                patch.object(M, "verify_manifest", return_value=(True, "test")), \
                patch.object(M, "load_manifest", return_value=None), \
                patch.object(M.os.path, "exists", return_value=False):
            with open(os.path.join(temp, "local.cart.npz"), "wb") as stream:
                stream.write(b"synthetic source")
            result = M.mount_cartridge("local", session_id=self.sid)
        self.assertIn("local", result)
        self.assertNotIn("Mount failed", result)
        self.assertIsNone(self.state["mempack_id"])
        self.state["mempack_id"] = "synthetic"
        M.unmount(session_id=self.sid)
        self.assertIsNone(self.state["mempack_id"])

    def test_missing_source_cannot_mount_a_fabricated_revision(self):
        self.state["mempack_id"] = "old-remote"
        fake = {"name": "local", "path": "/synthetic/missing.cart.npz"}
        loaded = {"texts": [], "embeddings": np.array([]), "format": "npz"}
        with patch.object(M, "find_cartridges", return_value=[fake]), \
                patch.object(M, "load_cartridge_safe", return_value=loaded), \
                patch.object(M, "verify_manifest", return_value=(True, "test")):
            result = M.mount_cartridge("local", session_id=self.sid)
        self.assertIn("failed", result.lower())
        self.assertEqual(self.state["mempack_id"], "old-remote")

    def test_a_session_can_save_its_own_next_revision(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("first", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            M.memory_store("second", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            loaded = M.load_cartridge_safe(os.path.join(temp, "cartridges", "test.cart.npz"))
            self.assertEqual(len(loaded["texts"]), 2)

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

    def test_empty_save_and_legacy_empty_save_both_remount(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            fresh = M.load_cartridge_safe(path)
            self.assertEqual(fresh["embeddings"].shape, (0, 768))
            with patch.object(M, "_mempack_load_for_mutation", return_value=(
                    {}, path, fresh["embeddings"], [])), \
                    patch.object(M, "_mempack_repack_and_persist", return_value=1) as persist:
                M._mempack_append_blobless("synthetic", "new", new_embedding=fake_embed("new"))
                self.assertEqual(persist.call_args.args[2].shape, (1, 768))
            # Recreate the exact legacy shape written when embeddings was None.
            np.savez_compressed(path, embeddings=None, passages=np.array([], dtype=object),
                                version="mcp-v3")
            legacy = M.load_cartridge_safe(path)
            self.assertEqual(legacy["embeddings"].shape, (0, 768))
            # Rewriting an archive under a new manifest is detected. A genuinely
            # legacy sidecar has no whole-content digest and uses the old check.
            self.assertFalse(M.verify_manifest(path, legacy["embeddings"], 0)[0])
            sidecar = path.rsplit(".", 1)[0] + "_manifest.json"
            manifest = M.load_manifest(path)
            del manifest["content_sha256"]
            with open(sidecar, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            M._sessions.pop(self.sid)
            result = M.mount_cartridge("test", session_id=self.sid)
            self.assertNotIn("SECURITY", result)
            self.assertIn("Stored memory #", M.memory_store("after restart", session_id=self.sid))

    def test_missing_embeddings_for_nonempty_cartridge_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                M.save_as_npz(os.path.join(temp, "bad.cart.npz"), None, ["has text"])

    def test_save_preserves_manifest_authority_fields(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            fields = {"briefing": "synthetic", "owner_id": "owner-1", "owner_perms": "rwda",
                      "group_perms": "r", "world_perms": "", "group_id": "group-1",
                      "max_patterns": 99, "cart_type": "agent-memory"}
            manifest_path = path.rsplit(".", 1)[0] + "_manifest.json"
            manifest = M.load_manifest(path)
            manifest.update(fields)
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            M._sessions.pop(self.sid)
            mounted = M.mount_cartridge("test", session_id=self.sid)
            self.assertNotIn("SECURITY", mounted)
            self.assertEqual(M._get_session(self.sid)["world_perms"], "")
            M.memory_store("another", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            saved = M.load_manifest(path)
            self.assertEqual({key: saved.get(key) for key in fields}, fields)
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            self.assertEqual(M._get_session(self.sid)["world_perms"], "")

    def test_save_refuses_unreadable_existing_manifest_before_cart_write(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            with open(path, "rb") as stream:
                before = stream.read()
            manifest_path = path.rsplit(".", 1)[0] + "_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as stream:
                stream.write("not-json")
            M.memory_store("new data", session_id=self.sid)
            self.assertIn("existing manifest is unreadable", M.save_cartridge(session_id=self.sid))
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), before)

    def test_invalid_manifest_values_refuse_before_cart_write(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            with open(path, "rb") as stream:
                before = stream.read()
            manifest_path = path.rsplit(".", 1)[0] + "_manifest.json"
            manifest = M.load_manifest(path)
            manifest["max_patterns"] = "unlimited"
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            M.memory_store("changed", session_id=self.sid)
            self.assertFalse(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), before)

    def test_save_as_carries_source_authority_and_refuses_existing_other_cart(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("private", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            source = os.path.join(temp, "cartridges", "test.cart.npz")
            fields = {"owner_id": "owner-1", "world_perms": "", "cart_type": "agent-memory"}
            manifest = M.load_manifest(source)
            manifest.update(fields)
            with open(source.rsplit(".", 1)[0] + "_manifest.json", "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            self.assertTrue(M.save_cartridge("copy", session_id=self.sid).startswith("Saved"))
            copy = os.path.join(temp, "cartridges", "copy.cart.npz")
            copy_manifest = M.load_manifest(copy)
            self.assertEqual({key: copy_manifest.get(key) for key in fields}, fields)
            copy_manifest.update({"owner_id": "other", "world_perms": "r"})
            with open(copy.rsplit(".", 1)[0] + "_manifest.json", "w", encoding="utf-8") as stream:
                json.dump(copy_manifest, stream)
            with open(copy, "rb") as stream:
                before = stream.read()
            result = M.save_cartridge("copy", session_id=self.sid)
            self.assertFalse(result.startswith("Saved"), result)
            with open(copy, "rb") as stream:
                self.assertEqual(stream.read(), before)

    def test_save_as_refuses_stale_source_authority_and_unknown_manifest_fields(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("private", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            source = os.path.join(temp, "cartridges", "test.cart.npz")
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            manifest_path = source.rsplit(".", 1)[0] + "_manifest.json"
            manifest = M.load_manifest(source)
            manifest["world_perms"] = ""  # Changed after this session mounted world-readable state.
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            self.assertFalse(M.save_cartridge("copy", session_id=self.sid).startswith("Saved"))
            self.assertFalse(os.path.exists(os.path.join(temp, "cartridges", "copy.cart.npz")))
            manifest["world_perms"] = "r"
            manifest["future_policy"] = {"share": False}
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            self.assertFalse(M.save_cartridge("copy", session_id=self.sid).startswith("Saved"))
            self.assertFalse(os.path.exists(os.path.join(temp, "cartridges", "copy.cart.npz")))

    def test_builder_cart_save_preserves_hippocampus_and_pattern_zero(self):
        from cartridge_builder import save_cartridge as build_cartridge

        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            cart_dir = os.path.join(temp, "cartridges")
            hippo_data = bytearray(64)
            hippo_data[28] = 1  # tombstone: losing this byte would resurrect a passage
            hippo, pattern0 = bytes(hippo_data), bytes(range(64))
            path, _, _ = build_cartridge(cart_dir, "test", fake_embed("seed").reshape(1, -1),
                                        ["seed"], metadata=[hippo], pattern0=pattern0)
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            self.assertIn("Stored memory #", M.memory_store("another", session_id=self.sid))
            result = M.save_cartridge(session_id=self.sid)
            self.assertTrue(result.startswith("Saved"), result)
            # The fixture is generated in this test, not external input.
            with np.load(path, allow_pickle=True) as data:
                self.assertIn("hippocampus", data.files)
                self.assertIn("pattern0", data.files)
                self.assertEqual(data["hippocampus"][0].tobytes(), hippo)
                self.assertEqual(data["pattern0"].tobytes(), pattern0)
                self.assertEqual(data["version"].item(), "mcp-v4")
            self.assertEqual(M.load_cartridge_safe(path)["hippocampus"][0]["flags"] & 1, 1)
            self.assertTrue(M.load_manifest(path)["has_hippocampus"])

    def test_missing_manifest_flag_does_not_authorize_dropping_builder_payload(self):
        from cartridge_builder import save_cartridge as build_cartridge

        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            path, _, _ = build_cartridge(os.path.join(temp, "cartridges"), "test",
                                        fake_embed("seed").reshape(1, -1), ["seed"],
                                        metadata=[bytes(64)], pattern0=bytes(range(64)))
            manifest_path = path.rsplit(".", 1)[0] + "_manifest.json"
            manifest = M.load_manifest(path)
            del manifest["has_hippocampus"]  # legacy or incomplete sidecar, not evidence of absence
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            self.assertIn("Stored memory #", M.memory_store("another", session_id=self.sid))
            result = M.save_cartridge(session_id=self.sid)
            self.assertTrue(result.startswith("Saved"), result)
            with np.load(path, allow_pickle=True) as data:
                self.assertTrue({"hippocampus", "pattern0"} <= set(data.files), data.files)

    def test_builder_payload_refuses_changed_old_passages(self):
        from cartridge_builder import save_cartridge as build_cartridge

        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]):
            path, _, _ = build_cartridge(os.path.join(temp, "cartridges"), "test",
                                        fake_embed("seed").reshape(1, -1), ["seed"],
                                        metadata=[bytes(64)], pattern0=bytes(64))
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            with open(path, "rb") as stream:
                before = stream.read()
            M._get_session(self.sid)["texts"][0] = "replaced"
            self.assertFalse(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), before)

    def test_builder_cart_copy_preserves_payload_and_authority(self):
        from cartridge_builder import save_cartridge as build_cartridge

        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]):
            path, _, _ = build_cartridge(os.path.join(temp, "cartridges"), "test",
                                        fake_embed("seed").reshape(1, -1), ["seed"],
                                        metadata=[bytes(64)], pattern0=bytes(range(64)))
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            with open(path, "rb") as stream:
                source = stream.read()
            result = M.save_cartridge("copy", session_id=self.sid)
            self.assertTrue(result.startswith("Saved"), result)
            copied = os.path.join(temp, "cartridges", "copy.cart.npz")
            with np.load(copied, allow_pickle=True) as data:
                self.assertEqual(data["hippocampus"][0].tobytes(), bytes(64))
                self.assertEqual(data["pattern0"].tobytes(), bytes(range(64)))
            self.assertTrue(M.load_manifest(copied)["has_hippocampus"])
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), source)

    def test_object_auxiliary_metadata_is_rejected_without_unpickling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "synthetic.npz")
            np.savez_compressed(path, version="mcp-v4", hippocampus=np.array(["bad"], dtype=object))
            with self.assertRaises(ValueError):
                M._preserved_cart_members(path, 1)

    def test_unknown_npz_member_cannot_be_silently_discarded(self):
        from cartridge_builder import save_cartridge as build_cartridge

        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]):
            path, _, _ = build_cartridge(os.path.join(temp, "cartridges"), "test",
                                        fake_embed("seed").reshape(1, -1), ["seed"])
            with zipfile.ZipFile(path, "a") as cart:
                cart.writestr("future_semantics.npy", b"synthetic unknown field")
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            with open(path, "rb") as stream:
                before = stream.read()
            self.assertFalse(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), before)

    def test_save_preserves_mode_and_refuses_a_hardlinked_destination(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            self.assertIn("Stored memory #", M.memory_store("seed", session_id=self.sid))
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            os.chmod(path, 0o640)
            self.assertIn("Stored memory #", M.memory_store("second", session_id=self.sid))
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o640)
            link = os.path.join(temp, "cartridges", "alias.cart.npz")
            os.link(path, link)
            with open(path, "rb") as stream:
                before = stream.read()
            self.assertIn("Stored memory #", M.memory_store("third", session_id=self.sid))
            self.assertFalse(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            with open(link, "rb") as stream:
                self.assertEqual(stream.read(), before)

    @unittest.skipUnless(all(hasattr(os, name) for name in ("setxattr", "getxattr")), "xattrs unavailable")
    def test_save_preserves_existing_extended_attributes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            try:
                os.setxattr(path, "user.membot_test", b"synthetic")
            except OSError as exc:
                self.skipTest(f"user xattrs unavailable: {exc}")
            M.memory_store("new", session_id=self.sid)
            result = M.save_cartridge(session_id=self.sid)
            self.assertTrue(result.startswith("Saved"), result)
            self.assertEqual(os.getxattr(path, "user.membot_test"), b"synthetic")

    @unittest.skipUnless(all(hasattr(os, name) for name in ("setxattr", "getxattr")), "xattrs unavailable")
    def test_xattr_copy_failure_does_not_replace_existing_cartridge(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            try:
                os.setxattr(path, "user.membot_test", b"synthetic")
            except OSError as exc:
                self.skipTest(f"user xattrs unavailable: {exc}")
            with open(path, "rb") as stream:
                before = stream.read()
            M.memory_store("new", session_id=self.sid)
            with patch.object(M.os, "setxattr", side_effect=PermissionError("synthetic")) as setter:
                result = M.save_cartridge(session_id=self.sid)
            self.assertFalse(result.startswith("Saved"), result)
            setter.assert_called()
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), before)
            self.assertEqual(os.getxattr(path, "user.membot_test"), b"synthetic")
            self.assertFalse(any(name.startswith(".membot-save-") for name in os.listdir(os.path.dirname(path))))

    def test_new_save_obeys_umask_and_temporary_file_is_not_discoverable(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            save_npz = np.savez_compressed
            discovered = []

            def inspect_then_save(*args, **kwargs):
                discovered.extend(cart["name"] for cart in M.find_cartridges())
                return save_npz(*args, **kwargs)

            old_mask = os.umask(0o022)
            try:
                with patch.object(M.np, "savez_compressed", side_effect=inspect_then_save):
                    result = M.save_cartridge(session_id=self.sid)
            finally:
                os.umask(old_mask)
            self.assertTrue(result.startswith("Saved"), result)
            self.assertFalse(any(name.startswith(".membot-save-") for name in discovered), discovered)
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o644)

    def test_in_place_temp_stays_private_until_data_is_written(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            os.chmod(path, 0o644)
            M.memory_store("next", session_id=self.sid)
            writer = np.savez_compressed
            observed = []

            def inspect_then_save(stream, **kwargs):
                observed.append(stat.S_IMODE(os.fstat(stream.fileno()).st_mode))
                return writer(stream, **kwargs)

            with patch.object(M.np, "savez_compressed", side_effect=inspect_then_save):
                self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            self.assertEqual(observed, [0o600])
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o644)

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

    def test_save_does_not_launder_a_changed_mounted_cartridge(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store(LONG, session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            changed = M.load_cartridge_safe(path)
            changed["per_pattern_meta"][0]["memory_group_id"] = "changed-after-mount"
            M.save_as_npz(path, changed["embeddings"], changed["texts"], changed["per_pattern_meta"])
            with open(path, "rb") as stream:
                changed_bytes = stream.read()
            M.memory_store("another note", session_id=self.sid)
            result = M.save_cartridge(session_id=self.sid)
            self.assertIn("changed", result.lower(), result)
            self.assertFalse(result.startswith("Saved"), result)
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), changed_bytes)
            self.assertFalse(M.verify_manifest(path, changed["embeddings"], len(changed["texts"]))[0])

    def test_manifest_covers_text_and_middle_embedding_not_just_group_metadata(self):
        for mutation in ("text", "middle_embedding"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp, \
                    patch.object(M, "BASE_DIR", temp), patch.object(M, "embed_text", fake_embed):
                self.setUp_fresh_state_for_digest()
                M.memory_store(LONG, session_id=self.sid)
                self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
                path = os.path.join(temp, "cartridges", "test.cart.npz")
                loaded = M.load_cartridge_safe(path)
                self.assertGreater(len(loaded["texts"]), 2)
                self.assertTrue(M.verify_manifest(path, loaded["embeddings"], len(loaded["texts"]))[0])
                if mutation == "text":
                    loaded["texts"][1] += " modified"
                else:
                    loaded["embeddings"][1, 0] += 1
                M.save_as_npz(path, loaded["embeddings"], loaded["texts"], loaded["per_pattern_meta"])
                self.assertFalse(M.verify_manifest(path, loaded["embeddings"], len(loaded["texts"]))[0])

    def setUp_fresh_state_for_digest(self):
        M._sessions.pop(self.sid, None)
        self.state = M._get_session(self.sid)
        self.state["cartridge_name"] = "test"

    def test_legacy_and_malformed_content_digests_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            sidecar = path.rsplit(".", 1)[0] + "_manifest.json"
            loaded = M.load_cartridge_safe(path)
            manifest = M.load_manifest(path)
            self.assertRegex(manifest["content_sha256"], r"^[0-9a-f]{64}$")
            ok, detail = M.verify_manifest(path, loaded["embeddings"], len(loaded["texts"]))
            self.assertTrue(ok)
            self.assertIn("not authenticated", detail)
            manifest["content_sha256"] = None
            with open(sidecar, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            self.assertFalse(M.verify_manifest(path, loaded["embeddings"], len(loaded["texts"]))[0])
            del manifest["content_sha256"]  # historical manifest: weaker compatibility check
            with open(sidecar, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            ok, detail = M.verify_manifest(path, loaded["embeddings"], len(loaded["texts"]))
            self.assertTrue(ok)
            self.assertIn("legacy sampled fingerprint only", detail)

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

    def test_two_concurrent_saves_cannot_both_succeed_after_stale_preflight(self):
        other_sid = "chunk-atomicity-race"
        self.addCleanup(M._sessions.pop, other_sid, None)
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            M._sessions.pop(self.sid)
            for sid in (self.sid, other_sid):
                self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=sid))
            M.memory_store(LONG, tags="first", session_id=self.sid)
            M.memory_store(LONG, tags="second", session_id=other_sid)
            second_at_save = threading.Event()
            first_at_save = threading.Event()
            first_finished = threading.Event()
            original_save = M.save_as_npz
            results = {}

            def gated_save(*args, **kwargs):
                # On an unlocked implementation, let the second caller pass
                # preflight before the first publishes, then delay its write
                # until the first whole save finishes. A correct serialization
                # may keep the second out of this function; do not deadlock it.
                if threading.current_thread().name == "second-save":
                    second_at_save.set()
                    if not first_finished.wait(5):
                        raise RuntimeError("first save did not finish")
                else:
                    first_at_save.set()
                    second_at_save.wait(0.5)
                return original_save(*args, **kwargs)

            def save(sid, first=False):
                try:
                    results[sid] = M.save_cartridge(session_id=sid)
                finally:
                    if first:
                        first_finished.set()

            with patch.object(M, "save_as_npz", side_effect=gated_save):
                first = threading.Thread(target=save, args=(self.sid, True), name="first-save")
                second = threading.Thread(target=save, args=(other_sid,), name="second-save")
                first.start()
                self.assertTrue(first_at_save.wait(5), "first did not reach the write")
                second.start()
                first.join(timeout=8)
                second.join(timeout=8)
            self.assertFalse(first.is_alive() or second.is_alive(), "synthetic save timed out")
            self.assertEqual(set(results), {self.sid, other_sid})
            self.assertTrue(results[self.sid].startswith("Saved"), results)
            loaded = M.load_cartridge_safe(os.path.join(temp, "cartridges", "test.cart.npz"))
            contents = "\n".join(loaded["texts"])
            successes = sum(result.startswith("Saved") for result in results.values())
            self.assertTrue(successes < 2 or ("[first]" in contents and "[second]" in contents), results)

    def test_direct_writer_after_preflight_cannot_be_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "CARTRIDGE_DIRS", [os.path.join(temp, "cartridges")]), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            M._sessions.pop(self.sid)
            self.assertNotIn("SECURITY", M.mount_cartridge("test", session_id=self.sid))
            M.memory_store("session edit", session_id=self.sid)
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            old = M.load_cartridge_safe(path)
            external_texts = list(old["texts"]) + ["external writer"]
            external_embeddings = np.vstack([old["embeddings"], fake_embed("external writer")])
            original_save = M.save_as_npz

            def interleaved(*args, **kwargs):
                # Model another writer which does not enter save_cartridge's process lock.
                original_save(path, external_embeddings, external_texts)
                M.save_manifest(path, external_embeddings, len(external_texts))
                return original_save(*args, **kwargs)

            with patch.object(M, "save_as_npz", side_effect=interleaved):
                outcome = M.save_cartridge(session_id=self.sid)
            saved = M.load_cartridge_safe(path)
            self.assertTrue(not outcome.startswith("Saved") or "external writer" in saved["texts"], outcome)

    def test_failed_manifest_publish_must_not_silently_split_the_pair(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(M, "BASE_DIR", temp), \
                patch.object(M, "embed_text", fake_embed):
            M.memory_store("seed", session_id=self.sid)
            self.assertTrue(M.save_cartridge(session_id=self.sid).startswith("Saved"))
            path = os.path.join(temp, "cartridges", "test.cart.npz")
            manifest_path = path.rsplit(".", 1)[0] + "_manifest.json"
            with open(path, "rb") as stream:
                original_cart = stream.read()
            with open(manifest_path, "rb") as stream:
                original_manifest = stream.read()
            M.memory_store("new memory", session_id=self.sid)
            with patch.object(M, "save_manifest", side_effect=OSError("synthetic manifest failure")):
                outcome = M.save_cartridge(session_id=self.sid)
            self.assertFalse(outcome.startswith("Saved"), outcome)
            with open(path, "rb") as stream:
                current_cart = stream.read()
            with open(manifest_path, "rb") as stream:
                current_manifest = stream.read()
            self.assertEqual((current_cart, current_manifest), (original_cart, original_manifest),
                             "a failed pair publication left the two files split")

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
