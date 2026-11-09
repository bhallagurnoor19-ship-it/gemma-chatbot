"""
Automated Test Suite for Unified RAG Phase 4.2
Validates:
1. Retriever excludes old generation after manifest commit.
2. Retriever excludes uncommitted / staged generations.
3. Retriever excludes chunks from files with no active manifest entry.
4. Retriever returns only active generation chunks.
5. Orphan cleanup removes stale/orphaned generations from ChromaDB.
6. Orphan cleanup preserves all active chunks in ChromaDB.
7. Cleanup failure does not invalidate successful indexing or break retrieval.
8. Interrupted indexing recovery: crash after staging before commit.
9. Interrupted indexing recovery: crash after commit before old chunk deletion.
10. Independent handling of document and codebase collections.
11. IncrementalIndexer.sync() incorporates orphan cleanup safely.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.config import (
    CODEBASE_COLLECTION_NAME,
    DOCUMENT_COLLECTION_NAME,
)
from rag.indexer import IncrementalIndexer
from rag.retriever import RetrievedChunk, UnifiedRetriever, RetrievalStatus
from rag.vector_store import VectorStoreManager


class DeterministicMockEmbedder:
    """Deterministic mock embedding service for isolated, fast unit testing."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        return self._make_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._make_vector(t) for t in texts]

    def _make_vector(self, text: str) -> List[float]:
        val = sum(ord(c) for c in text) % 1000
        vec = [0.0] * self.dimension
        vec[0] = float(val) / 1000.0
        vec[1] = 0.5
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class TestRAGPhase42ActiveGenerationAndCleanup(unittest.TestCase):
    """Phase 4.2 Test Suite: Active Generation Filtering and Orphan Cleanup."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="rag_test_p42_")
        self.chroma_dir = os.path.join(self.test_dir, "chroma")
        self.manifest_dir = os.path.join(self.test_dir, "manifests")
        self.manifest_path = os.path.join(self.manifest_dir, "index_manifest.json")
        os.makedirs(self.manifest_dir, exist_ok=True)

        self.mock_embedder = DeterministicMockEmbedder(dimension=768)
        self.vector_store = VectorStoreManager(persist_dir=self.chroma_dir)
        self.vector_store.initialize()

        self.indexer = IncrementalIndexer(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            base_dir=self.test_dir,
        )

        self.retriever = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            top_k=5,
            distance_threshold=1.5,
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass

    def _write_manifest(self, manifest_data: Dict[str, Any]):
        """Helper to write manifest JSON atomically."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    # ── Test 1: Retriever excludes old generation after commit ─────────────────
    def test_01_retriever_excludes_old_generation(self):
        """Verify that stale generation chunks remaining in ChromaDB are filtered out."""
        # Old generation (g1)
        old_id = "doc:guide.txt:chunk0:g1"
        old_text = "This is the outdated guide version 1.0."
        old_meta = {
            "source_path": "documents/guide.txt",
            "file_name": "guide.txt",
            "category": "document",
            "generation_id": "g1",
        }

        # Active generation (g2)
        active_id = "doc:guide.txt:chunk0:g2"
        active_text = "This is the active new guide version 2.0 with updated instructions."
        active_meta = {
            "source_path": "documents/guide.txt",
            "file_name": "guide.txt",
            "category": "document",
            "generation_id": "g2",
        }

        # Seed both into ChromaDB
        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            [old_id, active_id],
            [old_text, active_text],
            self.mock_embedder.embed_batch([old_text, active_text]),
            [old_meta, active_meta],
        )

        # Committed manifest points ONLY to g2
        manifest = {
            "version": "1.0",
            "last_updated": "2026-11-06T00:00:00",
            "documents": {
                "documents/guide.txt": {
                    "sha256": "hash2",
                    "generation_id": "g2",
                    "chunk_ids": [active_id],
                    "chunk_count": 1,
                }
            },
            "codebase": {},
        }
        self._write_manifest(manifest)

        # Query retriever
        response = self.retriever.retrieve("guide version instructions", mode="documents")
        self.assertEqual(response.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(response.chunks), 1)

        result_chunk = response.chunks[0]
        self.assertEqual(result_chunk.chunk_id, active_id)
        self.assertIn("version 2.0", result_chunk.content)
        self.assertNotIn("version 1.0", result_chunk.content)

    # ── Test 2: Retriever excludes uncommitted/staged generations ─────────────
    def test_02_retriever_excludes_uncommitted_generations(self):
        """Verify that newly staged chunks from an uncommitted run are excluded."""
        # Committed active generation (gen_active)
        active_id = "doc:policy.md:c0:gen_active"
        active_text = "Standard company policy document active version."
        active_meta = {
            "source_path": "documents/policy.md",
            "file_name": "policy.md",
            "category": "document",
            "generation_id": "gen_active",
        }

        # Uncommitted/staged generation (gen_staged)
        staged_id = "doc:policy.md:c0:gen_staged"
        staged_text = "Draft uncommitted policy document staging version."
        staged_meta = {
            "source_path": "documents/policy.md",
            "file_name": "policy.md",
            "category": "document",
            "generation_id": "gen_staged",
        }

        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            [active_id, staged_id],
            [active_text, staged_text],
            self.mock_embedder.embed_batch([active_text, staged_text]),
            [active_meta, staged_meta],
        )

        # Manifest contains ONLY gen_active
        manifest = {
            "version": "1.0",
            "documents": {
                "documents/policy.md": {
                    "sha256": "h_act",
                    "generation_id": "gen_active",
                    "chunk_ids": [active_id],
                }
            },
            "codebase": {},
        }
        self._write_manifest(manifest)

        response = self.retriever.retrieve("policy document", mode="documents")
        self.assertEqual(response.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(response.chunks), 1)
        self.assertEqual(response.chunks[0].chunk_id, active_id)
        self.assertNotIn("uncommitted", response.chunks[0].content)

    # ── Test 3: Retriever excludes files with no active manifest entry ────────
    def test_03_retriever_excludes_files_with_no_active_manifest_entry(self):
        """Verify that chunks from deleted or unindexed files are not returned."""
        # File 1: active file in manifest
        f1_id = "doc:active.txt:c0:g10"
        f1_text = "Active project configuration details."
        f1_meta = {
            "source_path": "documents/active.txt",
            "file_name": "active.txt",
            "category": "document",
            "generation_id": "g10",
        }

        # File 2: deleted file left in ChromaDB, absent from manifest
        f2_id = "doc:deleted.txt:c0:g01"
        f2_text = "Deleted legacy configuration details."
        f2_meta = {
            "source_path": "documents/deleted.txt",
            "file_name": "deleted.txt",
            "category": "document",
            "generation_id": "g01",
        }

        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            [f1_id, f2_id],
            [f1_text, f2_text],
            self.mock_embedder.embed_batch([f1_text, f2_text]),
            [f1_meta, f2_meta],
        )

        # Manifest only has documents/active.txt
        manifest = {
            "version": "1.0",
            "documents": {
                "documents/active.txt": {
                    "sha256": "h_act",
                    "generation_id": "g10",
                    "chunk_ids": [f1_id],
                }
            },
            "codebase": {},
        }
        self._write_manifest(manifest)

        response = self.retriever.retrieve("configuration details", mode="documents")
        self.assertEqual(response.status, RetrievalStatus.SUCCESS)
        # Deleted file must be strictly excluded
        returned_sources = [c.source_path for c in response.chunks]
        self.assertIn("documents/active.txt", returned_sources)
        self.assertNotIn("documents/deleted.txt", returned_sources)

    # ── Test 4: Retriever returns only active-generation chunks ────────────────
    def test_04_retriever_returns_only_active_generation_chunks(self):
        """Verify bulk candidate filtering retains only active generation chunks."""
        doc_ids = []
        doc_texts = []
        doc_metas = []
        active_ids = []

        # Create 5 files, each having an old generation (g_old) and active generation (g_new)
        for i in range(5):
            fname = f"doc_{i}.txt"
            spath = f"documents/{fname}"

            old_id = f"doc:{fname}:c0:g_old"
            active_id = f"doc:{fname}:c0:g_new"
            active_ids.append(active_id)

            doc_ids.extend([old_id, active_id])
            doc_texts.extend([f"Old content {i}", f"Active content {i}"])
            doc_metas.extend([
                {"source_path": spath, "file_name": fname, "category": "document", "generation_id": "g_old"},
                {"source_path": spath, "file_name": fname, "category": "document", "generation_id": "g_new"},
            ])

        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            doc_ids,
            doc_texts,
            self.mock_embedder.embed_batch(doc_texts),
            doc_metas,
        )

        # Manifest has only g_new for each file
        manifest = {
            "version": "1.0",
            "documents": {
                f"documents/doc_{i}.txt": {
                    "generation_id": "g_new",
                    "chunk_ids": [f"doc:doc_{i}.txt:c0:g_new"],
                }
                for i in range(5)
            },
            "codebase": {},
        }
        self._write_manifest(manifest)

        response = self.retriever.retrieve("content", mode="documents", top_k=5)
        self.assertEqual(response.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(response.chunks), 5)
        for chunk in response.chunks:
            self.assertIn(chunk.chunk_id, active_ids)
            self.assertTrue(chunk.chunk_id.endswith(":g_new"))
            self.assertFalse(chunk.chunk_id.endswith(":g_old"))

    # ── Test 5: Orphan cleanup removes stale generations from ChromaDB ────────
    def test_05_orphan_cleanup_removes_stale_generations(self):
        """Verify that IncrementalIndexer.clean_orphaned_generations deletes stale chunks."""
        # Insert 3 active chunks and 3 orphaned chunks
        active_ids = ["doc:fileA.txt:c0:g2", "doc:fileB.txt:c0:g2", "doc:fileC.txt:c0:g1"]
        orphan_ids = ["doc:fileA.txt:c0:g1", "doc:fileB.txt:c0:g1", "doc:deleted.txt:c0:g0"]

        all_ids = active_ids + orphan_ids
        all_texts = [f"Text for {cid}" for cid in all_ids]
        all_metas = [{"source_path": cid.split(":")[1], "category": "document"} for cid in all_ids]

        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            all_ids,
            all_texts,
            self.mock_embedder.embed_batch(all_texts),
            all_metas,
        )

        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        self.assertEqual(col.count(), 6)

        # Manifest registers only active_ids
        manifest = {
            "version": "1.0",
            "documents": {
                "fileA.txt": {"generation_id": "g2", "chunk_ids": ["doc:fileA.txt:c0:g2"]},
                "fileB.txt": {"generation_id": "g2", "chunk_ids": ["doc:fileB.txt:c0:g2"]},
                "fileC.txt": {"generation_id": "g1", "chunk_ids": ["doc:fileC.txt:c0:g1"]},
            },
            "codebase": {},
        }
        self._write_manifest(manifest)

        # Run orphan cleanup
        summary = self.indexer.clean_orphaned_generations(DOCUMENT_COLLECTION_NAME)

        self.assertEqual(summary["total_chunks_scanned"], 6)
        self.assertEqual(summary["active_chunks_retained"], 3)
        self.assertEqual(summary["orphaned_chunks_deleted"], 3)
        self.assertEqual(summary["errors"], 0)

        # Verify in ChromaDB
        self.assertEqual(col.count(), 3)
        remaining = col.get(ids=all_ids)
        remaining_ids = set(remaining.get("ids", []))
        self.assertEqual(remaining_ids, set(active_ids))
        for oid in orphan_ids:
            self.assertNotIn(oid, remaining_ids)

    # ── Test 6: Orphan cleanup preserves all active chunks in ChromaDB ────────
    def test_06_orphan_cleanup_preserves_all_active_generations(self):
        """Verify that active chunks are never touched when running cleanup."""
        active_ids = ["doc:f1.txt:c0:g1", "doc:f2.txt:c0:g1"]
        texts = ["Text 1", "Text 2"]
        metas = [
            {"source_path": "f1.txt", "category": "document", "generation_id": "g1"},
            {"source_path": "f2.txt", "category": "document", "generation_id": "g1"},
        ]

        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            active_ids,
            texts,
            self.mock_embedder.embed_batch(texts),
            metas,
        )

        manifest = {
            "version": "1.0",
            "documents": {
                "f1.txt": {"generation_id": "g1", "chunk_ids": ["doc:f1.txt:c0:g1"]},
                "f2.txt": {"generation_id": "g1", "chunk_ids": ["doc:f2.txt:c0:g1"]},
            },
            "codebase": {},
        }
        self._write_manifest(manifest)

        summary = self.indexer.clean_orphaned_generations()
        self.assertEqual(summary["orphaned_chunks_deleted"], 0)
        self.assertEqual(summary["active_chunks_retained"], 2)

        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        self.assertEqual(col.count(), 2)

    # ── Test 7: Cleanup failure does not invalidate successful indexing ───────
    def test_07_cleanup_failure_does_not_invalidate_indexing(self):
        """Verify that when vector store deletion fails, cleanup logs warning without raising."""
        active_id = "doc:ok.txt:c0:g2"
        orphan_id = "doc:ok.txt:c0:g1"
        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            [active_id, orphan_id],
            ["Active text", "Orphan text"],
            self.mock_embedder.embed_batch(["Active text", "Orphan text"]),
            [{"source_path": "ok.txt"}, {"source_path": "ok.txt"}],
        )

        manifest = {
            "version": "1.0",
            "documents": {
                "ok.txt": {"generation_id": "g2", "chunk_ids": [active_id]},
            },
            "codebase": {},
        }
        self._write_manifest(manifest)

        # Mock delete_by_ids to simulate a failure
        orig_delete = self.vector_store.delete_by_ids
        def failing_delete(col_name, ids):
            raise RuntimeError("Simulated storage delete error")

        self.vector_store.delete_by_ids = failing_delete

        try:
            summary = self.indexer.clean_orphaned_generations(DOCUMENT_COLLECTION_NAME)
            # Must not raise exception, summary reports error
            self.assertGreater(summary["errors"], 0)
            self.assertEqual(summary["orphaned_chunks_deleted"], 0)

            # Even though orphan was not deleted, retriever still filters it out!
            response = self.retriever.retrieve("text", mode="documents")
            self.assertEqual(response.status, RetrievalStatus.SUCCESS)
            self.assertEqual(len(response.chunks), 1)
            self.assertEqual(response.chunks[0].chunk_id, active_id)
        finally:
            self.vector_store.delete_by_ids = orig_delete

    # ── Test 8: Recovery after interrupted indexing (staging crash) ───────────
    def test_08_interrupted_indexing_recovery_staging_crash(self):
        """Verify recovery when a crash happens after staging chunks into ChromaDB before commit."""
        # 1. State before crash: file was at generation g1
        g1_id = "code:main.py:func:1_10:g1"
        g1_text = "def main():\n    return 'v1'"
        g1_meta = {"source_path": "main.py", "category": "codebase", "generation_id": "g1"}

        self.vector_store.add_documents(
            CODEBASE_COLLECTION_NAME,
            [g1_id],
            [g1_text],
            self.mock_embedder.embed_batch([g1_text]),
            [g1_meta],
        )

        manifest = {
            "version": "1.0",
            "documents": {},
            "codebase": {
                "main.py": {
                    "sha256": "h_v1",
                    "generation_id": "g1",
                    "chunk_ids": [g1_id],
                }
            },
        }
        self._write_manifest(manifest)

        # 2. Interruption: g2 was staged in ChromaDB, but manifest commit never happened
        g2_id = "code:main.py:func:1_10:g2"
        g2_text = "def main():\n    return 'v2_staged_never_committed'"
        g2_meta = {"source_path": "main.py", "category": "codebase", "generation_id": "g2"}

        self.vector_store.add_documents(
            CODEBASE_COLLECTION_NAME,
            [g2_id],
            [g2_text],
            self.mock_embedder.embed_batch([g2_text]),
            [g2_meta],
        )

        # 3. Retriever queries: must return g1, never uncommitted g2
        retrieval = self.retriever.retrieve("main function", mode="codebase")
        self.assertEqual(retrieval.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(retrieval.chunks), 1)
        self.assertEqual(retrieval.chunks[0].chunk_id, g1_id)
        self.assertIn("v1", retrieval.chunks[0].content)

        # 4. Orphan cleanup purges the uncommitted g2 chunk
        cleanup_summary = self.indexer.clean_orphaned_generations(CODEBASE_COLLECTION_NAME)
        self.assertEqual(cleanup_summary["orphaned_chunks_deleted"], 1)

        # Verify only g1 remains in ChromaDB
        code_col = self.vector_store.get_collection(CODEBASE_COLLECTION_NAME)
        self.assertEqual(code_col.count(), 1)
        self.assertEqual(code_col.get(ids=[g1_id])["ids"], [g1_id])
        self.assertEqual(code_col.get(ids=[g2_id])["ids"], [])

    # ── Test 9: Recovery after interrupted indexing (cleanup crash) ───────────
    def test_09_interrupted_indexing_recovery_cleanup_crash(self):
        """Verify recovery when manifest commit succeeded but old-generation deletion crashed."""
        # 1. State: manifest was committed to g2, but g1 was not deleted due to crash
        g1_id = "code:app.py:start:1_10:g1"
        g2_id = "code:app.py:start:1_10:g2"

        self.vector_store.add_documents(
            CODEBASE_COLLECTION_NAME,
            [g1_id, g2_id],
            ["start server v1", "start server v2 committed"],
            self.mock_embedder.embed_batch(["start server v1", "start server v2 committed"]),
            [
                {"source_path": "app.py", "category": "codebase", "generation_id": "g1"},
                {"source_path": "app.py", "category": "codebase", "generation_id": "g2"},
            ],
        )

        manifest = {
            "version": "1.0",
            "documents": {},
            "codebase": {
                "app.py": {
                    "sha256": "h_v2",
                    "generation_id": "g2",
                    "chunk_ids": [g2_id],
                }
            },
        }
        self._write_manifest(manifest)

        # 2. Retriever immediately filters out g1 and serves g2
        res = self.retriever.retrieve("start server", mode="codebase")
        self.assertEqual(res.status, RetrievalStatus.SUCCESS)
        self.assertEqual(len(res.chunks), 1)
        self.assertEqual(res.chunks[0].chunk_id, g2_id)
        self.assertIn("v2 committed", res.chunks[0].content)

        # 3. Running orphan cleanup purges stale g1
        cleanup_summary = self.indexer.clean_orphaned_generations(CODEBASE_COLLECTION_NAME)
        self.assertEqual(cleanup_summary["orphaned_chunks_deleted"], 1)
        code_col = self.vector_store.get_collection(CODEBASE_COLLECTION_NAME)
        self.assertEqual(code_col.count(), 1)
        self.assertEqual(code_col.get(ids=[g2_id])["ids"], [g2_id])

    # ── Test 10: Independent collections handling ─────────────────────────────
    def test_10_independent_collections_handling(self):
        """Verify document and codebase collections are handled independently."""
        # Insert document chunk and codebase chunk
        doc_id = "doc:manual.pdf:c0:g_doc"
        code_id = "code:api.py:handler:g_code"

        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            [doc_id],
            ["User manual content"],
            self.mock_embedder.embed_batch(["User manual content"]),
            [{"source_path": "manual.pdf", "category": "document", "generation_id": "g_doc"}],
        )
        self.vector_store.add_documents(
            CODEBASE_COLLECTION_NAME,
            [code_id],
            ["API handler code"],
            self.mock_embedder.embed_batch(["API handler code"]),
            [{"source_path": "api.py", "category": "codebase", "generation_id": "g_code"}],
        )

        # Manifest has manual.pdf active, but codebase is EMPTY
        manifest = {
            "version": "1.0",
            "documents": {
                "manual.pdf": {"generation_id": "g_doc", "chunk_ids": [doc_id]},
            },
            "codebase": {},
        }
        self._write_manifest(manifest)

        # Clean only documents collection
        doc_cleanup = self.indexer.clean_orphaned_generations(DOCUMENT_COLLECTION_NAME)
        self.assertEqual(doc_cleanup["details"][DOCUMENT_COLLECTION_NAME]["deleted"], 0)
        self.assertNotIn(CODEBASE_COLLECTION_NAME, doc_cleanup["details"])

        # Retrieval in document mode finds manual.pdf
        doc_res = self.retriever.retrieve("manual", mode="documents")
        self.assertEqual(len(doc_res.chunks), 1)

        # Retrieval in codebase mode returns no results because codebase manifest has no active entries
        code_res = self.retriever.retrieve("handler", mode="codebase")
        self.assertEqual(len(code_res.chunks), 0)

        # Now clean codebase collection: code_id should be identified as orphan and deleted
        code_cleanup = self.indexer.clean_orphaned_generations(CODEBASE_COLLECTION_NAME)
        self.assertEqual(code_cleanup["details"][CODEBASE_COLLECTION_NAME]["deleted"], 1)

        code_col = self.vector_store.get_collection(CODEBASE_COLLECTION_NAME)
        self.assertEqual(code_col.count(), 0)

        doc_col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        self.assertEqual(doc_col.count(), 1)

    # ── Test 11: IncrementalIndexer.sync() incorporates orphan cleanup safely ──
    def test_11_sync_incorporates_orphan_cleanup_safely(self):
        """Verify that indexer.sync() includes orphan cleanup in its return summary."""
        doc_file = os.path.join(self.test_dir, "test_doc.txt")
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write("A test document for full sync verification.")

        # Seed an orphan directly into ChromaDB before sync
        orphan_id = "doc:ghost.txt:c0:g_ghost"
        self.vector_store.add_documents(
            DOCUMENT_COLLECTION_NAME,
            [orphan_id],
            ["Ghost content that should be purged during sync"],
            self.mock_embedder.embed_batch(["Ghost content"]),
            [{"source_path": "ghost.txt", "category": "document"}],
        )

        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        self.assertEqual(col.count(), 1)

        # Run indexer.sync()
        sync_result = self.indexer.sync()

        self.assertEqual(sync_result["status"], "completed")
        self.assertIn("orphan_cleanup", sync_result)
        self.assertEqual(sync_result["orphan_cleanup"]["orphaned_chunks_deleted"], 1)

        # Ghost chunk is gone
        remaining = col.get(ids=[orphan_id])
        self.assertEqual(remaining.get("ids", []), [])


if __name__ == "__main__":
    unittest.main()
