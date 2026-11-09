"""
Automated Failure-Injection & Reliability Hardening Test Suite (Phase 4.1)
Validates:
1. Embedding generation failure (rollback & generation preservation)
2. ChromaDB insertion failure (rollback & generation preservation)
3. Partial insertion failure (rollback & generation preservation)
4. Verification failure (rollback & generation preservation)
5. Manifest commit failure (rollback & generation preservation)
6. Cleanup failure after successful commit (graceful warning, commit preserved)
7. Recovery after simulated interruption
8. Structured retrieval status handling (success, no_results, embedding_unavailable, database_error)
9. Context budget truncation safety (never malforms citations or leaves open fences)
"""

from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.config import (
    CODEBASE_COLLECTION_NAME,
    DOCUMENT_COLLECTION_NAME,
)
from rag.retriever import RetrievedChunk, RetrievalResponse, RetrievalStatus, UnifiedRetriever
from rag.prompt_builder import PromptBuilder
from rag.indexer import IncrementalIndexer
from rag.vector_store import VectorStoreManager


class ControllableMockEmbedder:
    """Mock embedder with controllable failure triggers."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.fail_batch = False
        self.fail_single = False
        self.call_count = 0

    def embed_text(self, text: str) -> List[float]:
        self.call_count += 1
        if self.fail_single:
            raise ConnectionError("Ollama embedding service connection timeout (mocked)")
        return self._make_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        if self.fail_batch:
            raise ConnectionError("Ollama batch embedding failure (mocked)")
        return [self._make_vector(t) for t in texts]

    def _make_vector(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dimension):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 255.0) * 2.0 - 1.0)
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [round(x / norm, 6) for x in vec]


class TestRAGPhase41Reliability(unittest.TestCase):
    """Failure-injection and reliability hardening test suite."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="rag_p41_test_")
        self.docs_dir = os.path.join(self.test_dir, "documents")
        self.code_dir = os.path.join(self.test_dir, "codebase")
        self.chroma_dir = os.path.join(self.test_dir, "chroma_db")
        self.manifest_path = os.path.join(self.test_dir, "manifests", "index_manifest.json")

        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)

        self.mock_embedder = ControllableMockEmbedder()
        self.vector_store = VectorStoreManager(persist_dir=self.chroma_dir)
        self.vector_store.initialize()

        self.indexer = IncrementalIndexer(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            base_dir=self.test_dir,
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass

    def _create_initial_document(self, name: str = "spec.txt", content: str = "Version 1.0 specification.") -> Tuple[str, str, List[str]]:
        """Helper to create and index an initial valid document."""
        doc_path = os.path.join(self.docs_dir, name)
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["indexed"], 1)

        rel_path = self.indexer._get_relative_path(doc_path)
        manifest = self.indexer.load_manifest()
        chunk_ids = manifest["documents"][rel_path]["chunk_ids"]
        return doc_path, rel_path, chunk_ids

    # ── Test 1: Embedding generation failure ──────────────────────────────────
    def test_01_embedding_generation_failure(self):
        doc_path, rel_path, v1_ids = self._create_initial_document()

        # Modify document
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 2.0 modified specification.")

        # Inject embedding batch failure
        self.mock_embedder.fail_batch = True

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["errors"], 1)
        self.assertEqual(res["updated"], 0)

        # Assert v1 generation chunks remain active in ChromaDB
        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        v1_data = col.get(ids=v1_ids)
        self.assertEqual(len(v1_data["ids"]), len(v1_ids))
        self.assertIn("Version 1.0", v1_data["documents"][0])

        # Assert manifest still points to v1
        manifest = self.indexer.load_manifest()
        self.assertEqual(manifest["documents"][rel_path]["chunk_ids"], v1_ids)

    # ── Test 2: ChromaDB insertion failure ────────────────────────────────────
    def test_02_chromadb_insertion_failure(self):
        doc_path, rel_path, v1_ids = self._create_initial_document()

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 2.0 with db insert failure test.")

        # Mock add_documents to fail
        orig_add = self.vector_store.add_documents
        def mock_failing_add(*args, **kwargs):
            raise RuntimeError("Simulated ChromaDB write failure during insert")

        self.vector_store.add_documents = mock_failing_add

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["errors"], 1)
        self.assertEqual(res["updated"], 0)

        # Restore
        self.vector_store.add_documents = orig_add

        # Assert v1 generation intact
        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        v1_data = col.get(ids=v1_ids)
        self.assertEqual(len(v1_data["ids"]), len(v1_ids))
        self.assertIn("Version 1.0", v1_data["documents"][0])

        # Assert manifest unchanged
        manifest = self.indexer.load_manifest()
        self.assertEqual(manifest["documents"][rel_path]["chunk_ids"], v1_ids)

    # ── Test 3: Partial insertion failure ────────────────────────────────────
    def test_03_partial_insertion_failure(self):
        doc_path, rel_path, v1_ids = self._create_initial_document()

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 2.0 with partial insertion mismatch.")

        # Mock add_documents returning count mismatch
        orig_add = self.vector_store.add_documents
        def mock_partial_add(*args, **kwargs):
            # Calls original add but returns mismatch count
            real_count = orig_add(*args, **kwargs)
            return real_count - 1  # simulate partial write

        self.vector_store.add_documents = mock_partial_add

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["errors"], 1)

        # Restore
        self.vector_store.add_documents = orig_add

        # Assert v1 generation chunks intact
        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        v1_data = col.get(ids=v1_ids)
        self.assertEqual(len(v1_data["ids"]), len(v1_ids))

        # Assert manifest unchanged
        manifest = self.indexer.load_manifest()
        self.assertEqual(manifest["documents"][rel_path]["chunk_ids"], v1_ids)

    # ── Test 4: Verification failure ─────────────────────────────────────────
    def test_04_verification_failure(self):
        doc_path, rel_path, v1_ids = self._create_initial_document()

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 2.0 with verification failure.")

        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        orig_get = col.get

        # Mock verification query returning empty list
        def mock_failing_get(*args, **kwargs):
            ids_requested = kwargs.get("ids", [])
            # If checking for newly inserted generation chunk, return empty
            if ids_requested and any(":g" in i for i in ids_requested):
                return {"ids": []}
            return orig_get(*args, **kwargs)

        col.get = mock_failing_get

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["errors"], 1)

        col.get = orig_get

        # Assert v1 generation intact
        v1_data = col.get(ids=v1_ids)
        self.assertEqual(len(v1_data["ids"]), len(v1_ids))

        manifest = self.indexer.load_manifest()
        self.assertEqual(manifest["documents"][rel_path]["chunk_ids"], v1_ids)

    # ── Test 5: Manifest commit failure ──────────────────────────────────────
    def test_05_manifest_commit_failure(self):
        doc_path, rel_path, v1_ids = self._create_initial_document()

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 2.0 with manifest save error.")

        # Mock save_manifest to raise IOError
        orig_save = self.indexer.save_manifest
        def mock_failing_save():
            raise IOError("Simulated disk full or permission error on manifest write")

        self.indexer.save_manifest = mock_failing_save

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["errors"], 1)

        self.indexer.save_manifest = orig_save

        # Assert newly inserted v2 chunks were rolled back from ChromaDB
        manifest_on_disk = self.indexer.load_manifest()
        self.assertEqual(manifest_on_disk["documents"][rel_path]["chunk_ids"], v1_ids)

        # Assert v1 chunks are still in ChromaDB
        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        v1_data = col.get(ids=v1_ids)
        self.assertEqual(len(v1_data["ids"]), len(v1_ids))

    # ── Test 6: Cleanup failure after successful commit ──────────────────────
    def test_06_cleanup_failure_after_successful_commit(self):
        doc_path, rel_path, v1_ids = self._create_initial_document()

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 2.0 with cleanup error on old chunks.")

        orig_delete = self.vector_store.delete_by_ids
        delete_called = False

        def mock_failing_delete(col_name, ids):
            nonlocal delete_called
            delete_called = True
            raise RuntimeError("Simulated cleanup deletion failure")

        self.vector_store.delete_by_ids = mock_failing_delete

        # Should log warning, but commit itself succeeded!
        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["updated"], 1)
        self.assertTrue(delete_called)

        self.vector_store.delete_by_ids = orig_delete

        # Manifest is committed to the new generation
        manifest = self.indexer.load_manifest()
        new_ids = manifest["documents"][rel_path]["chunk_ids"]
        self.assertNotEqual(new_ids, v1_ids)

        # New chunks verified in ChromaDB
        col = self.vector_store.get_collection(DOCUMENT_COLLECTION_NAME)
        new_data = col.get(ids=new_ids)
        self.assertEqual(len(new_data["ids"]), len(new_ids))

    # ── Test 7: Recovery after simulated interruption ─────────────────────────
    def test_07_recovery_after_simulated_interruption(self):
        doc_path, rel_path, v1_ids = self._create_initial_document()

        # Modify file on disk
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Version 2.0 text during interrupted run.")

        # Simulate interruption by raising KeyboardInterrupt inside indexer
        orig_safe = self.indexer._safe_replace_file_entry
        def mock_interrupted(*args, **kwargs):
            raise KeyboardInterrupt("Simulated SIGINT / Process kill")

        self.indexer._safe_replace_file_entry = mock_interrupted

        with self.assertRaises(KeyboardInterrupt):
            self.indexer.index_documents(self.docs_dir)

        self.indexer._safe_replace_file_entry = orig_safe

        # Verify manifest on disk still has v1
        fresh_indexer = IncrementalIndexer(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            base_dir=self.test_dir,
        )
        fresh_manifest = fresh_indexer.load_manifest()
        self.assertEqual(fresh_manifest["documents"][rel_path]["chunk_ids"], v1_ids)

        # Run fresh indexer: cleanly completes v2 indexing
        recovery_res = fresh_indexer.index_documents(self.docs_dir)
        self.assertEqual(recovery_res["updated"], 1)

        # Verified new generation committed
        final_manifest = fresh_indexer.load_manifest()
        self.assertNotEqual(final_manifest["documents"][rel_path]["chunk_ids"], v1_ids)

    # ── Test 8: Structured retrieval status handling ─────────────────────────
    def test_08_retrieval_status_handling(self):
        # 1. Success status
        doc_path, _, _ = self._create_initial_document("retrieval_target.txt", "Deploy server configuration guide.")
        retriever = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            distance_threshold=2.0,
        )

        res_success = retriever.retrieve("Deploy server", mode="documents")
        self.assertEqual(res_success.status, RetrievalStatus.SUCCESS)
        self.assertGreater(len(res_success), 0)
        self.assertIsNone(res_success.error_message)

        # Container emulation check
        self.assertTrue(bool(res_success))
        self.assertIn("Deploy", res_success[0].content)

        # 2. No results status
        res_no_results = retriever.retrieve("completely unrelated query", mode="documents", distance_threshold=0.0001)
        self.assertEqual(res_no_results.status, RetrievalStatus.NO_RESULTS)
        self.assertEqual(len(res_no_results), 0)
        self.assertFalse(bool(res_no_results))

        # 3. Embedding unavailable status
        failing_embedder = ControllableMockEmbedder()
        failing_embedder.fail_single = True
        retriever_fail_emb = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=failing_embedder,
            manifest_path=self.manifest_path,
        )

        res_emb_fail = retriever_fail_emb.retrieve("any query")
        self.assertEqual(res_emb_fail.status, RetrievalStatus.EMBEDDING_UNAVAILABLE)
        self.assertIn("Embedding service is currently unavailable", res_emb_fail.error_message)
        self.assertNotIn("Traceback", res_emb_fail.error_message)  # No internal leak!

        # 4. Database error status
        broken_store = VectorStoreManager(persist_dir=os.path.join(self.test_dir, "nonexistent_chroma"))
        broken_retriever = UnifiedRetriever(
            vector_store=broken_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
        )
        broken_store.query = MagicMock(side_effect=RuntimeError("Chroma sqlite lock error"))

        res_db_err = broken_retriever.retrieve("query")
        self.assertEqual(res_db_err.status, RetrievalStatus.DATABASE_ERROR)
        self.assertIn("Vector storage error", res_db_err.error_message)
        self.assertNotIn("Traceback", res_db_err.error_message)

    # ── Test 9: Context budget truncation safety ─────────────────────────────
    def test_09_prompt_budget_truncation_never_malforms_citation(self):
        builder = PromptBuilder(max_tokens=180, chars_per_token=4.0)  # ~720 chars budget: header (450 chars) + chunk (270 chars)

        chunks = [
            RetrievedChunk(
                content="Line 1: Extensive description of the architecture.\nLine 2: " + ("word " * 50),
                source_path="deep/nested/path/to/documentation_guide.pdf",
                file_name="documentation_guide.pdf",
                category="document",
                distance=0.1,
                similarity=0.9,
                page_number=12,
            ),
            RetrievedChunk(
                content="def compute():\n    " + ("x = 1\n    " * 30),
                source_path="deep/nested/code/algorithms.py",
                file_name="algorithms.py",
                category="codebase",
                distance=0.2,
                similarity=0.8,
                symbol_name="Algorithm.compute",
                start_line=100,
                end_line=130,
            ),
        ]

        context = builder.build_context_block(chunks)

        # Verification: Citations must NEVER be broken or malformed
        self.assertIn("[Document: documentation_guide.pdf, Page 12]", context)

        # Ensure closing brackets exist and citation is complete
        doc_tag = "[Document: documentation_guide.pdf, Page 12]"
        self.assertTrue(doc_tag in context)

        # If codebase citation was included, it must be intact
        if "algorithms.py" in context:
            self.assertIn("[Code: algorithms.py, Lines 100-130, Symbol: Algorithm.compute]", context)
            # Ensure fences are cleanly closed
            self.assertEqual(context.count("```") % 2, 0)


if __name__ == "__main__":
    unittest.main()
