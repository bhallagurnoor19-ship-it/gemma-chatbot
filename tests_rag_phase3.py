"""
Automated Test Suite for Unified RAG Phase 3
Validates:
1. PDF extraction and page metadata
2. TXT and Markdown extraction
3. Python AST extraction (class, method, function, imports, line numbers)
4. Source code and line preservation
5. Chunking and metadata correctness
6. New-file indexing
7. Modified-file re-indexing (only changed file re-indexed)
8. Unchanged-file skipping (0 re-embeddings)
9. Deleted-file cleanup (chunk IDs purged from ChromaDB)
10. Renamed-file handling
11. Interrupted-indexing recovery (manifest integrity)
12. Ignored files and secret exclusion (.env, server.log, .git)
13. Symlink / path traversal protection (is_safe_path)
14. Independent document and codebase collections
15. Graceful handling of corrupt and unsupported files
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pymupdf

from rag.config import (
    CODEBASE_COLLECTION_NAME,
    DOCUMENT_COLLECTION_NAME,
    is_path_ignored,
    is_safe_path,
)
from rag.document_pipeline import DocumentPipeline
from rag.codebase_pipeline import CodebasePipeline
from rag.indexer import IncrementalIndexer
from rag.vector_store import VectorStoreManager


class MockEmbeddingService:
    """Fast, deterministic in-memory mock embedding service for unit testing."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.call_count = 0
        self.embedded_texts_count = 0

    def embed_text(self, text: str) -> List[float]:
        self.call_count += 1
        self.embedded_texts_count += 1
        return self._make_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self.call_count += 1
        self.embedded_texts_count += len(texts)
        return [self._make_vector(t) for t in texts]

    def _make_vector(self, text: str) -> List[float]:
        # Deterministic 768-dim pseudo-vector from text hash
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dimension):
            byte_val = h[i % len(h)]
            vec.append(round((byte_val / 255.0) * 2.0 - 1.0, 6))
        # Normalize roughly
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [round(x / norm, 6) for x in vec]


class TestRAGPhase3(unittest.TestCase):
    """Comprehensive test suite for Phase 3 ingestion and incremental indexer."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="rag_p3_test_")
        self.docs_dir = os.path.join(self.test_dir, "documents")
        self.code_dir = os.path.join(self.test_dir, "codebase")
        self.chroma_dir = os.path.join(self.test_dir, "chroma_db")
        self.manifest_path = os.path.join(self.test_dir, "manifests", "index_manifest.json")

        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)

        self.mock_embedder = MockEmbeddingService()
        self.vector_store = VectorStoreManager(persist_dir=self.chroma_dir)
        self.vector_store.initialize()

        self.doc_pipeline = DocumentPipeline(base_dir=self.test_dir)
        self.code_pipeline = CodebasePipeline(base_dir=self.test_dir)

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

    # ── Test 1: PDF extraction and page metadata ──────────────────────────────
    def test_01_pdf_extraction_and_page_metadata(self):
        pdf_path = os.path.join(self.docs_dir, "test_doc.pdf")
        doc = pymupdf.open()
        p1 = doc.new_page()
        p1.insert_text((50, 50), "Architecture Overview: This is page one describing system design.")
        p2 = doc.new_page()
        p2.insert_text((50, 50), "Database Schema: This is page two describing persistent storage tables.")
        doc.save(pdf_path)
        doc.close()

        chunks = self.doc_pipeline.process_file(pdf_path)
        self.assertGreaterEqual(len(chunks), 2)

        page_numbers = [c["metadata"]["page_number"] for c in chunks]
        self.assertIn(1, page_numbers)
        self.assertIn(2, page_numbers)

        p1_chunks = [c for c in chunks if c["metadata"]["page_number"] == 1]
        p2_chunks = [c for c in chunks if c["metadata"]["page_number"] == 2]

        self.assertTrue(all("Architecture Overview" in c["text"] for c in p1_chunks))
        self.assertTrue(all("Database Schema" in c["text"] for c in p2_chunks))
        self.assertEqual(p1_chunks[0]["metadata"]["total_pages"], 2)
        self.assertEqual(p1_chunks[0]["metadata"]["file_type"], "pdf")

    # ── Test 2: TXT and Markdown extraction ──────────────────────────────────
    def test_02_txt_and_markdown_extraction(self):
        txt_path = os.path.join(self.docs_dir, "guide.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Line 1: System installation guide.\n\nLine 2: Prerequisites include Python 3.13.\n")

        md_path = os.path.join(self.docs_dir, "readme.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Project Gemma\n\n## Section 1\nThis is a markdown technical specification.\n")

        txt_chunks = self.doc_pipeline.process_file(txt_path)
        md_chunks = self.doc_pipeline.process_file(md_path)

        self.assertGreaterEqual(len(txt_chunks), 1)
        self.assertGreaterEqual(len(md_chunks), 1)

        self.assertEqual(txt_chunks[0]["metadata"]["file_type"], "txt")
        self.assertEqual(md_chunks[0]["metadata"]["file_type"], "md")
        self.assertEqual(txt_chunks[0]["metadata"]["page_number"], 1)

    # ── Test 3: Python AST extraction (class, method, function, imports, line numbers) ──
    def test_03_python_ast_extraction(self):
        py_code = (
            '"""Module docstring."""\n'
            "import os\n"
            "import sys\n\n"
            "class MathService:\n"
            '    """Service for math operations."""\n'
            "    def add(self, a, b):\n"
            "        return a + b\n\n"
            "    def multiply(self, a, b):\n"
            "        return a * b\n\n"
            "def global_helper(val):\n"
            "    return val * 2\n"
        )
        py_path = os.path.join(self.code_dir, "math_module.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(py_code)

        chunks = self.code_pipeline.process_file(py_path)
        self.assertGreaterEqual(len(chunks), 4)

        symbols = {c["metadata"]["symbol_name"]: c["metadata"]["symbol_type"] for c in chunks}
        self.assertIn("module", symbols)
        self.assertEqual(symbols["module"], "module_header")
        self.assertIn("MathService", symbols)
        self.assertEqual(symbols["MathService"], "class")
        self.assertIn("MathService.add", symbols)
        self.assertEqual(symbols["MathService.add"], "method")
        self.assertIn("MathService.multiply", symbols)
        self.assertEqual(symbols["MathService.multiply"], "method")
        self.assertIn("global_helper", symbols)
        self.assertEqual(symbols["global_helper"], "function")

    # ── Test 4: Source code and line preservation ────────────────────────────
    def test_04_source_code_and_line_preservation(self):
        py_code = (
            "import math\n\n"
            "def calculate_area(radius):\n"
            "    # Pi times r squared\n"
            "    return math.pi * radius * radius\n"
        )
        py_path = os.path.join(self.code_dir, "calc.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(py_code)

        chunks = self.code_pipeline.process_file(py_path)
        func_chunk = next(c for c in chunks if c["metadata"]["symbol_name"] == "calculate_area")

        start = func_chunk["metadata"]["start_line"]
        end = func_chunk["metadata"]["end_line"]
        self.assertEqual(start, 3)
        self.assertEqual(end, 5)

        lines = py_code.splitlines()
        expected_text = "\n".join(lines[start - 1 : end]).strip()
        self.assertEqual(func_chunk["text"], expected_text)

    # ── Test 5: Chunking and metadata correctness ────────────────────────────
    def test_05_chunking_and_metadata_correctness(self):
        json_path = os.path.join(self.code_dir, "config.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write('{\n  "app_name": "gemma_bot",\n  "version": "1.0",\n  "port": 5000\n}\n')

        chunks = self.code_pipeline.process_file(json_path)
        self.assertEqual(len(chunks), 1)

        meta = chunks[0]["metadata"]
        self.assertEqual(meta["language"], "json")
        self.assertEqual(meta["symbol_type"], "config")
        self.assertEqual(meta["category"], "codebase")

        # Verify all metadata values conform to ChromaDB scalar types
        for k, v in meta.items():
            self.assertIsInstance(v, (str, int, float, bool))

    # ── Test 6: New-file indexing ────────────────────────────────────────────
    def test_06_new_file_indexing(self):
        f1 = os.path.join(self.docs_dir, "doc1.txt")
        with open(f1, "w", encoding="utf-8") as f:
            f.write("First document content for retrieval.")

        f2 = os.path.join(self.docs_dir, "doc2.txt")
        with open(f2, "w", encoding="utf-8") as f:
            f.write("Second document content with instructions.")

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["indexed"], 2)
        self.assertEqual(res["skipped"], 0)
        self.assertEqual(res["errors"], 0)

        # Check ChromaDB collection count
        stats = self.vector_store.get_collection_stats(DOCUMENT_COLLECTION_NAME)
        self.assertEqual(stats["count"], res["total_chunks"])

        # Check manifest
        self.assertTrue(os.path.exists(self.manifest_path))
        manifest = self.indexer.load_manifest()
        self.assertEqual(len(manifest["documents"]), 2)

    # ── Test 7: Modified-file re-indexing (only changed file re-indexed) ──────
    def test_07_modified_file_reindexing(self):
        f1 = os.path.join(self.docs_dir, "doc1.txt")
        with open(f1, "w", encoding="utf-8") as f:
            f.write("Initial content 1.")

        f2 = os.path.join(self.docs_dir, "doc2.txt")
        with open(f2, "w", encoding="utf-8") as f:
            f.write("Initial content 2.")

        self.indexer.index_documents(self.docs_dir)
        initial_calls = self.mock_embedder.call_count

        # Modify only doc1.txt
        with open(f1, "w", encoding="utf-8") as f:
            f.write("Updated content 1 with new information.")

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["indexed"], 0)
        self.assertEqual(res["updated"], 1)
        self.assertEqual(res["skipped"], 1)

        # Verify only 1 batch embedding occurred during re-index
        self.assertEqual(self.mock_embedder.call_count, initial_calls + 1)

    # ── Test 8: Unchanged-file skipping (0 re-embeddings) ────────────────────
    def test_08_unchanged_file_skipping(self):
        f1 = os.path.join(self.docs_dir, "stable.txt")
        with open(f1, "w", encoding="utf-8") as f:
            f.write("Stable content that never changes.")

        self.indexer.index_documents(self.docs_dir)
        call_count_before = self.mock_embedder.call_count

        # Second indexing run: nothing modified
        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["indexed"], 0)
        self.assertEqual(res["updated"], 0)
        self.assertEqual(res["skipped"], 1)

        # Absolutely zero new embedding calls!
        self.assertEqual(self.mock_embedder.call_count, call_count_before)

    # ── Test 9: Deleted-file cleanup (chunk IDs purged from ChromaDB) ─────────
    def test_09_deleted_file_cleanup(self):
        f1 = os.path.join(self.docs_dir, "keep.txt")
        with open(f1, "w", encoding="utf-8") as f:
            f.write("Document to keep.")

        f2 = os.path.join(self.docs_dir, "remove.txt")
        with open(f2, "w", encoding="utf-8") as f:
            f.write("Document to delete.")

        self.indexer.index_documents(self.docs_dir)
        initial_count = self.vector_store.get_collection_stats(DOCUMENT_COLLECTION_NAME)["count"]

        # Delete remove.txt
        os.remove(f2)

        res = self.indexer.index_documents(self.docs_dir)
        self.assertEqual(res["deleted"], 1)

        # Check ChromaDB decreased
        new_count = self.vector_store.get_collection_stats(DOCUMENT_COLLECTION_NAME)["count"]
        self.assertLess(new_count, initial_count)

        # Check manifest
        manifest = self.indexer.load_manifest()
        self.assertNotIn(self.indexer._get_relative_path(f2), manifest["documents"])

    # ── Test 10: Renamed-file handling ───────────────────────────────────────
    def test_10_renamed_file_handling(self):
        old_path = os.path.join(self.docs_dir, "original_name.txt")
        with open(old_path, "w", encoding="utf-8") as f:
            f.write("Important text to preserve across renames.")

        self.indexer.index_documents(self.docs_dir)
        old_rel = self.indexer._get_relative_path(old_path)
        self.assertIn(old_rel, self.indexer.manifest["documents"])

        # Rename file on disk
        new_path = os.path.join(self.docs_dir, "new_name.txt")
        os.rename(old_path, new_path)

        res = self.indexer.index_documents(self.docs_dir)
        new_rel = self.indexer._get_relative_path(new_path)

        # Old path pruned, new path indexed
        manifest = self.indexer.load_manifest()
        self.assertNotIn(old_rel, manifest["documents"])
        self.assertIn(new_rel, manifest["documents"])

    # ── Test 11: Interrupted-indexing recovery (manifest integrity) ───────────
    def test_11_interrupted_indexing_recovery(self):
        # 1. Test atomic manifest write (no left-over temp files)
        self.indexer.save_manifest()
        tmp_file = f"{self.manifest_path}.tmp"
        self.assertFalse(os.path.exists(tmp_file))

        # 2. Corrupt manifest intentionally to simulate crash
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json corrupted halfway ...")

        # New indexer should recover cleanly by creating backup and clean manifest
        fresh_indexer = IncrementalIndexer(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            base_dir=self.test_dir,
        )
        self.assertEqual(fresh_indexer.manifest["documents"], {})
        self.assertTrue(os.path.exists(f"{self.manifest_path}.bak"))

    # ── Test 12: Ignored files and secret exclusion (.env, server.log, .git) ──
    def test_12_ignored_files_and_secret_exclusion(self):
        env_file = os.path.join(self.test_dir, ".env")
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("SECRET_KEY=123456\n")

        log_file = os.path.join(self.test_dir, "server.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("Sensitive system logs.\n")

        self.assertTrue(is_path_ignored(env_file, self.test_dir))
        self.assertTrue(is_path_ignored(log_file, self.test_dir))

        self.assertFalse(self.doc_pipeline.can_process(env_file))
        self.assertFalse(self.code_pipeline.can_process(env_file))

    # ── Test 13: Symlink / path traversal protection (is_safe_path) ──────────
    def test_13_symlink_path_traversal_protection(self):
        inside_path = os.path.join(self.test_dir, "documents", "valid.txt")
        traversal_path = os.path.join(self.test_dir, "..", "..", "sensitive.txt")

        self.assertTrue(is_safe_path(inside_path, self.test_dir))
        self.assertFalse(is_safe_path(traversal_path, self.test_dir))

        # can_process should reject traversal paths
        self.assertFalse(self.doc_pipeline.can_process(traversal_path))
        self.assertFalse(self.code_pipeline.can_process(traversal_path))

    # ── Test 14: Independent document and codebase collections ───────────────
    def test_14_independent_document_and_codebase_collections(self):
        doc_file = os.path.join(self.docs_dir, "policy.txt")
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write("Company vacation policy document.")

        code_file = os.path.join(self.code_dir, "service.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write("def calculate_vacation_days(years):\n    return years * 5\n")

        self.indexer.index_documents(self.docs_dir)
        self.indexer.index_codebase(self.code_dir)

        doc_stats = self.vector_store.get_collection_stats(DOCUMENT_COLLECTION_NAME)
        code_stats = self.vector_store.get_collection_stats(CODEBASE_COLLECTION_NAME)

        self.assertGreater(doc_stats["count"], 0)
        self.assertGreater(code_stats["count"], 0)

        # Verify separation: querying documents doesn't return python code
        doc_res = self.vector_store.query(
            DOCUMENT_COLLECTION_NAME,
            query_embeddings=[self.mock_embedder.embed_text("vacation")],
            n_results=5,
        )
        for meta in doc_res["metadatas"][0]:
            self.assertEqual(meta["category"], "document")

        code_res = self.vector_store.query(
            CODEBASE_COLLECTION_NAME,
            query_embeddings=[self.mock_embedder.embed_text("vacation")],
            n_results=5,
        )
        for meta in code_res["metadatas"][0]:
            self.assertEqual(meta["category"], "codebase")

    # ── Test 15: Graceful handling of corrupt and unsupported files ──────────
    def test_15_graceful_handling_of_corrupt_and_unsupported_files(self):
        # 1. Corrupt fake PDF with binary junk
        corrupt_pdf = os.path.join(self.docs_dir, "corrupt.pdf")
        with open(corrupt_pdf, "wb") as f:
            f.write(b"NOT_A_REAL_PDF_HEADER_1234567890\x00\xff\xfe")

        chunks = self.doc_pipeline.process_file(corrupt_pdf)
        self.assertEqual(chunks, [])

        # 2. Python file with SyntaxError
        broken_py = os.path.join(self.code_dir, "broken.py")
        with open(broken_py, "w", encoding="utf-8") as f:
            f.write("def syntax_error_func( :\n    print('broken')\n")

        # Should fall back to line-based chunking rather than throwing unhandled exception
        broken_chunks = self.code_pipeline.process_file(broken_py)
        self.assertGreaterEqual(len(broken_chunks), 1)
        self.assertEqual(broken_chunks[0]["metadata"]["symbol_type"], "code_block")


if __name__ == "__main__":
    unittest.main()
