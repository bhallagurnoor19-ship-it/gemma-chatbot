"""
Automated Test Suite for Phase 2: Embedding Service & Persistent ChromaDB Storage
Validates all 11 requirements specified in Phase 2 architectural guidelines.
Uses isolated temporary directories to prevent vector database contamination.
"""

import os
import shutil
import tempfile
import unittest
from typing import List

from rag import (
    EmbeddingService,
    OllamaConnectionError,
    EmbeddingModelNotFoundError,
    EmbeddingDimensionError,
    VectorStoreManager,
    DOCUMENT_COLLECTION_NAME,
    CODEBASE_COLLECTION_NAME,
    sanitize_metadata,
)
from rag.config import (
    OLLAMA_BASE_URL,
    PRIMARY_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)


class TestPhase2RAG(unittest.TestCase):
    """Test suite for Phase 2 Shared Embedding Service & ChromaDB Storage."""

    @classmethod
    def setUpClass(cls):
        """Prepare shared embedding service and test directories."""
        cls.embedding_service = EmbeddingService(
            base_url=OLLAMA_BASE_URL,
            model_name=PRIMARY_EMBEDDING_MODEL,
            batch_size=4,
            timeout=60,
            expected_dimension=EMBEDDING_DIMENSION,
        )
        cls.test_dir = tempfile.mkdtemp(prefix="rag_test_chroma_")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary ChromaDB storage."""
        if os.path.exists(cls.test_dir):
            try:
                shutil.rmtree(cls.test_dir, ignore_errors=True)
            except Exception:
                pass

    # ── Test 1: Ollama Connectivity ──────────────────────────────────────────
    def test_01_ollama_connectivity(self):
        """Verify that Ollama server is running and reachable."""
        is_alive = self.embedding_service.check_ollama_alive()
        self.assertTrue(is_alive, f"Ollama should be reachable at {OLLAMA_BASE_URL}")

    # ── Test 2: Embedding Model Availability ─────────────────────────────────
    def test_02_embedding_model_availability(self):
        """Verify that target embedding model is installed in Ollama."""
        is_installed = self.embedding_service.is_model_installed()
        self.assertTrue(
            is_installed,
            f"Embedding model '{PRIMARY_EMBEDDING_MODEL}' must be installed in Ollama."
        )

    # ── Test 3: Single-Text Embedding ────────────────────────────────────────
    def test_03_single_text_embedding(self):
        """Generate embedding for a single text query and verify vector."""
        query = "Unified RAG architectural foundation for multi-LLM chatbot"
        vector = self.embedding_service.embed_query(query)

        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), EMBEDDING_DIMENSION)
        self.assertTrue(all(isinstance(x, float) for x in vector))

    # ── Test 4: Batch Embedding ──────────────────────────────────────────────
    def test_04_batch_embedding(self):
        """Generate embeddings for multiple chunks across batch boundaries."""
        chunks = [
            "Chunk A: Python architecture and modular decomposition.",
            "Chunk B: DeepSeek Coder V2 principal architect blueprint.",
            "Chunk C: ChromaDB persistent vector database with cosine metric.",
            "Chunk D: Incremental indexing using SHA-256 file manifests.",
            "Chunk E: Speech recognition pipeline with PyMuPDF document extraction.",
        ]
        vectors = self.embedding_service.embed_documents(chunks)

        self.assertEqual(len(vectors), len(chunks))
        for vec in vectors:
            self.assertEqual(len(vec), EMBEDDING_DIMENSION)
            self.assertTrue(all(isinstance(x, float) for x in vec))

    # ── Test 5: Embedding Dimension Consistency & Validation ─────────────────
    def test_05_dimension_consistency_and_validation(self):
        """Verify strict vector validation and dimension mismatch rejection."""
        # Valid vector validation
        mock_valid = [0.1] * EMBEDDING_DIMENSION
        result = self.embedding_service._validate_vector(mock_valid)
        self.assertEqual(len(result), EMBEDDING_DIMENSION)

        # Dimension mismatch rejection
        mock_short = [0.1] * (EMBEDDING_DIMENSION - 10)
        with self.assertRaises(EmbeddingDimensionError):
            self.embedding_service._validate_vector(mock_short)

        # Non-numeric rejection
        mock_invalid = ["not_a_float"] * EMBEDDING_DIMENSION
        with self.assertRaises(EmbeddingDimensionError):
            self.embedding_service._validate_vector(mock_invalid)

    # ── Test 6: ChromaDB Initialization ──────────────────────────────────────
    def test_06_chromadb_initialization(self):
        """Verify ChromaDB PersistentClient initializes cleanly at configured path."""
        manager = VectorStoreManager(persist_dir=self.test_dir)
        manager.initialize()

        self.assertTrue(manager.is_initialized())
        self.assertTrue(os.path.exists(self.test_dir))

    # ── Test 7: Separate Document and Codebase Collections ───────────────────
    def test_07_separate_document_and_codebase_collections(self):
        """Verify document and codebase collections are strictly partitioned."""
        manager = VectorStoreManager(persist_dir=self.test_dir)
        manager.initialize()

        doc_stats = manager.get_collection_stats(DOCUMENT_COLLECTION_NAME)
        code_stats = manager.get_collection_stats(CODEBASE_COLLECTION_NAME)

        self.assertEqual(doc_stats["name"], DOCUMENT_COLLECTION_NAME)
        self.assertEqual(code_stats["name"], CODEBASE_COLLECTION_NAME)

        # Insert item into documents collection only
        dummy_vec = [0.05] * EMBEDDING_DIMENSION
        manager.add_documents(
            collection_name=DOCUMENT_COLLECTION_NAME,
            ids=["doc_1"],
            documents=["Architecture whitepaper page 1"],
            embeddings=[dummy_vec],
            metadatas=[{"source": "whitepaper.pdf", "page": 1}]
        )

        # Confirm count isolated to documents collection
        doc_count = manager.get_collection_stats(DOCUMENT_COLLECTION_NAME)["count"]
        code_count = manager.get_collection_stats(CODEBASE_COLLECTION_NAME)["count"]

        self.assertEqual(doc_count, 1)
        self.assertEqual(code_count, 0)

    # ── Test 8: Vector Insertion and Retrieval ───────────────────────────────
    def test_08_vector_insertion_and_retrieval(self):
        """Insert vectors with sanitized metadata and retrieve via cosine similarity."""
        manager = VectorStoreManager(persist_dir=self.test_dir)
        manager.initialize()

        # Insert two code chunks
        vec_router = self.embedding_service.embed_query("classify_intent router logic")
        vec_voice = self.embedding_service.embed_query("speech recognition voice dictation")

        manager.add_documents(
            collection_name=CODEBASE_COLLECTION_NAME,
            ids=["code_router", "code_voice"],
            documents=[
                "def classify_intent(messages): return 'pipeline'",
                "def transcribe_audio(): return 'recognized text'",
            ],
            embeddings=[vec_router, vec_voice],
            metadatas=[
                {"file": "app.py", "function": "classify_intent", "tags": ["router", "ml"]},
                {"file": "app.py", "function": "transcribe_audio", "tags": ["voice"]},
            ]
        )

        # Query for router functionality
        query_vec = self.embedding_service.embed_query("intent routing classification")
        results = manager.query(
            collection_name=CODEBASE_COLLECTION_NAME,
            query_embeddings=query_vec,
            n_results=1
        )

        matched_ids = results.get("ids", [[]])[0]
        self.assertIn("code_router", matched_ids)

        # Verify metadata sanitization preserved tags as string
        matched_meta = results.get("metadatas", [[]])[0][0]
        self.assertEqual(matched_meta.get("function"), "classify_intent")
        self.assertIn("router", matched_meta.get("tags"))

    # ── Test 9: Deletion by Chunk IDs ────────────────────────────────────────
    def test_09_deletion_by_chunk_ids(self):
        """Verify vector deletion by chunk ID."""
        manager = VectorStoreManager(persist_dir=self.test_dir)
        manager.initialize()

        # Delete code_voice from codebase_collection
        success = manager.delete_by_ids(CODEBASE_COLLECTION_NAME, ["code_voice"])
        self.assertTrue(success)

        # Query should no longer return code_voice
        results = manager.query(
            collection_name=CODEBASE_COLLECTION_NAME,
            query_embeddings=[0.1] * EMBEDDING_DIMENSION,
            n_results=5
        )
        returned_ids = results.get("ids", [[]])[0]
        self.assertNotIn("code_voice", returned_ids)

    # ── Test 10: Persistence Across Client Reopening ─────────────────────────
    def test_10_persistence_across_reopen(self):
        """Verify persistent database preserves data after closing and reopening client."""
        # Step 1: Open manager, ensure item exists, then dereference
        manager1 = VectorStoreManager(persist_dir=self.test_dir)
        manager1.initialize()
        initial_doc_count = manager1.get_collection_stats(DOCUMENT_COLLECTION_NAME)["count"]
        del manager1

        # Step 2: Open completely new client on same directory
        manager2 = VectorStoreManager(persist_dir=self.test_dir)
        manager2.initialize()
        reopened_doc_count = manager2.get_collection_stats(DOCUMENT_COLLECTION_NAME)["count"]

        self.assertEqual(initial_doc_count, reopened_doc_count)
        self.assertGreater(reopened_doc_count, 0)

    # ── Test 11: Graceful Handling of Ollama Being Unavailable ───────────────
    def test_11_graceful_handling_of_ollama_unavailable(self):
        """Verify graceful error reporting when Ollama server or model is missing."""
        # Unreachable port
        dead_service = EmbeddingService(
            base_url="http://localhost:59999",
            model_name=PRIMARY_EMBEDDING_MODEL,
            timeout=2
        )
        with self.assertRaises(OllamaConnectionError) as ctx:
            dead_service.embed_query("Test unreachable Ollama")
        self.assertIn("Cannot reach Ollama server", str(ctx.exception))

        # Missing model on valid server
        missing_model_service = EmbeddingService(
            base_url=OLLAMA_BASE_URL,
            model_name="non_existent_model_xyz",
            timeout=5
        )
        with self.assertRaises(EmbeddingModelNotFoundError) as ctx2:
            missing_model_service.embed_query("Test missing model")
        self.assertIn("ollama pull non_existent_model_xyz", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
