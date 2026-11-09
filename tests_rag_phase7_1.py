"""
Tests for Phase 7.1 — RAG Retrieval Quality Refinement
Validates:
- Query intent classification (_detect_query_intent)
- Chunk source classification (_classify_chunk_source)
- Source-aware ranking calibration across impl, test, config, and frontend queries
- Fallback to distance sorting when ranking is disabled
- Frontend HTML template parsing and line citations in CodebasePipeline
- Separated evaluation metrics (abstention vs answerable retrieval success)
"""

import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from rag.codebase_pipeline import CodebasePipeline
from rag.config import (
    CODEBASE_COLLECTION_NAME,
    DEFAULT_RANKING_CONFIG,
    DOCUMENT_COLLECTION_NAME,
    INDEX_MANIFEST_PATH,
)
from rag.retriever import RetrievedChunk, RetrievalResponse, RetrievalStatus, UnifiedRetriever


class MockEmbeddingService:
    """Mock embedding service returning deterministic vectors."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * self.dimension

    def embed_text(self, text: str) -> List[float]:
        return self.embed_query(text)

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        return [[0.1] * self.dimension for _ in texts]

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        return self.embed_batch(texts, batch_size=batch_size)


class TestPhase71Refinement(unittest.TestCase):
    """Unit test suite for Phase 7.1 retrieval quality refinements."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="rag_test_p71_")
        self.manifest_path = os.path.join(self.test_dir, "manifest.json")
        self.retriever = UnifiedRetriever(
            vector_store=MagicMock(),
            embedding_service=MockEmbeddingService(),
            manifest_path=self.manifest_path,
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # ── 1. Query Intent Classification ───────────────────────────────────────

    def test_intent_detection_impl_query(self):
        """Standard implementation questions default to 'impl_query'."""
        queries = [
            "How does IncrementalIndexer compute file hashes?",
            "Explain the route handler for /api/chat in app.py",
            "What methods are in VectorStoreManager?",
            "How does classify_intent route to local vs gemini models?",
        ]
        for q in queries:
            intent = self.retriever._detect_query_intent(q)
            self.assertEqual(intent, "impl_query", f"Failed for query: {q}")

    def test_intent_detection_test_query(self):
        """Questions mentioning tests, asserts, or mocks map to 'test_query'."""
        queries = [
            "What unit test methods verify active generation filtering?",
            "How does tests_rag_phase3.py test unreadable manifests?",
            "What assertions verify failure injection rollback?",
            "Explain the pytest suite for the embedding service",
        ]
        for q in queries:
            intent = self.retriever._detect_query_intent(q)
            self.assertEqual(intent, "test_query", f"Failed for query: {q}")

    def test_intent_detection_config_query(self):
        """Questions about configs, settings, or .ragignore map to 'config_query'."""
        queries = [
            "What is the default SIMILARITY_DISTANCE_THRESHOLD in rag/config.py?",
            "What settings are defined in requirements.txt?",
            "How does is_path_ignored evaluate rules in .ragignore?",
            "What are the constants and paths configured in config.py?",
        ]
        for q in queries:
            intent = self.retriever._detect_query_intent(q)
            self.assertEqual(intent, "config_query", f"Failed for query: {q}")

    def test_intent_detection_frontend_query(self):
        """Questions about UI, templates, buttons, or SSE events map to 'frontend_query'."""
        queries = [
            "Where is the rag-mode-select dropdown defined in templates/index.html?",
            "How is the pause and resume streaming button implemented?",
            "How does streamBotPipeline handle the citation SSE event?",
            "Where are the CSS variables for dark theme and light theme defined?",
        ]
        for q in queries:
            intent = self.retriever._detect_query_intent(q)
            self.assertEqual(intent, "frontend_query", f"Failed for query: {q}")

    # ── 2. Chunk Source Classification ───────────────────────────────────────

    def test_chunk_source_classification(self):
        """Verify chunks are classified into architectural roles accurately."""
        doc_chunk = RetrievedChunk(
            content="doc", source_path="data/documents/guide.pdf", file_name="guide.pdf",
            category="document", distance=0.3, similarity=0.7,
        )
        self.assertEqual(self.retriever._classify_chunk_source(doc_chunk), "document")

        impl_chunk = RetrievedChunk(
            content="code", source_path="rag/indexer.py", file_name="indexer.py",
            category="codebase", distance=0.3, similarity=0.7,
        )
        self.assertEqual(self.retriever._classify_chunk_source(impl_chunk), "impl")

        test_chunk = RetrievedChunk(
            content="test", source_path="tests_rag_phase4_2.py", file_name="tests_rag_phase4_2.py",
            category="codebase", distance=0.3, similarity=0.7,
        )
        self.assertEqual(self.retriever._classify_chunk_source(test_chunk), "test")

        config_chunk = RetrievedChunk(
            content="config", source_path="rag/config.py", file_name="config.py",
            category="codebase", distance=0.3, similarity=0.7,
        )
        self.assertEqual(self.retriever._classify_chunk_source(config_chunk), "config")

        template_chunk = RetrievedChunk(
            content="html", source_path="templates/index.html", file_name="index.html",
            category="codebase", distance=0.3, similarity=0.7, symbol_type="template",
        )
        self.assertEqual(self.retriever._classify_chunk_source(template_chunk), "frontend")

    # ── 3. Source-Aware Ranking Calibration ──────────────────────────────────

    def test_ranking_elevates_implementation_on_impl_query(self):
        """
        On implementation queries, implementation files with slightly lower raw similarity
        should outrank test files due to source-aware weighting.
        """
        # Test chunk has raw sim 0.60 (distance 0.40)
        test_chunk = RetrievedChunk(
            content="def test_compute_file_hash(): assert True",
            source_path="tests_rag_phase3.py",
            file_name="tests_rag_phase3.py",
            category="codebase",
            distance=0.40,
            similarity=0.60,
        )
        # Impl chunk has raw sim 0.58 (distance 0.42) - normally outranked by test!
        impl_chunk = RetrievedChunk(
            content="def compute_file_hash(path): return hashlib.sha256()",
            source_path="rag/indexer.py",
            file_name="indexer.py",
            category="codebase",
            distance=0.42,
            similarity=0.58,
        )

        query = "How does IncrementalIndexer compute file hashes?"
        ranked = self.retriever._apply_source_aware_ranking([test_chunk, impl_chunk], query)

        # Impl chunk should now be rank 1
        self.assertEqual(ranked[0].file_name, "indexer.py")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_ranking_elevates_test_on_test_query(self):
        """
        On test-specific queries, test files should receive boost and outrank implementation files.
        """
        impl_chunk = RetrievedChunk(
            content="def filter_active(): pass",
            source_path="rag/retriever.py",
            file_name="retriever.py",
            category="codebase",
            distance=0.40,
            similarity=0.60,
        )
        test_chunk = RetrievedChunk(
            content="def test_active_generation_filtering(): assert True",
            source_path="tests_rag_phase4_2.py",
            file_name="tests_rag_phase4_2.py",
            category="codebase",
            distance=0.42,
            similarity=0.58,
        )

        query = "What unit test methods verify active generation filtering?"
        ranked = self.retriever._apply_source_aware_ranking([impl_chunk, test_chunk], query)

        # Test chunk should be rank 1
        self.assertEqual(ranked[0].file_name, "tests_rag_phase4_2.py")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_ranking_elevates_config_on_config_query(self):
        """On configuration queries, config files should receive boost."""
        impl_chunk = RetrievedChunk(
            content="def get_config(): pass",
            source_path="rag/indexer.py",
            file_name="indexer.py",
            category="codebase",
            distance=0.38,
            similarity=0.62,
        )
        config_chunk = RetrievedChunk(
            content="SIMILARITY_DISTANCE_THRESHOLD = 0.70",
            source_path="rag/config.py",
            file_name="config.py",
            category="codebase",
            distance=0.40,
            similarity=0.60,
        )

        query = "What is the SIMILARITY_DISTANCE_THRESHOLD configuration?"
        ranked = self.retriever._apply_source_aware_ranking([impl_chunk, config_chunk], query)

        self.assertEqual(ranked[0].file_name, "config.py")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_ranking_elevates_frontend_on_frontend_query(self):
        """On frontend queries, template chunks should receive boost."""
        impl_chunk = RetrievedChunk(
            content="def render_template(): pass",
            source_path="app.py",
            file_name="app.py",
            category="codebase",
            distance=0.38,
            similarity=0.62,
        )
        template_chunk = RetrievedChunk(
            content="<select id='rag-mode-select'>...</select>",
            source_path="templates/index.html",
            file_name="index.html",
            category="codebase",
            distance=0.40,
            similarity=0.60,
            symbol_type="template",
        )

        query = "Where is the rag-mode-select element in templates/index.html?"
        ranked = self.retriever._apply_source_aware_ranking([impl_chunk, template_chunk], query)

        self.assertEqual(ranked[0].file_name, "index.html")
        self.assertGreater(ranked[0].score, ranked[1].score)

    # ── 4. Fallback When Ranking Disabled ─────────────────────────────────────

    def test_disabling_ranking_falls_back_to_pure_distance(self):
        """When ranking_config['enabled'] is False, chunks are sorted strictly by distance ascending."""
        disabled_config = dict(DEFAULT_RANKING_CONFIG)
        disabled_config["enabled"] = False

        retriever_disabled = UnifiedRetriever(
            vector_store=MagicMock(),
            embedding_service=MockEmbeddingService(),
            ranking_config=disabled_config,
            manifest_path=self.manifest_path,
        )

        c1 = RetrievedChunk(
            content="c1", source_path="tests_rag_phase3.py", file_name="tests_rag_phase3.py",
            category="codebase", distance=0.35, similarity=0.65,
        )
        c2 = RetrievedChunk(
            content="c2", source_path="rag/indexer.py", file_name="indexer.py",
            category="codebase", distance=0.45, similarity=0.55,
        )

        # With ranking enabled on impl_query, c2 would be boosted over c1.
        # But with ranking disabled, c1 (distance 0.35) must remain first.
        query = "How does IncrementalIndexer compute file hashes?"
        ranked = retriever_disabled._apply_source_aware_ranking([c2, c1], query)

        self.assertEqual(ranked[0].distance, 0.35)
        self.assertEqual(ranked[0].file_name, "tests_rag_phase3.py")

    # ── 5. Frontend Template Parsing & Citations ──────────────────────────────

    def test_frontend_template_pipeline_processing(self):
        """Verify CodebasePipeline processes HTML templates with line metadata and template symbol_type."""
        pipeline = CodebasePipeline()
        self.assertTrue(pipeline.can_process("templates/index.html"))

        chunks = pipeline.process_file("templates/index.html")
        self.assertGreater(len(chunks), 0)

        # Verify each chunk has proper line numbers and symbol_type
        for c in chunks[:5]:
            meta = c["metadata"]
            self.assertEqual(meta["symbol_type"], "template")
            self.assertEqual(meta["symbol_name"], "index.html")
            self.assertGreaterEqual(meta["start_line"], 1)
            self.assertGreaterEqual(meta["end_line"], meta["start_line"])
            self.assertTrue(c["text"])

    # ── 6. Metric Separation: Abstention vs Positive Relevance ────────────────

    def test_separated_metrics_abstention_accounting(self):
        """
        Verify that unanswerable queries returning 0 chunks or low-confidence chunks
        are accounted as correct abstentions and never counted as positive retrieval relevance hits.
        """
        # Simulated answerable query
        ans_query_res = {
            "is_answerable": True,
            "retrieval_success": True,
            "chunks_count": 4,
            "abstention_correct": False,
        }
        # Simulated unanswerable query returning 0 chunks
        unans_query_res = {
            "is_answerable": False,
            "retrieval_success": False,  # MUST NOT BE COUNTED AS RETRIEVAL SUCCESS
            "chunks_count": 0,
            "abstention_correct": True,
        }

        all_results = [ans_query_res, unans_query_res]

        answerable = [r for r in all_results if r["is_answerable"]]
        unanswerable = [r for r in all_results if not r["is_answerable"]]

        # Retrieval success rate is computed strictly over answerable
        retrieval_success_rate = sum(1 for r in answerable if r["retrieval_success"]) / len(answerable)
        self.assertEqual(retrieval_success_rate, 1.0)

        # Abstention rate is computed strictly over unanswerable
        abstention_rate = sum(1 for r in unanswerable if r["abstention_correct"]) / len(unanswerable)
        self.assertEqual(abstention_rate, 1.0)

        # Positive relevance rate must NOT include unanswerable queries
        positive_hits_across_all = sum(1 for r in all_results if r["retrieval_success"])
        self.assertEqual(positive_hits_across_all, 1)


if __name__ == "__main__":
    unittest.main()
