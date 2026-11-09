"""
Automated Test Suite for Unified RAG Phase 5
Integration of Unified RAG into the Existing Chatbot

Validates:
1. Greetings and pleasantries bypass retrieval (auto -> mode='off', status='off').
2. Document questions trigger document retrieval (auto -> mode='documents').
3. Repository and codebase questions trigger codebase retrieval (auto -> mode='codebase').
4. Combined questions trigger both retrieval pipelines (auto -> mode='both').
5. Manifest errors (missing or malformed) produce structured DATABASE_ERROR and never return unfiltered ChromaDB results.
6. Candidate starvation demonstration and resolution (oversampling and orphan cleanup).
7. Retrieval failure falls back gracefully without breaking the chatbot endpoint.
8. No-results queries remain fully functional.
9. Retrieved context reaches Gemma chat prompt with untrusted header and citation instructions.
10. DeepSeek/Qwen pipeline receives context (Architect receives codebase context, Coder receives blueprint + reference).
11. Gemini integration receives augmented prompt in Content parts.
12. Conversation history is NOT mutated or polluted across turns.
13. Server-Sent Events (SSE) streaming protocol and event contract are preserved.
14. Flask application routes remain fully intact (>= 12 rules).
"""

import base64
import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app
from rag.config import (
    CODEBASE_COLLECTION_NAME,
    DOCUMENT_COLLECTION_NAME,
    DATA_DIR,
)
from rag.indexer import IncrementalIndexer
from rag.prompt_builder import PromptBuilder, UNTRUSTED_CONTENT_HEADER
from rag.retriever import RetrievedChunk, RetrievalResponse, RetrievalStatus, UnifiedRetriever
from rag.service import (
    RAGContextResult,
    RAGService,
    detect_retrieval_mode,
    get_rag_service,
)
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


class TestRAGPhase5Integration(unittest.TestCase):
    """Phase 5 Test Suite: Unified RAG Chatbot Integration."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="rag_test_p5_")
        self.chroma_dir = os.path.join(self.test_dir, "chroma")
        self.manifest_dir = os.path.join(self.test_dir, "manifests")
        self.manifest_path = os.path.join(self.manifest_dir, "index_manifest.json")
        os.makedirs(self.manifest_dir, exist_ok=True)

        self.mock_embedder = DeterministicMockEmbedder(dimension=768)
        self.vector_store = VectorStoreManager(persist_dir=self.chroma_dir)
        self.vector_store.initialize()

        # Initialize clean empty manifest with documents and codebase keys
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "documents": {}, "codebase": {}}, f)

        self.retriever = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            top_k=5,
            distance_threshold=1.5,
        )

        self.rag_service = RAGService(
            retriever=self.retriever,
            prompt_builder=PromptBuilder(max_tokens=1000),
        )

        app.app.config["TESTING"] = True
        self.client = app.app.test_client()

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass

    # ── Test 1: Greetings Bypass Retrieval ──────────────────────────────────
    def test_01_greetings_bypass_retrieval(self):
        """Verify that greetings and pleasantries bypass retrieval and return mode='off'."""
        greetings = [
            "hello",
            "Hi there!",
            "good morning",
            "hey",
            "thanks",
            "thank you so much",
            "who are you?",
            "how are you doing",
        ]
        for query in greetings:
            mode = detect_retrieval_mode(query)
            self.assertEqual(mode, "off", f"Greeting '{query}' should resolve to 'off', got '{mode}'")

            result = self.rag_service.retrieve_context(query, mode="auto")
            self.assertFalse(result.has_context, f"Greeting '{query}' should not produce context")
            self.assertEqual(result.mode, "off")
            self.assertEqual(result.status, "off")
            self.assertEqual(result.context_text, "")
            self.assertEqual(result.sources, [])

    # ── Test 2: Document Questions Trigger Document Retrieval ──────────────
    def test_02_document_questions_trigger_document_retrieval(self):
        """Verify that document questions trigger document retrieval."""
        doc_queries = [
            "what does the user guide pdf say about deployment?",
            "according to the doc, how do we configure ssl certificates?",
            "find in documents: retention policy and compliance",
            "summarize the manual documentation",
            "what are the guidelines in the handbook?",
        ]
        for query in doc_queries:
            mode = detect_retrieval_mode(query)
            self.assertEqual(mode, "documents", f"Query '{query}' should resolve to 'documents', got '{mode}'")

    # ── Test 3: Repository Questions Trigger Codebase Retrieval ────────────
    def test_03_repository_questions_trigger_codebase_retrieval(self):
        """Verify that repository and source code questions trigger codebase retrieval."""
        code_queries = [
            "how are routes defined in app.py?",
            "explain the class structure in rag/indexer.py",
            "show me the functions in codebase for vector storage",
            "find in repo how stream_pipeline is implemented",
            "what endpoints exist in the project structure?",
        ]
        for query in code_queries:
            mode = detect_retrieval_mode(query)
            self.assertEqual(mode, "codebase", f"Query '{query}' should resolve to 'codebase', got '{mode}'")

    # ── Test 4: Combined Questions Trigger Both Pipelines ──────────────────
    def test_04_combined_questions_trigger_both(self):
        """Verify that combined questions trigger both retrieval pipelines."""
        both_queries = [
            "how does the documentation match the code implementation in app.py?",
            "compare the pdf manual with the repository source code",
            "does the doc specify the same parameters as the script functions?",
        ]
        for query in both_queries:
            mode = detect_retrieval_mode(query)
            self.assertEqual(mode, "both", f"Query '{query}' should resolve to 'both', got '{mode}'")

    # ── Test 5: Manifest Error Produces Structured Error ───────────────────
    def test_05_manifest_error_produces_structured_error(self):
        """Verify that missing, malformed, or unreadable manifest produces DATABASE_ERROR and never unfiltered chunks."""
        raw_vec = self.mock_embedder.embed_text("sample content")
        self.vector_store.add_documents(
            collection_name=DOCUMENT_COLLECTION_NAME,
            ids=["chunk_raw_1", "chunk_raw_2"],
            embeddings=[raw_vec, raw_vec],
            documents=["Sample doc 1", "Sample doc 2"],
            metadatas=[
                {"generation_id": "gen_orphan", "source_path": "orphan.pdf", "file_name": "orphan.pdf"},
                {"generation_id": "gen_orphan", "source_path": "orphan.pdf", "file_name": "orphan.pdf"},
            ],
        )

        # Case A: Missing manifest file
        missing_path = os.path.join(self.test_dir, "nonexistent_manifest.json")
        retriever_missing = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=missing_path,
        )
        resp_missing = retriever_missing.retrieve("sample content", mode="documents")
        self.assertEqual(resp_missing.status, RetrievalStatus.DATABASE_ERROR)
        self.assertIn("does not exist", resp_missing.error_message)
        self.assertEqual(len(resp_missing.chunks), 0, "Unfiltered chunks must NEVER be returned on missing manifest")

        # Case B: Malformed / Corrupted manifest JSON
        corrupt_path = os.path.join(self.test_dir, "corrupt_manifest.json")
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("{corrupt_json: invalid content [}")

        retriever_corrupt = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=corrupt_path,
        )
        resp_corrupt = retriever_corrupt.retrieve("sample content", mode="documents")
        self.assertEqual(resp_corrupt.status, RetrievalStatus.DATABASE_ERROR)
        self.assertIn("could not be read or parsed", resp_corrupt.error_message)
        self.assertEqual(len(resp_corrupt.chunks), 0, "Unfiltered chunks must NEVER be returned on corrupt manifest")

        # Case C: Manifest root is not a dict
        invalid_type_path = os.path.join(self.test_dir, "invalid_type_manifest.json")
        with open(invalid_type_path, "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)

        retriever_invalid_type = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=invalid_type_path,
        )
        resp_invalid = retriever_invalid_type.retrieve("sample content", mode="documents")
        self.assertEqual(resp_invalid.status, RetrievalStatus.DATABASE_ERROR)
        self.assertIn("not a dictionary", resp_invalid.error_message)
        self.assertEqual(len(resp_invalid.chunks), 0, "Unfiltered chunks must NEVER be returned on invalid manifest format")

    # ── Test 6: Candidate Starvation Demonstration & Resolution ────────────
    def test_06_candidate_starvation_demonstration(self):
        """
        Demonstrate candidate starvation:
        - 30 stale orphaned chunks with perfect cosine distance (0.0) crowd out 2 active chunks.
        - With oversample_factor=1 (no oversampling), starvation causes 0 chunks to be returned.
        - With oversampling (min_candidates=35) or orphan cleanup, active chunks are successfully returned.
        """
        query_text = "target query text"
        target_vec = self.mock_embedder.embed_text(query_text)

        # Active chunks vector has slight distance
        active_vec = [v for v in target_vec]
        active_vec[0] = (active_vec[0] + 0.05)
        norm = sum(x * x for x in active_vec) ** 0.5
        active_vec = [x / norm for x in active_vec]

        # 1. Insert 2 active chunks
        active_gen = "gen_active_1"
        self.vector_store.add_documents(
            collection_name=CODEBASE_COLLECTION_NAME,
            ids=["chunk_active_1", "chunk_active_2"],
            embeddings=[active_vec, active_vec],
            documents=["Active codebase logic chunk 1", "Active codebase logic chunk 2"],
            metadatas=[
                {"generation_id": active_gen, "source_path": "valid.py", "file_name": "valid.py"},
                {"generation_id": active_gen, "source_path": "valid.py", "file_name": "valid.py"},
            ],
        )

        # Register active chunks in manifest under codebase category
        manifest_data = {
            "version": 1,
            "documents": {},
            "codebase": {
                "valid.py": {
                    "generation_id": active_gen,
                    "chunk_ids": ["chunk_active_1", "chunk_active_2"],
                }
            },
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        # 2. Insert 30 stale orphaned chunks with perfect distance to query
        stale_gen = "gen_stale_orphan"
        stale_ids = [f"chunk_stale_{i}" for i in range(30)]
        stale_vecs = [target_vec for _ in range(30)]
        stale_docs = [f"Stale orphaned chunk {i}" for i in range(30)]
        stale_metas = [{"generation_id": stale_gen, "source_path": "old.py", "file_name": "old.py"} for _ in range(30)]
        self.vector_store.add_documents(
            collection_name=CODEBASE_COLLECTION_NAME,
            ids=stale_ids,
            embeddings=stale_vecs,
            documents=stale_docs,
            metadatas=stale_metas,
        )

        # SCENARIO A: Starvation without sufficient oversampling
        starved_retriever = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            top_k=2,
            distance_threshold=1.5,
            oversample_factor=1,
            min_candidates=2,  # Forces ChromaDB to fetch only 2 candidates
        )
        resp_starved = starved_retriever.retrieve(query_text, mode="codebase")
        self.assertEqual(len(resp_starved.chunks), 0, "Candidate starvation should yield 0 results when oversampling is disabled")

        # SCENARIO B: Oversampling solves starvation
        healthy_retriever = UnifiedRetriever(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            top_k=2,
            distance_threshold=1.5,
            oversample_factor=5,
            min_candidates=35,
        )
        resp_healthy = healthy_retriever.retrieve(query_text, mode="codebase")
        self.assertEqual(len(resp_healthy.chunks), 2, "Oversampling must capture and return active chunks")
        for chunk in resp_healthy.chunks:
            self.assertEqual(chunk.source_path, "valid.py")

        # SCENARIO C: Orphan cleanup permanently solves starvation
        indexer = IncrementalIndexer(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedder,
            manifest_path=self.manifest_path,
            base_dir=self.test_dir,
        )
        cleanup_res = indexer.clean_orphaned_generations(CODEBASE_COLLECTION_NAME)
        self.assertEqual(cleanup_res["orphaned_chunks_deleted"], 30)

        # Now even the starved retriever succeeds because orphans were removed from ChromaDB
        resp_after_cleanup = starved_retriever.retrieve(query_text, mode="codebase")
        self.assertEqual(len(resp_after_cleanup.chunks), 2, "After orphan cleanup, retrieval succeeds without starvation")

    # ── Test 7: Retrieval Failure Falls Back Gracefully ────────────────────
    def test_07_retrieval_failure_falls_back_gracefully(self):
        """Verify that retrieval exceptions are caught gracefully and do not break /api/chat."""
        broken_retriever = MagicMock()
        broken_retriever.retrieve.side_effect = RuntimeError("ChromaDB connection exploded")
        failing_rag_service = RAGService(retriever=broken_retriever)

        res = failing_rag_service.retrieve_context("query about app.py", mode="codebase")
        self.assertFalse(res.has_context)
        self.assertEqual(res.status, "error")
        self.assertIn("exploded", res.error_message)

        # Test /api/chat endpoint fallback
        with patch("app.get_rag_service", return_value=failing_rag_service):
            with patch("app.http_requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.iter_lines.return_value = [
                    json.dumps({"message": {"content": "Hello from fallback!"}, "done": True}).encode("utf-8")
                ]
                mock_post.return_value = mock_resp
                mock_post.return_value.__enter__.return_value = mock_resp

                post_data = {
                    "messages": [{"role": "user", "content": "Tell me about app.py"}],
                    "mode": "chat",
                    "rag_mode": "codebase",
                }
                with self.client:
                    response = self.client.post("/api/chat", json=post_data)
                    self.assertEqual(response.status_code, 200)
                    content = b"".join(response.response).decode("utf-8")
                    self.assertIn("Hello from fallback!", content)
                    self.assertIn('"rag_status": "error"', content)

    # ── Test 8: No Results Remains Functional ──────────────────────────────
    def test_08_no_results_remains_functional(self):
        """Verify that when retrieval finds no matching chunks, chatbot runs normally without context."""
        empty_retriever = MagicMock()
        empty_retriever.retrieve.return_value = RetrievalResponse(
            chunks=[],
            query="obscure keyword",
            status=RetrievalStatus.NO_RESULTS,
        )
        rag_service = RAGService(retriever=empty_retriever)
        res = rag_service.retrieve_context("obscure keyword", mode="documents")
        self.assertFalse(res.has_context)
        self.assertEqual(res.status, "no_results")
        self.assertEqual(res.context_text, "")

        with patch("app.get_rag_service", return_value=rag_service):
            with patch("app.http_requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.iter_lines.return_value = [
                    json.dumps({"message": {"content": "Direct assistant response."}, "done": True}).encode("utf-8")
                ]
                mock_post.return_value = mock_resp
                mock_post.return_value.__enter__.return_value = mock_resp

                post_data = {
                    "messages": [{"role": "user", "content": "What is obscure keyword?"}],
                    "mode": "chat",
                }
                with self.client:
                    response = self.client.post("/api/chat", json=post_data)
                    self.assertEqual(response.status_code, 200)
                    b"".join(response.response)

                    called_kwargs = mock_post.call_args[1]
                    payload_messages = called_kwargs["json"]["messages"]
                    user_msg = payload_messages[-1]["content"]
                    self.assertEqual(user_msg, "What is obscure keyword?")
                    self.assertNotIn(UNTRUSTED_CONTENT_HEADER, user_msg)

    # ── Test 9: Retrieved Context Reaches Gemma Prompt ─────────────────────
    def test_09_retrieved_context_reaches_gemma_prompt(self):
        """Verify that retrieved context reaches Gemma chat prompt with untrusted header and citation instructions."""
        mock_chunks = [
            RetrievedChunk(
                chunk_id="chunk_1",
                content="Antigravity architecture specifies a modular RAG pipeline.",
                source_path="docs/architecture.pdf",
                file_name="architecture.pdf",
                category="document",
                distance=0.12,
                similarity=0.88,
                page_number=4,
            )
        ]
        retriever_mock = MagicMock()
        retriever_mock.retrieve.return_value = RetrievalResponse(
            chunks=mock_chunks,
            query="tell me about architecture",
            status=RetrievalStatus.SUCCESS,
        )
        mock_service = RAGService(retriever=retriever_mock)

        with patch("app.get_rag_service", return_value=mock_service):
            with patch("app.http_requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.iter_lines.return_value = [
                    json.dumps({"message": {"content": "Based on [Document: architecture.pdf, Page 4]..."}, "done": True}).encode("utf-8")
                ]
                mock_post.return_value = mock_resp
                mock_post.return_value.__enter__.return_value = mock_resp

                post_data = {
                    "messages": [{"role": "user", "content": "What does the architecture document say?"}],
                    "mode": "chat",
                    "rag_mode": "documents",
                }
                with self.client:
                    response = self.client.post("/api/chat", json=post_data)
                    self.assertEqual(response.status_code, 200)
                    b"".join(response.response)

                    called_json = mock_post.call_args[1]["json"]
                    sent_messages = called_json["messages"]
                    self.assertEqual(sent_messages[0]["role"], "system")
                    self.assertEqual(sent_messages[0]["content"], app.CHAT_SYSTEM_PROMPT)

                    sent_user_content = sent_messages[-1]["content"]
                    self.assertIn(UNTRUSTED_CONTENT_HEADER, sent_user_content)
                    self.assertIn("architecture.pdf", sent_user_content)
                    self.assertIn("Antigravity architecture specifies a modular RAG pipeline.", sent_user_content)
                    self.assertIn("What does the architecture document say?", sent_user_content)

    # ── Test 10: DeepSeek & Qwen Pipeline Receives Context ──────────────────
    def test_10_deepseek_qwen_pipeline_receives_context(self):
        """Verify that DeepSeek (Architect) and Qwen (Coder) both receive context appropriately in pipeline mode."""
        mock_chunks = [
            RetrievedChunk(
                chunk_id="chunk_code_1",
                content="class UnifiedRetriever: def retrieve(self, query): pass",
                source_path="rag/retriever.py",
                file_name="retriever.py",
                category="codebase",
                distance=0.08,
                similarity=0.92,
                start_line=10,
                end_line=20,
            )
        ]
        retriever_mock = MagicMock()
        retriever_mock.retrieve.return_value = RetrievalResponse(
            chunks=mock_chunks,
            query="how does retriever work",
            status=RetrievalStatus.SUCCESS,
        )
        mock_service = RAGService(retriever=retriever_mock)

        with patch("app.get_rag_service", return_value=mock_service):
            with patch("app.http_requests.post") as mock_post:
                arch_mock_resp = MagicMock()
                arch_mock_resp.iter_lines.return_value = [
                    json.dumps({"message": {"content": "Architectural Blueprint for retriever"}, "done": True}).encode("utf-8")
                ]
                coder_mock_resp = MagicMock()
                coder_mock_resp.iter_lines.return_value = [
                    json.dumps({"message": {"content": "def test_retriever(): pass"}, "done": True}).encode("utf-8")
                ]

                # Return arch on first call, coder on second call
                mock_post.side_effect = [arch_mock_resp, coder_mock_resp]
                mock_post.return_value.__enter__.side_effect = [arch_mock_resp, coder_mock_resp]

                post_data = {
                    "messages": [{"role": "user", "content": "Write unit tests for the retriever in rag/retriever.py"}],
                    "mode": "pipeline",
                    "rag_mode": "codebase",
                }
                with self.client:
                    response = self.client.post("/api/chat", json=post_data)
                    self.assertEqual(response.status_code, 200)
                    b"".join(response.response)

                    self.assertEqual(mock_post.call_count, 2)

                    # Call 1: Architect
                    arch_call_json = mock_post.call_args_list[0][1]["json"]
                    self.assertEqual(arch_call_json["messages"][0]["content"], app.ARCHITECT_SYSTEM_PROMPT)
                    arch_user_text = arch_call_json["messages"][-1]["content"]
                    self.assertIn("retriever.py", arch_user_text)
                    self.assertIn("class UnifiedRetriever", arch_user_text)

                    # Call 2: Coder
                    coder_call_json = mock_post.call_args_list[1][1]["json"]
                    self.assertEqual(coder_call_json["messages"][0]["content"], app.CODER_SYSTEM_PROMPT)
                    coder_user_text = coder_call_json["messages"][-1]["content"]
                    self.assertIn("ARCHITECTURAL BLUEPRINT", coder_user_text)
                    self.assertIn("Architectural Blueprint for retriever", coder_user_text)
                    self.assertIn("REPOSITORY REFERENCE CONTEXT", coder_user_text)
                    self.assertIn("retriever.py", coder_user_text)

    # ── Test 11: Gemini Integration Receives Context ───────────────────────
    def test_11_gemini_integration_receives_context(self):
        """Verify that Gemini API integration receives augmented prompt in Content parts."""
        mock_chunks = [
            RetrievedChunk(
                chunk_id="chunk_gemini_1",
                content="The company's Q3 revenue reached 10M USD.",
                source_path="financials.pdf",
                file_name="financials.pdf",
                category="document",
                distance=0.15,
                similarity=0.85,
                page_number=1,
            )
        ]
        retriever_mock = MagicMock()
        retriever_mock.retrieve.return_value = RetrievalResponse(
            chunks=mock_chunks,
            query="what was Q3 revenue",
            status=RetrievalStatus.SUCCESS,
        )
        mock_service = RAGService(retriever=retriever_mock)

        # Mock google.genai
        mock_client = MagicMock()
        mock_stream_chunk = MagicMock()
        mock_stream_chunk.text = "Q3 revenue was 10M USD according to [Document: financials.pdf, Page 1]."
        mock_client.models.generate_content_stream.return_value = [mock_stream_chunk]

        with patch("app.get_rag_service", return_value=mock_service):
            with patch("app.get_gemini_api_key", return_value="dummy-gemini-key"):
                with patch("google.genai.Client", return_value=mock_client):
                    post_data = {
                        "messages": [{"role": "user", "content": "What was the Q3 revenue in the report?"}],
                        "mode": "gemini",
                        "rag_mode": "documents",
                    }
                    with self.client:
                        response = self.client.post("/api/chat", json=post_data)
                        self.assertEqual(response.status_code, 200)
                        b"".join(response.response)

                        mock_client.models.generate_content_stream.assert_called_once()
                        kwargs = mock_client.models.generate_content_stream.call_args[1]
                        contents = kwargs["contents"]
                        user_content = contents[-1]
                        self.assertEqual(user_content.role, "user")
                        part_text = user_content.parts[0].text
                        self.assertIn(UNTRUSTED_CONTENT_HEADER, part_text)
                        self.assertIn("financials.pdf", part_text)
                        self.assertIn("The company's Q3 revenue reached 10M USD.", part_text)
                        self.assertIn("What was the Q3 revenue in the report?", part_text)

    # ── Test 12: Conversation History Not Polluted ─────────────────────────
    def test_12_conversation_history_not_polluted(self):
        """Verify that conversation history is not mutated in-place and does not accumulate context blocks."""
        messages = [
            {"role": "user", "content": "What is defined in app.py?"}
        ]
        messages_snapshot = json.loads(json.dumps(messages))

        mock_chunks = [
            RetrievedChunk(
                chunk_id="chunk_app_1",
                content="@app.route('/api/chat')",
                source_path="app.py",
                file_name="app.py",
                category="codebase",
                distance=0.1,
                similarity=0.9,
                start_line=387,
                end_line=400,
            )
        ]
        retriever_mock = MagicMock()
        retriever_mock.retrieve.return_value = RetrievalResponse(
            chunks=mock_chunks,
            query="What is defined in app.py?",
            status=RetrievalStatus.SUCCESS,
        )
        mock_service = RAGService(retriever=retriever_mock)

        with patch("app.get_rag_service", return_value=mock_service):
            with patch("app.http_requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.iter_lines.return_value = [
                    json.dumps({"message": {"content": "Routes are defined in app.py."}, "done": True}).encode("utf-8")
                ]
                mock_post.return_value = mock_resp
                mock_post.return_value.__enter__.return_value = mock_resp

                with self.client:
                    response = self.client.post("/api/chat", json={"messages": messages, "mode": "chat"})
                    self.assertEqual(response.status_code, 200)
                    b"".join(response.response)

                    # Check that caller's messages list was NOT modified
                    self.assertEqual(messages, messages_snapshot)
                    self.assertEqual(messages[0]["content"], "What is defined in app.py?")

                    # Simulate Turn 2
                    messages.append({"role": "assistant", "content": "Routes are defined in app.py."})
                    messages.append({"role": "user", "content": "Now what about config.py?"})

                    response2 = self.client.post("/api/chat", json={"messages": messages, "mode": "chat"})
                    self.assertEqual(response2.status_code, 200)
                    b"".join(response2.response)

                    # In Ollama payload for Turn 2, turn 1 must remain clean without context
                    sent_json = mock_post.call_args[1]["json"]
                    sent_msgs = sent_json["messages"]
                    turn1_user = sent_msgs[1]
                    self.assertEqual(turn1_user["content"], "What is defined in app.py?")
                    self.assertNotIn(UNTRUSTED_CONTENT_HEADER, turn1_user["content"])

    # ── Test 13: SSE Response Format Preserved ─────────────────────────────
    def test_13_sse_response_format_preserved(self):
        """Verify that SSE event stream follows the exact contract: route -> stage_start -> token -> stage_end -> done."""
        with patch("app.http_requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.iter_lines.return_value = [
                json.dumps({"message": {"content": "Word1 "}, "done": False}).encode("utf-8"),
                json.dumps({"message": {"content": "Word2."}, "done": True}).encode("utf-8"),
            ]
            mock_post.return_value = mock_resp
            mock_post.return_value.__enter__.return_value = mock_resp

            post_data = {
                "messages": [{"role": "user", "content": "Hi there!"}],
                "mode": "chat",
            }
            with self.client:
                response = self.client.post("/api/chat", json=post_data)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "text/event-stream")

                raw_stream = b"".join(response.response).decode("utf-8")
                events = []
                for line in raw_stream.split("\n"):
                    line = line.strip()
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))

                # 1. Route event
                self.assertEqual(events[0]["type"], "route")
                self.assertIn("route", events[0])
                self.assertIn("rag_mode", events[0])
                self.assertIn("rag_status", events[0])
                self.assertIn("rag_sources", events[0])

                # 2. Stage start
                self.assertEqual(events[1]["type"], "stage_start")
                self.assertEqual(events[1]["stage"], "chat")

                # 3. Tokens
                self.assertEqual(events[2]["type"], "token")
                self.assertEqual(events[2]["token"], "Word1 ")
                self.assertEqual(events[3]["type"], "token")
                self.assertEqual(events[3]["token"], "Word2.")

                # 4. Done event
                self.assertEqual(events[4]["type"], "done")
                self.assertTrue(events[4]["done"])

    # ── Test 14: Flask Routes Intact ───────────────────────────────────────
    def test_14_flask_routes_intact(self):
        """Verify that all Flask routes exist, route rules count >= 12, and basic endpoints respond."""
        rules = list(app.app.url_map.iter_rules())
        self.assertGreaterEqual(len(rules), 12, f"Expected at least 12 route rules, found {len(rules)}")

        endpoints = {r.endpoint for r in rules}
        expected_endpoints = {
            "index",
            "health",
            "get_settings",
            "update_settings",
            "verify_gemini",
            "list_models",
            "aider_info",
            "chat",
            "upload_image",
            "voice_status",
            "transcribe_audio_endpoint",
        }
        for ep in expected_endpoints:
            self.assertIn(ep, endpoints, f"Missing expected endpoint '{ep}' in Flask route map")

        # Check /api/health
        health_resp = self.client.get("/api/health")
        self.assertEqual(health_resp.status_code, 200)
        health_json = health_resp.get_json()
        self.assertIn("models", health_json)
        self.assertIn("chat_model", health_json)

        # Check /api/models
        models_resp = self.client.get("/api/models")
        self.assertEqual(models_resp.status_code, 200)
        self.assertIn("models", models_resp.get_json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
