"""
Unit and integration tests for Ollama Embedding Leases and Adaptive Memory Coordination (Stage 3).

Verifies that:
1. Query embedding acquires and releases an embedding lease.
2. Batch embedding acquires and holds a lease across batch operations, releasing upon completion.
3. Exceptions and timeouts during embedding unconditionally release leases without leaks.
4. DeepSeek 16B waits for active embedding operations to finish before acquiring its generation lease.
5. Embedding operations wait while DeepSeek 16B holds an active generation lease under adaptive policy.
6. Eviction failure of EmbeddingGemma fails closed with ModelEvictionError and denies DeepSeek lease.
7. Persistent policy leaves EmbeddingGemma resident without eviction on lease release.
8. Ephemeral policy immediately evicts EmbeddingGemma when active embedding leases drop to 0.
9. Adaptive policy allows Gemma/Qwen to co-exist with EmbeddingGemma, but evicts EmbeddingGemma for DeepSeek.
10. Concurrent embedding and generation requests serialize cleanly without deadlock.
11. Existing embedding aliases (embed_text, embed_batch) and fallback/validation behavior remain intact.
"""

import contextlib
import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import requests

from model_lifecycle import (
    EmbeddingPolicy,
    ModelEvictionError,
    ModelLeaseCancelledError,
    ModelLeaseTimeoutError,
    ModelLeaseType,
    OllamaLifecycleManager,
    reset_model_manager_for_testing,
)
from rag.embeddings import (
    EmbeddingDimensionError,
    EmbeddingGenerationError,
    EmbeddingModelNotFoundError,
    EmbeddingService,
    OllamaConnectionError,
)


class MockOllamaDaemon:
    """Thread-safe simulated Ollama daemon for deterministic unit testing."""

    def __init__(self, resident_models=None):
        self._lock = threading.Lock()
        self.resident_models = set(resident_models or [])
        self.eviction_succeeds = True
        self.ps_status_code = 200
        self.embed_status_code = 200
        self.evicted_models = []
        self.embed_call_count = 0

    def mock_get(self, url, timeout=None):
        with self._lock:
            resp = MagicMock()
            if "/api/ps" in url:
                resp.status_code = self.ps_status_code
                if self.ps_status_code == 200:
                    resp.json.return_value = {
                        "models": [{"name": m, "model": m} for m in self.resident_models]
                    }
                else:
                    resp.text = "Error querying /api/ps"
            elif "/api/tags" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "models": [
                        {"name": "embeddinggemma:latest"},
                        {"name": "gemma4:12b"},
                        {"name": "deepseek-coder-v2:16b"},
                        {"name": "Qwen2.5-Coder:7B"},
                    ]
                }
            return resp

    def mock_post(self, url, json=None, timeout=None):
        with self._lock:
            resp = MagicMock()
            if "/api/generate" in url and json and json.get("keep_alive") == 0:
                m = json.get("model")
                self.evicted_models.append(m)
                if self.eviction_succeeds:
                    self.resident_models.discard(m)
                    self.resident_models.discard(f"{m}:latest" if ":" not in m else m)
                    resp.status_code = 200
                else:
                    resp.status_code = 500
                    resp.text = "Eviction failed"
                return resp

            if "/api/embed" in url:
                self.embed_call_count += 1
                resp.status_code = self.embed_status_code
                if self.embed_status_code == 200:
                    inputs = json.get("input", "")
                    if isinstance(inputs, list):
                        vectors = [[0.1] * 768 for _ in inputs]
                    else:
                        vectors = [[0.1] * 768]
                    resp.json.return_value = {"embeddings": vectors}
                else:
                    resp.text = "Internal server error"
                return resp

            resp.status_code = 200
            return resp


class TestModelLifecycleStage3(unittest.TestCase):
    def setUp(self):
        self.mock_daemon = MockOllamaDaemon()
        self.manager = OllamaLifecycleManager(
            base_url="http://localhost:11434",
            max_concurrent_generation=1,
            embedding_policy="adaptive",
            embedding_model_name="embeddinggemma:latest",
            default_timeout=2.0,
            poll_interval=0.02,
        )
        self.manager.session.get = self.mock_daemon.mock_get
        self.manager.session.post = self.mock_daemon.mock_post
        reset_model_manager_for_testing(self.manager)

        self.service = EmbeddingService(
            base_url="http://localhost:11434",
            model_name="embeddinggemma:latest",
            batch_size=2,
            timeout=5,
            expected_dimension=768,
        )
        # Mock requests in EmbeddingService to route through MockOllamaDaemon
        self.patcher_post = patch("rag.embeddings.requests.post", side_effect=self.mock_daemon.mock_post)
        self.patcher_get = patch("rag.embeddings.requests.get", side_effect=self.mock_daemon.mock_get)
        self.patcher_post.start()
        self.patcher_get.start()

    def tearDown(self):
        self.patcher_post.stop()
        self.patcher_get.stop()
        reset_model_manager_for_testing(None)

    def test_01_query_embedding_acquires_and_releases_lease(self):
        """Service embed_query acquires an embedding lease, executes /api/embed, and releases it."""
        lease_observed_active = []
        orig_acquire = self.manager.acquire_embedding_lease

        @contextlib.contextmanager
        def spy_acquire(**kwargs):
            with orig_acquire(**kwargs) as lease:
                lease_observed_active.append(lease.is_active)
                yield lease

        self.manager.acquire_embedding_lease = spy_acquire

        vec = self.service.embed_query("What is the system architecture?")
        self.assertEqual(len(vec), 768)
        self.assertTrue(any(lease_observed_active))
        self.assertEqual(self.manager.get_diagnostics()["active_embed_leases"], 0)

    def test_02_batch_embedding_acquires_and_holds_lease_across_batches(self):
        """Service embed_documents acquires lease across all batches and releases on completion."""
        active_counts = []
        orig_acquire = self.manager.acquire_embedding_lease

        @contextlib.contextmanager
        def spy_acquire(**kwargs):
            with orig_acquire(**kwargs) as lease:
                active_counts.append(self.manager.get_diagnostics()["active_embed_leases"])
                yield lease

        self.manager.acquire_embedding_lease = spy_acquire

        texts = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"]
        vectors = self.service.embed_documents(texts, batch_size=2)

        self.assertEqual(len(vectors), 5)
        for v in vectors:
            self.assertEqual(len(v), 768)

        # 5 items with batch_size=2 results in 3 /api/embed calls
        self.assertEqual(self.mock_daemon.embed_call_count, 3)
        self.assertTrue(all(c > 0 for c in active_counts))
        self.assertEqual(self.manager.get_diagnostics()["active_embed_leases"], 0)

    def test_03_embedding_exceptions_release_lease(self):
        """When /api/embed encounters an HTTP error or timeout, the embedding lease is released."""
        self.mock_daemon.embed_status_code = 500

        with self.assertRaises(EmbeddingGenerationError):
            self.service.embed_query("Query that will trigger 500")

        self.assertEqual(self.manager.get_diagnostics()["active_embed_leases"], 0)

    def test_04_deepseek_waits_for_active_embedding_operation(self):
        """Under adaptive policy, DeepSeek 16B cannot acquire generation lease while embedding is active."""
        self.manager.embedding_policy = EmbeddingPolicy.ADAPTIVE
        embed_active = threading.Event()
        release_embed = threading.Event()
        deepseek_acquired = threading.Event()

        def embed_worker():
            with self.manager.acquire_embedding_lease():
                embed_active.set()
                release_embed.wait(timeout=2.0)

        def deepseek_worker():
            embed_active.wait(timeout=2.0)
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
                deepseek_acquired.set()

        t_emb = threading.Thread(target=embed_worker)
        t_ds = threading.Thread(target=deepseek_worker)
        t_emb.start()
        t_ds.start()

        embed_active.wait(timeout=2.0)
        # DeepSeek must NOT have acquired the lease while embedding is held
        time.sleep(0.05)
        self.assertFalse(deepseek_acquired.is_set())

        # Release embedding
        release_embed.set()
        t_emb.join(timeout=2.0)
        t_ds.join(timeout=2.0)

        self.assertTrue(deepseek_acquired.is_set())
        self.assertEqual(self.manager.get_diagnostics()["active_embed_leases"], 0)

    def test_05_embedding_waits_while_deepseek_is_active(self):
        """Under adaptive policy, embedding requests wait while DeepSeek 16B holds an active generation lease."""
        self.manager.embedding_policy = EmbeddingPolicy.ADAPTIVE
        deepseek_active = threading.Event()
        release_deepseek = threading.Event()
        embed_acquired = threading.Event()

        def deepseek_worker():
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
                deepseek_active.set()
                release_deepseek.wait(timeout=2.0)

        def embed_worker():
            deepseek_active.wait(timeout=2.0)
            with self.manager.acquire_embedding_lease():
                embed_acquired.set()

        t_ds = threading.Thread(target=deepseek_worker)
        t_emb = threading.Thread(target=embed_worker)
        t_ds.start()
        t_emb.start()

        deepseek_active.wait(timeout=2.0)
        time.sleep(0.05)
        self.assertFalse(embed_acquired.is_set())

        release_deepseek.set()
        t_ds.join(timeout=2.0)
        t_emb.join(timeout=2.0)

        self.assertTrue(embed_acquired.is_set())
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    def test_06_eviction_failure_prevents_deepseek_acquisition(self):
        """If EmbeddingGemma cannot be evicted from Ollama, DeepSeek lease is denied (fails closed)."""
        self.manager.embedding_policy = EmbeddingPolicy.ADAPTIVE
        self.mock_daemon.resident_models = {"embeddinggemma:latest"}
        self.mock_daemon.eviction_succeeds = False

        with self.assertRaises(ModelEvictionError):
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
                pass  # pragma: no cover

        # Fails closed: DeepSeek was not granted a lease
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    def test_07_persistent_policy_behavior(self):
        """Under persistent policy, EmbeddingGemma is never evicted when its lease releases."""
        self.manager.embedding_policy = EmbeddingPolicy.PERSISTENT
        self.mock_daemon.resident_models = {"embeddinggemma:latest"}

        vec = self.service.embed_query("Sample text")
        self.assertEqual(len(vec), 768)

        # Eviction was not requested
        self.assertEqual(len(self.mock_daemon.evicted_models), 0)
        self.assertIn("embeddinggemma:latest", self.mock_daemon.resident_models)

    def test_08_ephemeral_policy_behavior(self):
        """Under ephemeral policy, EmbeddingGemma is evicted immediately when lease drops to 0."""
        self.manager.embedding_policy = EmbeddingPolicy.EPHEMERAL
        self.mock_daemon.resident_models = {"embeddinggemma:latest"}

        vec = self.service.embed_query("Sample text")
        self.assertEqual(len(vec), 768)

        # Eviction must have been issued for embeddinggemma
        self.assertTrue(any("embeddinggemma" in m for m in self.mock_daemon.evicted_models))
        self.assertNotIn("embeddinggemma:latest", self.mock_daemon.resident_models)

    def test_09_adaptive_policy_allows_gemma_and_qwen_coexistence(self):
        """Under adaptive policy, Gemma and Qwen do NOT evict EmbeddingGemma; only DeepSeek does."""
        self.manager.embedding_policy = EmbeddingPolicy.ADAPTIVE
        self.mock_daemon.resident_models = {"embeddinggemma:latest"}

        # Gemma 4 generation: EmbeddingGemma must NOT be evicted
        with self.manager.acquire_generation_lease("gemma4:12b"):
            self.assertIn("embeddinggemma:latest", self.mock_daemon.resident_models)

        # Qwen 2.5 generation: EmbeddingGemma must NOT be evicted
        with self.manager.acquire_generation_lease("Qwen2.5-Coder:7B"):
            self.assertIn("embeddinggemma:latest", self.mock_daemon.resident_models)

        # DeepSeek 16B generation: EmbeddingGemma MUST be evicted!
        with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
            self.assertNotIn("embeddinggemma:latest", self.mock_daemon.resident_models)

    def test_10_concurrent_embedding_and_generation_serialize_without_deadlock(self):
        """Concurrent embedding and generation threads execute to completion without circular wait."""
        self.manager.embedding_policy = EmbeddingPolicy.ADAPTIVE
        results = []
        lock = threading.Lock()

        def run_embed(text):
            vec = self.service.embed_query(text)
            with lock:
                results.append(("embed", len(vec)))

        def run_gemma():
            with self.manager.acquire_generation_lease("gemma4:12b"):
                time.sleep(0.04)
                with lock:
                    results.append(("gemma", True))

        def run_deepseek():
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
                time.sleep(0.04)
                with lock:
                    results.append(("deepseek", True))

        threads = [
            threading.Thread(target=run_embed, args=("Query 1",)),
            threading.Thread(target=run_gemma),
            threading.Thread(target=run_embed, args=("Query 2",)),
            threading.Thread(target=run_deepseek),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # All 4 operations finished cleanly without deadlock
        self.assertEqual(len(results), 4)
        embed_count = sum(1 for r in results if r[0] == "embed")
        self.assertEqual(embed_count, 2)
        self.assertEqual(self.manager.get_diagnostics()["active_embed_leases"], 0)
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    def test_11_embedding_aliases_and_validation(self):
        """Existing aliases (embed_text, embed_batch) and input validation function identically."""
        # 1. embed_text alias
        v1 = self.service.embed_text("Query via alias")
        self.assertEqual(len(v1), 768)

        # 2. embed_batch alias
        v_list = self.service.embed_batch(["Batch text 1", "Batch text 2"])
        self.assertEqual(len(v_list), 2)
        self.assertEqual(len(v_list[0]), 768)

        # 3. Empty input validation
        with self.assertRaises(ValueError):
            self.service.embed_query("")
        with self.assertRaises(ValueError):
            self.service.embed_query("   ")

        # 4. Empty list returns empty list
        self.assertEqual(self.service.embed_documents([]), [])

        # 5. verify_model_available throws EmbeddingModelNotFoundError when missing
        def get_missing_tags(url, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"models": []}
            return resp

        self.mock_daemon.mock_get = get_missing_tags
        unverified_service = EmbeddingService(model_name="nonexistent:model")
        with self.assertRaises(EmbeddingModelNotFoundError):
            unverified_service.verify_model_available(force=True)

    def test_12_end_to_end_rag_retrieval_and_deepseek_generation_lifecycle(self):
        """End-to-end flow: RAG embedding executes first, releases lease, then DeepSeek acquires lease."""
        import app

        client = app.app.test_client()

        events_order = []
        lock = threading.Lock()
        orig_acq_gen = self.manager.acquire_generation_lease
        orig_acq_emb = self.manager.acquire_embedding_lease

        @contextlib.contextmanager
        def spy_gen(target_model, **kwargs):
            with lock:
                events_order.append(("acquire_gen", target_model))
            try:
                with orig_acq_gen(target_model, **kwargs) as lease:
                    yield lease
            finally:
                with lock:
                    events_order.append(("release_gen", target_model))

        @contextlib.contextmanager
        def spy_emb(**kwargs):
            with lock:
                events_order.append(("acquire_embed", "embeddinggemma"))
            try:
                with orig_acq_emb(**kwargs) as lease:
                    yield lease
            finally:
                with lock:
                    events_order.append(("release_embed", "embeddinggemma"))

        self.manager.acquire_generation_lease = spy_gen
        self.manager.acquire_embedding_lease = spy_emb

        def mock_ollama_post(url, json=None, timeout=None):
            if "/api/chat" in url:
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.iter_lines.return_value = [
                    '{"message": {"content": "DeepSeek architectural design..."}, "done": false}',
                    '{"message": {"content": ""}, "done": true, "done_reason": "stop"}',
                ]
                resp.__enter__.return_value = resp
                return resp
            return self.mock_daemon.mock_post(url, json=json, timeout=timeout)

        with patch("app.http_requests.post", side_effect=mock_ollama_post):
            response = client.post(
                "/api/chat",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Perform an architectural assessment of our codebase.",
                        }
                    ],
                    "mode": "deepseek",
                    "rag_mode": "codebase",
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.get_data(as_text=True)

        event_names = [e[0] for e in events_order]
        # Must have executed embedding acquisition and release BEFORE generation acquisition
        self.assertIn("acquire_embed", event_names)
        self.assertIn("release_embed", event_names)
        self.assertIn("acquire_gen", event_names)
        self.assertIn("release_gen", event_names)

        idx_rel_embed = event_names.index("release_embed")
        idx_acq_gen = event_names.index("acquire_gen")
        # Critical ordering invariant: Embedding released before DeepSeek acquired!
        self.assertLess(idx_rel_embed, idx_acq_gen)
        self.assertEqual(self.manager.get_diagnostics()["active_embed_leases"], 0)
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})


if __name__ == "__main__":
    unittest.main()
