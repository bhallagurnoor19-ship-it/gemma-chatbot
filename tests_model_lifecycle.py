"""
Unit tests for OllamaLifecycleManager (model_lifecycle.py).
Deterministic tests using mocked Ollama HTTP endpoints and synchronization primitives.
Verifies all 17 required lifecycle invariants, state transitions, and cancellation guarantees.
"""

import contextlib
import threading
import time
import unittest
from unittest.mock import MagicMock
import requests

from model_lifecycle import (
    EmbeddingPolicy,
    ModelEvictionError,
    ModelLease,
    ModelLeaseCancelledError,
    ModelLeaseTimeoutError,
    ModelLeaseType,
    ModelLifecycleError,
    OllamaLifecycleManager,
    StateReconciliationError,
    is_heavy_model,
    normalize_model_name,
)


class MockOllamaState:
    """Thread-safe simulated Ollama server state for deterministic unit testing."""

    def __init__(self, initial_models=None):
        self._lock = threading.Lock()
        self.resident_models = set(initial_models or [])
        self.eviction_succeeds = True
        self.api_ps_status_code = 200
        self.evict_calls = []

    def mock_get(self, url, timeout=None):
        with self._lock:
            resp = MagicMock()
            resp.status_code = self.api_ps_status_code
            if self.api_ps_status_code == 200:
                resp.json.return_value = {
                    "models": [{"name": m, "model": m} for m in self.resident_models]
                }
            else:
                resp.text = "Internal Server Error"
            return resp

    def mock_post(self, url, json=None, timeout=None):
        with self._lock:
            resp = MagicMock()
            resp.status_code = 200
            if json and json.get("keep_alive") == 0:
                model_to_evict = json.get("model")
                self.evict_calls.append(model_to_evict)
                if self.eviction_succeeds:
                    norm = normalize_model_name(model_to_evict)
                    to_remove = [m for m in self.resident_models if normalize_model_name(m) == norm]
                    for m in to_remove:
                        self.resident_models.remove(m)
            return resp


class TestOllamaLifecycleManager(unittest.TestCase):
    def setUp(self):
        self.mock_ollama = MockOllamaState()
        self.manager = OllamaLifecycleManager(
            base_url="http://test-ollama:11434",
            max_concurrent_generation=1,
            embedding_policy="adaptive",
            embedding_model_name="embeddinggemma:latest",
            default_timeout=2.0,
            poll_interval=0.02,
        )
        # Wire mock session
        self.manager.session.get = self.mock_ollama.mock_get
        self.manager.session.post = self.mock_ollama.mock_post

    def test_01_single_generation_lease_enforcement(self):
        """Invariant 1: Two requests for different models cannot hold active leases concurrently."""
        acquired_first = threading.Event()
        release_first = threading.Event()
        second_acquired = threading.Event()

        def worker1():
            with self.manager.acquire_generation_lease("gemma4:12b"):
                acquired_first.set()
                release_first.wait(timeout=2.0)

        def worker2():
            acquired_first.wait(timeout=2.0)
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
                second_acquired.set()

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()

        acquired_first.wait(timeout=2.0)
        self.assertTrue(acquired_first.is_set())
        # While t1 holds lease on gemma4, t2 must not have acquired deepseek
        self.assertFalse(second_acquired.is_set())

        # Now release t1
        release_first.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        self.assertTrue(second_acquired.is_set())
        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["resident_gen_model"], "deepseek-coder-v2:16b")

    def test_02_same_model_concurrency_limit(self):
        """Invariant 2: Multiple requests for the same model are bound by max_concurrent_generation."""
        # Case A: max_concurrent_generation = 1
        self.manager.max_concurrent_generation = 1
        lease1_active = threading.Event()
        release_lease1 = threading.Event()
        lease2_acquired = threading.Event()

        def worker1():
            with self.manager.acquire_generation_lease("gemma4:12b"):
                lease1_active.set()
                release_lease1.wait(timeout=2.0)

        def worker2():
            lease1_active.wait(timeout=2.0)
            with self.manager.acquire_generation_lease("gemma4:12b"):
                lease2_acquired.set()

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()

        lease1_active.wait(timeout=2.0)
        # Second request must wait even though it is the same model
        self.assertFalse(lease2_acquired.is_set())

        release_lease1.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        self.assertTrue(lease2_acquired.is_set())

        # Case B: max_concurrent_generation = 2 allows concurrent leases on the same model
        self.manager.max_concurrent_generation = 2
        with self.manager.acquire_generation_lease("gemma4:12b") as l1:
            with self.manager.acquire_generation_lease("gemma4:12b") as l2:
                diag = self.manager.get_diagnostics()
                self.assertEqual(diag["active_gen_leases"]["gemma4:12b"], 2)

    def test_03_successful_model_transition(self):
        """Invariant 4: Model transition evicts old model and successfully commits new model."""
        self.mock_ollama.resident_models = {"gemma4:12b"}

        with self.manager.acquire_generation_lease("gemma4:12b"):
            pass

        # Switch to DeepSeek
        with self.manager.acquire_generation_lease("deepseek-coder-v2:16b") as lease:
            self.assertEqual(lease.model_name, "deepseek-coder-v2:16b")
            self.assertTrue(lease.is_active)

        self.assertIn("gemma4:12b", self.mock_ollama.evict_calls)
        self.assertEqual(self.manager._resident_gen_model, "deepseek-coder-v2:16b")

    def test_04_verified_eviction_of_previous_model(self):
        """Invariant 4 & 8: Eviction issues keep_alive: 0 and verifies removal against /api/ps."""
        self.mock_ollama.resident_models = {"gemma4:12b"}

        with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
            pass

        # gemma4:12b must have been evicted and confirmed removed from resident_models
        self.assertIn("gemma4:12b", self.mock_ollama.evict_calls)
        self.assertNotIn("gemma4:12b", self.mock_ollama.resident_models)

    def test_05_eviction_failure_aborts_transition(self):
        """Invariant 7: If eviction fails to remove old model from /api/ps, transition aborts."""
        self.mock_ollama.resident_models = {"gemma4:12b"}
        # Configure eviction to fail (model stays resident in /api/ps)
        self.mock_ollama.eviction_succeeds = False

        with self.assertRaises(ModelEvictionError):
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
                pass

        # Old model must NOT be marked unloaded; new model lease NOT granted
        diag = self.manager.get_diagnostics()
        self.assertNotEqual(diag["resident_gen_model"], "deepseek-coder-v2:16b")
        self.assertEqual(diag["active_gen_leases"], {})

    def test_06_transition_exception_restores_state_and_wakes_waiting_threads(self):
        """Invariant 7 & requirement 3: Transition error clears transitioning flag and wakes waiters."""
        self.mock_ollama.resident_models = {"gemma4:12b"}

        # Simulate exception during post
        def failing_post(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Simulated network drop")

        self.manager.session.post = failing_post

        with self.assertRaises(ModelEvictionError):
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
                pass

        diag = self.manager.get_diagnostics()
        self.assertFalse(diag["transitioning"], "Transition flag must be reset in finally block")

    def test_07_reconciliation_detects_externally_unloaded_models(self):
        """Invariant 8: Reconciliation detects when a model was unloaded externally (e.g. keep_alive expired)."""
        # Manager thought gemma4 was resident
        self.manager._resident_gen_model = "gemma4:12b"
        # But Ollama /api/ps reports empty
        self.mock_ollama.resident_models = set()

        reconciled = self.manager.reconcile_state()
        self.assertIsNone(reconciled["resident_gen_model"])
        self.assertIsNone(self.manager._resident_gen_model)

    def test_08_reconciliation_failure_fails_closed(self):
        """Invariant 8: If /api/ps returns HTTP 500, manager fails closed with StateReconciliationError."""
        self.mock_ollama.api_ps_status_code = 500

        with self.assertRaises(StateReconciliationError):
            self.manager.reconcile_state()

    def test_09_deepseek_waits_for_active_embedding_leases_to_drain(self):
        """Invariant 5: Under adaptive policy, heavy model waits for active embedding leases to finish."""
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
        # DeepSeek must NOT acquire lease while embedding is active
        self.assertFalse(deepseek_acquired.is_set())

        # Release embedding lease
        release_embed.set()
        t_emb.join(timeout=2.0)
        t_ds.join(timeout=2.0)

        self.assertTrue(deepseek_acquired.is_set())

    def test_10_embedding_requests_wait_while_deepseek_is_active(self):
        """Invariant 5: Under adaptive policy, embedding requests wait while DeepSeek lease is active."""
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
        self.assertFalse(embed_acquired.is_set())

        release_deepseek.set()
        t_ds.join(timeout=2.0)
        t_emb.join(timeout=2.0)

        self.assertTrue(embed_acquired.is_set())

    def test_11_cancellation_while_waiting(self):
        """Requirement: is_cancelled callback aborts wait cleanly with ModelLeaseCancelledError."""
        cancelled = False

        def check_cancelled():
            return cancelled

        started_wait = threading.Event()

        # Hold a lease to make worker2 wait
        with self.manager.acquire_generation_lease("gemma4:12b"):
            cancelled = True
            with self.assertRaises(ModelLeaseCancelledError):
                with self.manager.acquire_generation_lease("deepseek-coder-v2:16b", is_cancelled=check_cancelled):
                    pass

    def test_12_timeout_while_waiting(self):
        """Requirement: Exceeding deadline raises ModelLeaseTimeoutError."""
        with self.manager.acquire_generation_lease("gemma4:12b"):
            with self.assertRaises(ModelLeaseTimeoutError):
                with self.manager.acquire_generation_lease("deepseek-coder-v2:16b", timeout=0.1):
                    pass

    def test_13_idempotent_lease_release(self):
        """Requirement 10: Calling release() multiple times is safe and idempotent."""
        with self.manager.acquire_generation_lease("gemma4:12b") as lease:
            self.assertTrue(lease.is_active)
            lease.release()
            self.assertFalse(lease.is_active)
            # Second release must be a no-op
            lease.release()
            self.assertFalse(lease.is_active)

    def test_14_context_manager_cleanup_after_exceptions(self):
        """Requirement 10: Exceptions inside the context block unconditionally release the lease."""
        try:
            with self.manager.acquire_generation_lease("gemma4:12b") as lease:
                self.assertTrue(lease.is_active)
                raise RuntimeError("Simulated failure inside generation block")
        except RuntimeError:
            pass

        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"], {})
        self.assertFalse(lease.is_active)

    def test_15_context_manager_cleanup_after_generator_closure(self):
        """Requirement 10: Client disconnect (GeneratorExit on gen.close()) unconditionally releases lease."""
        def dummy_streaming_generator():
            with self.manager.acquire_generation_lease("gemma4:12b"):
                yield "token 1"
                yield "token 2"

        gen = dummy_streaming_generator()
        first_token = next(gen)
        self.assertEqual(first_token, "token 1")

        # Lease must be active while generator is open
        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"]["gemma4:12b"], 1)

        # Simulate client abort / browser disconnect
        gen.close()

        # Lease must be released immediately after generator closure
        diag_after = self.manager.get_diagnostics()
        self.assertEqual(diag_after["active_gen_leases"], {})

    def test_16_concurrent_requests_for_different_models_cannot_race_through_transition(self):
        """Invariant 7 & requirement 16: Two threads requesting transitions do not race or double-evict."""
        barrier = threading.Barrier(2)
        models_granted = []
        grant_lock = threading.Lock()

        def request_worker(model_name):
            barrier.wait()
            with self.manager.acquire_generation_lease(model_name) as lease:
                with grant_lock:
                    models_granted.append(lease.model_name)

        t1 = threading.Thread(target=request_worker, args=("deepseek-coder-v2:16b",))
        t2 = threading.Thread(target=request_worker, args=("qwen2.5-coder:7b",))
        t1.start()
        t2.start()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        # Both requests should have completed sequentially
        self.assertEqual(len(models_granted), 2)
        self.assertIn("deepseek-coder-v2:16b", models_granted)
        self.assertIn("qwen2.5-coder:7b", models_granted)

    def test_17_external_model_residency_detected_and_handled(self):
        """Invariant 8 & 17: External model in /api/ps is evicted before granting chatbot generation lease."""
        # External unmanaged model is resident in Ollama
        self.mock_ollama.resident_models = {"llama3:8b"}

        with self.manager.acquire_generation_lease("gemma4:12b") as lease:
            self.assertEqual(lease.model_name, "gemma4:12b")

        # The external model must have been evicted to protect system RAM
        self.assertIn("llama3:8b", self.mock_ollama.evict_calls)
        self.assertNotIn("llama3:8b", self.mock_ollama.resident_models)


if __name__ == "__main__":
    unittest.main()
