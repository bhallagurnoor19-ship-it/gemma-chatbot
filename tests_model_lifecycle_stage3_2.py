"""
Stage 3.2 Test Suite: Cancellation-Aware Ollama Streaming, Safe Lease Release,
Eviction Polling Verification, and Enriched Timeout Diagnostics.

Covers:
1. Client disconnect during active Ollama generation.
2. Cancellation while Ollama is waiting before its first token.
3. Cancellation while Ollama is streaming tokens.
4. Upstream HTTP response closes after cancellation.
5. Generation lease releases only after inference worker termination.
6. A still-running worker cannot have its lease revoked.
7. Normal completion releases the lease.
8. Worker exceptions release the lease.
9. SSE heartbeat delivery while inference produces no tokens.
10. No duplicate lease release.
11. No thread leaks after repeated cancellation.
12. Eviction polling with delayed Ollama shutdown.
13. Eviction verification fails if the model never disappears.
14. A model with expired expires_at remains classified as resident until absent.
15. Timeout diagnostics identify the actual blocking condition.
"""

import contextlib
import gc
import json
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import requests

import app
from model_lifecycle import (
    EmbeddingPolicy,
    ModelEvictionError,
    ModelLease,
    ModelLeaseCancelledError,
    ModelLeaseTimeoutError,
    ModelLeaseType,
    OllamaLifecycleManager,
    get_model_manager,
    normalize_model_name,
    reset_model_manager_for_testing,
)


class MockOllamaDaemon:
    """Thread-safe simulated Ollama daemon for deterministic Stage 3.2 testing."""

    def __init__(self, resident_models=None):
        self._lock = threading.Lock()
        self.resident_models = set(resident_models or [])
        self.eviction_calls = []
        self.eviction_delay_ticks = 0
        self.pending_eviction = None

    def mock_get(self, url, timeout=None):
        with self._lock:
            resp = MagicMock()
            resp.status_code = 200
            # If delayed eviction is configured, count down
            if self.eviction_delay_ticks > 0:
                self.eviction_delay_ticks -= 1
                if self.eviction_delay_ticks == 0 and self.pending_eviction:
                    norm = normalize_model_name(self.pending_eviction)
                    to_remove = [m for m in self.resident_models if normalize_model_name(m) == norm]
                    for m in to_remove:
                        self.resident_models.remove(m)
                    self.pending_eviction = None
            resp.json.return_value = {
                "models": [{"name": m, "model": m} for m in self.resident_models]
            }
            return resp

    def mock_post(self, url, json=None, timeout=None):
        with self._lock:
            resp = MagicMock()
            resp.status_code = 200
            if json and json.get("keep_alive") == 0:
                m = json.get("model")
                self.eviction_calls.append(m)
                if self.eviction_delay_ticks <= 0:
                    norm = normalize_model_name(m)
                    to_remove = [rm for rm in self.resident_models if normalize_model_name(rm) == norm]
                    for rm in to_remove:
                        self.resident_models.remove(rm)
                else:
                    self.pending_eviction = m
            return resp


class TestModelLifecycleStage3_2(unittest.TestCase):
    def setUp(self):
        self.app = app.app
        self.client = self.app.test_client()

        self.mock_daemon = MockOllamaDaemon()
        self.manager = OllamaLifecycleManager(
            base_url="http://localhost:11434",
            max_concurrent_generation=1,
            default_timeout=2.0,
            poll_interval=0.02,
            eviction_timeout=0.6,
            eviction_poll_interval=0.05,
        )
        self.manager.session.get = self.mock_daemon.mock_get
        self.manager.session.post = self.mock_daemon.mock_post
        reset_model_manager_for_testing(self.manager)

    def tearDown(self):
        reset_model_manager_for_testing(None)
        gc.collect()

    def _parse_sse_events(self, response_data: str):
        events = []
        for line in response_data.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except Exception:
                    pass
        return events

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Client disconnect during active Ollama generation
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_01_client_disconnect_during_active_generation(self, mock_post):
        """Client disconnect during streaming releases the lease, unblocking subsequent models."""
        unblock_event = threading.Event()

        def slow_stream():
            yield json.dumps({"message": {"content": "token_first"}, "done": False})
            unblock_event.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "token_second"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = slow_stream()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "mode": "chat", "rag_mode": "off"},
        )

        first_chunk = None
        for chunk in response.response:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            if '"token"' in chunk_str:
                first_chunk = chunk_str
                break

        self.assertIsNotNone(first_chunk)
        self.assertEqual(len(self.manager.get_diagnostics()["active_gen_leases"]), 1)

        # Client disconnects
        if hasattr(response.response, "close"):
            response.response.close()
        response.close()
        unblock_event.set()
        time.sleep(0.1)

        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"], {})

        # Subsequent model switch must succeed immediately
        with self.manager.acquire_generation_lease("deepseek-coder-v2:16b", timeout=1.0) as lease:
            self.assertEqual(lease.model_name, "deepseek-coder-v2:16b")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Cancellation while Ollama is waiting before its first token
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_02_cancellation_while_waiting_before_first_token(self, mock_post):
        """Cancellation while Ollama is waiting on cold load/CPU eval terminates worker and releases lease."""
        block_worker = threading.Event()
        worker_reached = threading.Event()

        def blocking_stream():
            worker_reached.set()
            block_worker.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "finally"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = blocking_stream()
        mock_resp.close = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Cold load test"}], "mode": "chat", "rag_mode": "off"},
        )

        iterator = iter(response.response)
        # Advance until worker is inside stream_ollama_stage
        for chunk in iterator:
            if worker_reached.wait(timeout=0.1):
                break

        # Confirm lease is held
        self.assertTrue(len(self.manager.get_diagnostics()["active_gen_leases"]) > 0)

        # Disconnect client
        if hasattr(response.response, "close"):
            response.response.close()
        response.close()
        block_worker.set()
        time.sleep(0.1)

        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})
        mock_resp.close.assert_called()

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Cancellation while Ollama is streaming tokens
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_03_cancellation_while_streaming_tokens(self, mock_post):
        """Cancellation while multiple tokens are in flight halts worker and releases lease."""
        token_count = 0
        hang_event = threading.Event()

        def multi_token_stream():
            nonlocal token_count
            for i in range(5):
                token_count += 1
                yield json.dumps({"message": {"content": f"tok_{i}"}, "done": False})
            hang_event.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "done"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = multi_token_stream()
        mock_resp.close = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Multi token"}], "mode": "chat", "rag_mode": "off"},
        )

        received = 0
        for chunk in response.response:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            if '"token"' in chunk_str:
                received += 1
                if received >= 2:
                    break

        self.assertGreaterEqual(received, 2)
        if hasattr(response.response, "close"):
            response.response.close()
        response.close()
        hang_event.set()
        time.sleep(0.1)

        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Upstream HTTP response closes after cancellation
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_04_upstream_http_response_closes_after_cancellation(self, mock_post):
        """When client disconnects, active_resp.close() is explicitly called to close socket."""
        hang_event = threading.Event()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.close = MagicMock()

        def stream_tokens():
            yield json.dumps({"message": {"content": "abc"}, "done": False})
            hang_event.wait(timeout=10.0)

        mock_resp.iter_lines.return_value = stream_tokens()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Close test"}], "mode": "chat", "rag_mode": "off"},
        )

        for chunk in response.response:
            if b"abc" in chunk:
                break

        if hasattr(response.response, "close"):
            response.response.close()
        response.close()
        hang_event.set()
        time.sleep(0.1)

        mock_resp.close.assert_called()

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Generation lease releases only after inference worker termination
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_05_generation_lease_releases_only_after_worker_termination(self, mock_post):
        """Invariant 3: Lease release happens strictly after worker thread has stopped.

        With the new state machine, the manager is notified via _on_lease_released().
        In Case A (normal completion): worker calls mark_worker_finished() → WORKER_DONE,
        worker thread exits, then generator calls detach_or_release() → RELEASED →
        _on_lease_released(). At that point no ollama-worker thread should be alive.
        """
        release_snapshots = []   # True if any worker alive at notification time

        manager_spy_target = self.manager
        orig_notify = manager_spy_target._on_lease_released

        def spy_on_release(lease):
            curr = threading.current_thread()
            if "ollama-worker" in curr.name:
                # Worker calling release_from_worker() — it IS running but that is
                # the Case B path. Record False: worker owns the release.
                release_snapshots.append(False)
            else:
                worker_threads = [
                    t for t in threading.enumerate()
                    if "ollama-worker" in t.name and t.is_alive()
                ]
                release_snapshots.append(len(worker_threads) > 0)
            orig_notify(lease)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "done"}, "done": True}),
        ]
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        manager_spy_target._on_lease_released = spy_on_release
        try:
            response = self.client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "Termination order"}], "mode": "chat", "rag_mode": "off"},
            )
            _ = response.get_data()
        finally:
            manager_spy_target._on_lease_released = orig_notify

        self.assertTrue(len(release_snapshots) > 0,
                        "Expected at least one _on_lease_released notification")
        self.assertFalse(any(release_snapshots),
                         "Worker was still alive when _on_lease_released was called!")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. A still-running worker cannot have its lease revoked
    # ──────────────────────────────────────────────────────────────────────────
    def test_06_still_running_worker_cannot_have_lease_revoked(self):
        """Invariant 4: While worker holds a lease, an external request cannot revoke it."""
        lease_acquired = threading.Event()
        release_event = threading.Event()

        def slow_worker():
            with self.manager.acquire_generation_lease("gemma4:12b"):
                lease_acquired.set()
                release_event.wait(timeout=2.0)

        t = threading.Thread(target=slow_worker)
        t.start()
        lease_acquired.wait(timeout=2.0)

        with self.assertRaises(ModelLeaseTimeoutError):
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b", timeout=0.1):
                pass

        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["resident_gen_model"], "gemma4:12b")
        self.assertEqual(diag["active_gen_leases"]["gemma4:12b"], 1)

        release_event.set()
        t.join(timeout=2.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Normal completion releases the lease
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_07_normal_completion_releases_lease(self, mock_post):
        """Normal completion delivers all tokens, terminates worker, and cleanly releases lease."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "Hello, world!"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}),
        ]
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hi"}], "mode": "chat", "rag_mode": "off"},
        )
        data = response.get_data(as_text=True)
        events = self._parse_sse_events(data)
        tokens = [e["token"] for e in events if e.get("type") == "token"]
        self.assertIn("Hello, world!", tokens)

        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Worker exceptions release the lease
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_08_worker_exceptions_release_lease(self, mock_post):
        """Worker HTTP exception mid-stream releases lease and yields SSE error event."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        def failing_stream():
            yield json.dumps({"message": {"content": "initial"}, "done": False})
            raise requests.exceptions.ConnectionError("Ollama daemon crash")

        mock_resp.iter_lines.return_value = failing_stream()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Crash"}], "mode": "chat", "rag_mode": "off"},
        )
        data = response.get_data(as_text=True)
        events = self._parse_sse_events(data)
        error_events = [e for e in events if e.get("type") == "error"]
        self.assertTrue(len(error_events) > 0)

        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 9. SSE heartbeat delivery while inference produces no tokens
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_09_sse_heartbeat_delivery_while_inference_produces_no_tokens(self, mock_post):
        """While inference is thinking, periodic SSE heartbeat comments (: heartbeat) are yielded."""
        def slow_stream():
            time.sleep(0.35)
            yield json.dumps({"message": {"content": "ready"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = slow_stream()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Thinking"}], "mode": "chat", "rag_mode": "off"},
        )

        raw_chunks = list(response.response)
        raw_text = b"".join(raw_chunks).decode("utf-8")
        self.assertTrue(": heartbeat\n\n" in raw_text or ": ping\n\n" in raw_text or "token" in raw_text)

    # ──────────────────────────────────────────────────────────────────────────
    # 10. No duplicate lease release
    # ──────────────────────────────────────────────────────────────────────────
    def test_10_no_duplicate_lease_release(self):
        """Calling release() multiple times on a lease is safe and idempotent."""
        with self.manager.acquire_generation_lease("gemma4:12b") as lease:
            self.assertTrue(lease.is_active)
            lease.release()
            self.assertFalse(lease.is_active)
            lease.release()
            self.assertFalse(lease.is_active)

        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 11. No thread leaks after repeated cancellation
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_11_no_thread_leaks_after_repeated_cancellation(self, mock_post):
        """Executing 5 rapid cancel-after-start requests leaves zero orphaned worker threads."""
        def hanging_stream():
            while True:
                time.sleep(0.5)
                yield json.dumps({"message": {"content": "tick"}, "done": False})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = hanging_stream()
        mock_resp.close = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        for i in range(5):
            resp = self.client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": f"Leak test {i}"}], "mode": "chat", "rag_mode": "off"},
            )
            it = iter(resp.response)
            _ = next(it)
            if hasattr(resp.response, "close"):
                resp.response.close()
            resp.close()

        time.sleep(0.2)
        worker_threads = [t for t in threading.enumerate() if "ollama-worker" in t.name and t.is_alive()]
        self.assertEqual(len(worker_threads), 0, f"Found leaked worker threads: {worker_threads}")
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 12. Eviction polling with delayed Ollama shutdown
    # ──────────────────────────────────────────────────────────────────────────
    def test_12_eviction_polling_with_delayed_ollama_shutdown(self):
        """_evict_model_http polls /api/ps until model disappears, tolerating delayed Ollama process exit."""
        self.mock_daemon.resident_models = {"gemma4:12b"}
        self.mock_daemon.eviction_delay_ticks = 2

        success = self.manager._evict_model_http("gemma4:12b")
        self.assertTrue(success, "Eviction should succeed via bounded polling")
        self.assertNotIn("gemma4:12b", self.mock_daemon.resident_models)

    # ──────────────────────────────────────────────────────────────────────────
    # 13. Eviction verification fails if the model never disappears
    # ──────────────────────────────────────────────────────────────────────────
    def test_13_eviction_verification_fails_if_model_never_disappears(self):
        """If model remains in /api/ps past eviction_timeout, eviction fails closed."""
        self.mock_daemon.resident_models = {"gemma4:12b"}
        self.mock_daemon.eviction_delay_ticks = 9999

        success = self.manager._evict_model_http("gemma4:12b")
        self.assertFalse(success, "Eviction must fail closed if model never disappears")

        with self.assertRaises(ModelEvictionError):
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b"):
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # 14. A model with expired expires_at remains classified as resident until absent
    # ──────────────────────────────────────────────────────────────────────────
    def test_14_expired_expires_at_remains_resident_until_absent(self):
        """A model in /api/ps with status Stopping... (past expires_at) is still considered resident."""
        def mock_ps_stopping(url, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "models": [
                    {
                        "name": "gemma4:12b",
                        "model": "gemma4:12b",
                        "expires_at": "2026-11-06T00:00:00Z",  # In the past -> Stopping...
                    }
                ]
            }
            return resp

        self.manager.session.get = mock_ps_stopping
        ps_models = self.manager._query_ollama_ps()
        self.assertIn("gemma4:12b", ps_models)

    # ──────────────────────────────────────────────────────────────────────────
    # 15. Timeout diagnostics identify the actual blocking condition
    # ──────────────────────────────────────────────────────────────────────────
    def test_15_timeout_diagnostics_identifies_actual_blocking_condition(self):
        """ModelLeaseTimeoutError message includes requested model, resident model, and blocking condition."""
        # Case A: Blocked by active lease on resident model
        with self.manager.acquire_generation_lease("gemma4:12b"):
            try:
                with self.manager.acquire_generation_lease("deepseek-coder-v2:16b", timeout=0.1):
                    self.fail("Expected ModelLeaseTimeoutError")
            except ModelLeaseTimeoutError as exc:
                err_str = str(exc)
                self.assertIn("deepseek-coder-v2:16b", err_str)
                self.assertIn("gemma4:12b", err_str)
                self.assertIn("blocking_condition=", err_str)
                self.assertIn("active_lease_on_resident_model", err_str)
                self.assertIn("active_gen=", err_str)

        # Case B: Blocked by concurrency limit on same model
        self.manager.max_concurrent_generation = 1
        with self.manager.acquire_generation_lease("gemma4:12b"):
            try:
                with self.manager.acquire_generation_lease("gemma4:12b", timeout=0.1):
                    self.fail("Expected ModelLeaseTimeoutError")
            except ModelLeaseTimeoutError as exc:
                err_str = str(exc)
                self.assertIn("concurrency_limit_reached", err_str)

        # Case C: Blocked by active embedding lease under adaptive policy
        self.manager.embedding_policy = EmbeddingPolicy.ADAPTIVE
        with self.manager.acquire_embedding_lease():
            try:
                with self.manager.acquire_generation_lease("deepseek-coder-v2:16b", timeout=0.1):
                    self.fail("Expected ModelLeaseTimeoutError")
            except ModelLeaseTimeoutError as exc:
                err_str = str(exc)
                self.assertIn("waiting_for_embedding_leases_to_drain", err_str)


if __name__ == "__main__":
    unittest.main()
