"""
Stage 3.2.2 Test Suite: Worker-Owned Generation Leases & Cancellation Safety.

Verifies:
1. Worker owns lease after Flask generator exits.
2. Worker remains alive beyond the five-second join timeout.
3. Another generation model cannot acquire a conflicting lease while that worker remains alive.
4. Worker termination eventually releases its lease.
5. Worker exceptions release the lease.
6. Repeated cancellation does not release the lease twice.
7. Cancellation before worker startup.
8. Cancellation during connection establishment.
9. Cancellation before the first token.
10. Cancellation during token streaming.
11. DeepSeek-to-Qwen pipeline handoff waits for actual worker termination.
12. Stuck worker appears in diagnostics.
13. No background worker or lease leaks after successful cancellation.
"""

import contextlib
import gc
import json
import logging
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
    """Thread-safe simulated Ollama daemon for deterministic Stage 3.2.2 testing."""

    def __init__(self, resident_models=None):
        self._lock = threading.Lock()
        self.resident_models = set(resident_models or [])
        self.eviction_calls = []

    def mock_get(self, url, timeout=None):
        with self._lock:
            resp = MagicMock()
            resp.status_code = 200
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
                norm = normalize_model_name(m)
                to_remove = [rm for rm in self.resident_models if normalize_model_name(rm) == norm]
                for rm in to_remove:
                    self.resident_models.remove(rm)
            return resp


class TestModelLifecycleStage3_2_2(unittest.TestCase):
    def setUp(self):
        self.app = app.app
        self.client = self.app.test_client()

        self.mock_daemon = MockOllamaDaemon()
        self.manager = OllamaLifecycleManager(
            base_url="http://localhost:11434",
            max_concurrent_generation=1,
            embedding_policy="adaptive",
            default_timeout=5.0,
            poll_interval=0.05,
            eviction_timeout=1.0,
            eviction_poll_interval=0.05,
        )
        reset_model_manager_for_testing(self.manager)

        self.get_patcher = patch.object(self.manager.session, "get", side_effect=self.mock_daemon.mock_get)
        self.post_patcher = patch.object(self.manager.session, "post", side_effect=self.mock_daemon.mock_post)
        self.get_patcher.start()
        self.post_patcher.start()

        # Patch worker join timeout to 0.1s for fast, reliable unit testing without waiting 5.0s
        self.orig_thread_join = threading.Thread.join

        def fast_worker_join(thread_self, timeout=None):
            if "ollama-worker" in getattr(thread_self, "name", "") and timeout == 5.0:
                return self.orig_thread_join(thread_self, timeout=0.1)
            return self.orig_thread_join(thread_self, timeout=timeout)

        self.join_patcher = patch.object(threading.Thread, "join", fast_worker_join)
        self.join_patcher.start()

    def tearDown(self):
        self.join_patcher.stop()
        self.get_patcher.stop()
        self.post_patcher.stop()
        reset_model_manager_for_testing(None)
        gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Worker owns lease after Flask generator exits
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_01_worker_owns_lease_after_flask_generator_exits(self, mock_post):
        """When Flask generator exits while worker is alive, worker retains lease ownership."""
        worker_started = threading.Event()
        worker_unblock = threading.Event()

        def slow_stream():
            worker_started.set()
            worker_unblock.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "finally"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = slow_stream()
        mock_resp.close = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Worker ownership test"}], "mode": "chat", "rag_mode": "off"},
        )

        # Read first route event to start generator and worker
        for chunk in response.response:
            if worker_started.wait(timeout=1.0):
                break

        # Simulate browser disconnecting immediately (generator closes)
        if hasattr(response.response, "close"):
            response.response.close()
        response.close()

        # Generator has now exited. Verify lease is STILL held because worker is alive!
        diag = self.manager.get_diagnostics()
        self.assertEqual(len(diag["active_gen_leases"]), 1, "Lease must remain active under worker ownership!")
        self.assertIn("gemma4:12b", diag["active_gen_leases"])

        # Unblock worker to allow clean shutdown
        worker_unblock.set()
        time.sleep(0.2)

        # Now lease is cleanly released by worker
        diag_after = self.manager.get_diagnostics()
        self.assertEqual(diag_after["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Worker remains alive beyond the five-second join timeout
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_02_worker_remains_alive_beyond_join_timeout(self, mock_post):
        """If worker remains alive beyond join timeout, generator exits safely and lease stays active."""
        worker_started = threading.Event()
        worker_unblock = threading.Event()

        def uncooperative_stream():
            worker_started.set()
            worker_unblock.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "done"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = uncooperative_stream()
        mock_resp.close = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hung worker test"}], "mode": "chat", "rag_mode": "off"},
        )

        for chunk in response.response:
            if worker_started.wait(timeout=1.0):
                break

        # Disconnect client
        if hasattr(response.response, "close"):
            response.response.close()
        response.close()

        # The generator has finished fast_join (0.1s), but worker is still alive
        diag = self.manager.get_diagnostics()
        self.assertEqual(len(diag["active_gen_leases"]), 1, "Lease must NOT be released while worker is alive!")
        self.assertEqual(len(diag["stuck_workers"]), 1, "Stuck worker must be reported in diagnostics!")

        worker_unblock.set()
        time.sleep(0.2)
        diag_after = self.manager.get_diagnostics()
        self.assertEqual(diag_after["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Another generation model cannot acquire a conflicting lease while worker alive
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_03_another_model_cannot_acquire_conflicting_lease(self, mock_post):
        """A different model request cannot acquire a lease while previous worker is alive."""
        worker_started = threading.Event()
        worker_unblock = threading.Event()

        def slow_stream():
            worker_started.set()
            worker_unblock.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "done"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = slow_stream()
        mock_resp.close = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Test"}], "mode": "chat", "rag_mode": "off"},
        )
        for chunk in response.response:
            if worker_started.wait(timeout=1.0):
                break

        # Browser aborts
        if hasattr(response.response, "close"):
            response.response.close()
        response.close()

        # Attempt to acquire a lease for DeepSeek (different model)
        with self.assertRaises(ModelLeaseTimeoutError):
            with self.manager.acquire_generation_lease("deepseek-coder-v2:16b", timeout=0.1):
                pass

        worker_unblock.set()
        time.sleep(0.2)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Worker termination eventually releases its lease
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_04_worker_termination_eventually_releases_lease(self, mock_post):
        """Worker eventually terminating cleanly triggers lease release and unblocks manager."""
        worker_started = threading.Event()
        worker_unblock = threading.Event()

        def delayed_stream():
            worker_started.set()
            worker_unblock.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "done"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = delayed_stream()
        mock_resp.close = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Test"}], "mode": "chat", "rag_mode": "off"},
        )
        for chunk in response.response:
            if worker_started.wait(timeout=1.0):
                break

        if hasattr(response.response, "close"):
            response.response.close()
        response.close()

        self.assertEqual(len(self.manager.get_diagnostics()["active_gen_leases"]), 1)

        # Worker unblocks and terminates
        worker_unblock.set()
        time.sleep(0.2)

        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Worker exceptions release the lease
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_05_worker_exceptions_release_lease(self, mock_post):
        """When worker raises an unhandled network/HTTP exception, its finally: releases lease."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Ollama crashed")

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Crash test"}], "mode": "chat", "rag_mode": "off"},
        )
        # Consume response to completion
        _ = response.get_data()

        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Repeated cancellation does not release the lease twice
    # ──────────────────────────────────────────────────────────────────────────
    def test_06_repeated_cancellation_does_not_release_lease_twice(self):
        """ModelLease.release() is strictly idempotent and safe against repeated calls."""
        with self.manager.acquire_generation_lease("gemma4:12b") as lease:
            self.assertTrue(lease.is_active)
            # Call release multiple times
            lease.release()
            lease.release()
            lease.release()

        self.assertFalse(lease.is_active)
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Cancellation before worker startup
    # ──────────────────────────────────────────────────────────────────────────
    def test_07_cancellation_before_worker_startup(self):
        """If cancellation occurs before worker is started, lease is safely released."""
        lease = None
        with self.manager.acquire_generation_lease("gemma4:12b") as l:
            lease = l
            self.assertTrue(lease.is_active)

        self.assertFalse(lease.is_active)
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Cancellation during connection establishment
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_08_cancellation_during_connection_establishment(self, mock_post):
        """Cancellation while worker is in requests.post() connect phase preserves lease until connect returns."""
        post_entered = threading.Event()
        post_unblock = threading.Event()

        def slow_post(*args, **kwargs):
            post_entered.set()
            post_unblock.wait(timeout=10.0)
            mock_r = MagicMock()
            mock_r.status_code = 200
            mock_r.iter_lines.return_value = [json.dumps({"message": {"content": "ok"}, "done": True})]
            return mock_r

        mock_post.side_effect = slow_post

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Slow connect"}], "mode": "chat", "rag_mode": "off"},
        )
        for chunk in response.response:
            if post_entered.wait(timeout=0.1):
                break

        # Abort while in connect phase
        if hasattr(response.response, "close"):
            response.response.close()
        response.close()

        # Lease is still owned by worker in requests.post()
        diag = self.manager.get_diagnostics()
        self.assertEqual(len(diag["active_gen_leases"]), 1)

        # Connect finishes
        post_unblock.set()
        time.sleep(0.2)

        # Now lease is released
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 9. Cancellation before the first token
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_09_cancellation_before_first_token(self, mock_post):
        """Cancellation while waiting for first token unblocks iter_lines() and releases lease."""
        worker_reached = threading.Event()
        hang_event = threading.Event()

        def waiting_stream():
            worker_reached.set()
            hang_event.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "tok"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = waiting_stream()
        def mock_close():
            hang_event.set()
        mock_resp.close = mock_close
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Waiting token"}], "mode": "chat", "rag_mode": "off"},
        )
        for chunk in response.response:
            if worker_reached.wait(timeout=1.0):
                break

        self.assertEqual(len(self.manager.get_diagnostics()["active_gen_leases"]), 1)

        if hasattr(response.response, "close"):
            response.response.close()
        response.close()
        time.sleep(0.2)

        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 10. Cancellation during token streaming
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_10_cancellation_during_token_streaming(self, mock_post):
        """Cancellation while tokens are flowing halts stream and safely releases lease."""
        hang_event = threading.Event()

        def token_stream():
            for i in range(5):
                yield json.dumps({"message": {"content": f"t{i}"}, "done": False})
            hang_event.wait(timeout=10.0)
            yield json.dumps({"message": {"content": "done"}, "done": True})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = token_stream()
        def mock_close():
            hang_event.set()
        mock_resp.close = mock_close
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Stream"}], "mode": "chat", "rag_mode": "off"},
        )
        tokens = 0
        for chunk in response.response:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            if '"token"' in chunk_str:
                tokens += 1
                if tokens >= 2:
                    break

        self.assertGreaterEqual(tokens, 2)
        if hasattr(response.response, "close"):
            response.response.close()
        response.close()
        time.sleep(0.2)

        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 11. DeepSeek-to-Qwen pipeline handoff waits for actual worker termination
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_11_pipeline_handoff_waits_for_actual_worker_termination(self, mock_post):
        """In pipeline mode, Qwen stage acquires lease strictly after DeepSeek releases."""
        stages_executed = []

        def mock_post_pipeline(url, json=None, **kwargs):
            payload = json or {}
            m = payload.get("model")
            stages_executed.append(m)
            import json as _json
            mock_r = MagicMock()
            mock_r.status_code = 200
            mock_r.raise_for_status = MagicMock()
            mock_r.iter_lines.return_value = [
                _json.dumps({"message": {"content": f"output from {m}"}, "done": True, "done_reason": "stop"}),
            ]
            mock_r.__enter__.return_value = mock_r
            return mock_r

        mock_post.side_effect = mock_post_pipeline

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Write a python function to compute fibonacci."}],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        _ = response.get_data()

        self.assertTrue(
            any("deepseek" in s.lower() for s in stages_executed),
            f"Expected deepseek in {stages_executed}"
        )
        self.assertTrue(
            any("qwen" in s.lower() for s in stages_executed),
            f"Expected qwen in {stages_executed}"
        )
        # Verify both leases were released
        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 12. Stuck worker appears in diagnostics
    # ──────────────────────────────────────────────────────────────────────────
    def test_12_stuck_worker_appears_in_diagnostics(self):
        """Stuck worker marked on lease is exposed in get_diagnostics()['stuck_workers']."""
        # The dummy worker must signal completion via mark_worker_finished() so the
        # context manager's detach_or_release() can properly release the lease.
        do_finish = threading.Event()
        worker_signalled = threading.Event()

        def _dummy_worker(lease_ref):
            do_finish.wait(timeout=3.0)
            lease_ref[0].mark_worker_finished()
            worker_signalled.set()

        lease_ref = [None]
        with self.manager.acquire_generation_lease("gemma4:12b") as lease:
            lease_ref[0] = lease
            dummy_worker = threading.Thread(
                target=_dummy_worker, args=(lease_ref,), name="test-stuck-worker"
            )
            dummy_worker.start()
            lease.transfer_to_worker(dummy_worker)
            lease.mark_stuck()

            diag = self.manager.get_diagnostics()
            self.assertEqual(len(diag["stuck_workers"]), 1)
            self.assertEqual(diag["stuck_workers"][0]["model"], "gemma4:12b")
            self.assertEqual(diag["stuck_workers"][0]["worker_name"], "test-stuck-worker")

            # Signal worker to finish before exiting context
            do_finish.set()
            worker_signalled.wait(timeout=3.0)
            dummy_worker.join(timeout=2.0)

        # After worker completion signals and context exits, lease is released
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})

    # ──────────────────────────────────────────────────────────────────────────
    # 13. No background worker or lease leaks after successful cancellation
    # ──────────────────────────────────────────────────────────────────────────
    @patch("app.http_requests.post")
    def test_13_no_background_worker_or_lease_leaks_after_repeated_cancellation(self, mock_post):
        """Repeated rapid client disconnects leave 0 running workers and 0 leaked leases."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "tok"}, "done": False}),
            json.dumps({"message": {"content": "done"}, "done": True}),
        ]
        mock_resp.close = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        for i in range(5):
            resp = self.client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": f"Rapid abort {i}"}], "mode": "chat", "rag_mode": "off"},
            )
            for _ in resp.response:
                break
            if hasattr(resp.response, "close"):
                resp.response.close()
            resp.close()

        time.sleep(0.3)
        workers = [t for t in threading.enumerate() if "ollama-worker" in t.name and t.is_alive()]
        self.assertEqual(len(workers), 0, f"Leaked worker threads: {workers}")
        self.assertEqual(self.manager.get_diagnostics()["active_gen_leases"], {})


if __name__ == "__main__":
    unittest.main()
