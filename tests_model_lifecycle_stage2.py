"""
Integration tests for Ollama Generation Leases in app.py (Stage 2).

Verifies that:
1. Single-stage generation routes (chat/gemma, deepseek, qwen) acquire and release their model lease.
2. Cloud route (gemini) completely bypasses local generation lease acquisition.
3. Multi-stage pipeline route (Architect -> Coder) releases the Architect lease before acquiring the Coder lease.
4. Pure architectural queries in pipeline mode do not acquire a Coder lease.
5. Client disconnection (generator.close()) immediately releases the active model lease without leaks.
6. Stream exceptions and timeouts cleanly release leases and emit structured SSE error events.
7. Model lease timeout errors are caught and surfaced via SSE without crashing the server.
8. Standard SSE event contract and schema remain identical.
"""

import contextlib
import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import requests

import app
from model_lifecycle import (
    ModelLeaseTimeoutError,
    ModelLifecycleError,
    OllamaLifecycleManager,
    get_model_manager,
    reset_model_manager_for_testing,
)


class MockOllamaServer:
    """Thread-safe simulated Ollama daemon for integration testing."""

    def __init__(self, resident_models=None):
        self._lock = threading.Lock()
        self.resident_models = set(resident_models or [])

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
                if m in self.resident_models:
                    self.resident_models.remove(m)
            return resp


class TestModelLifecycleStage2(unittest.TestCase):
    def setUp(self):
        self.app = app.app
        self.client = self.app.test_client()

        # Create isolated lifecycle manager for deterministic test verification
        self.mock_server = MockOllamaServer()
        self.manager = OllamaLifecycleManager(
            base_url="http://localhost:11434",
            max_concurrent_generation=1,
            default_timeout=2.0,
            poll_interval=0.02,
        )
        self.manager.session.get = self.mock_server.mock_get
        self.manager.session.post = self.mock_server.mock_post
        reset_model_manager_for_testing(self.manager)

    def tearDown(self):
        reset_model_manager_for_testing(None)

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

    def _make_mock_response(self, text: str, done_reason: str = "stop"):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = [
            json.dumps({"message": {"content": text}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": done_reason}),
        ]
        resp.__enter__.return_value = resp
        return resp

    @patch("app.http_requests.post")
    def test_chat_route_acquires_and_releases_lease(self, mock_post):
        """Single-stage Gemma chat acquires lease during streaming and releases it on completion."""
        mock_post.return_value = self._make_mock_response("Hello from Gemma 4!")

        lease_observed_active = []
        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def spy_acquire(target_model, **kwargs):
            with orig_acquire(target_model, **kwargs) as lease:
                lease_observed_active.append(lease.is_active)
                yield lease

        self.manager.acquire_generation_lease = spy_acquire

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]
        self.assertIn("route", types)
        self.assertIn("stage_start", types)
        self.assertIn("token", types)
        self.assertIn("done", types)

        # Lease was actively observed during streaming
        self.assertTrue(any(lease_observed_active))
        # After stream finished, no generation leases remain active
        reconciled = self.manager.reconcile_state()
        self.assertEqual(reconciled["active_gen_leases"], {})

    @patch("app.http_requests.post")
    def test_deepseek_route_acquires_and_releases_lease(self, mock_post):
        """Standalone DeepSeek route acquires lease for architect model and releases it."""
        mock_post.return_value = self._make_mock_response("Architecture design...")

        lease_models = []
        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def spy_acquire(target_model, **kwargs):
            lease_models.append(target_model)
            with orig_acquire(target_model, **kwargs) as lease:
                yield lease

        self.manager.acquire_generation_lease = spy_acquire

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Review this system design."}],
                "mode": "deepseek",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]
        self.assertIn("done", types)

        self.assertEqual(len(lease_models), 1)
        self.assertIn("deepseek", lease_models[0].lower())
        self.assertEqual(self.manager.reconcile_state()["active_gen_leases"], {})

    @patch("app.http_requests.post")
    def test_qwen_route_acquires_and_releases_lease(self, mock_post):
        """Standalone Qwen route acquires lease for coder model and releases it."""
        mock_post.return_value = self._make_mock_response("def hello(): return 'world'")

        lease_models = []
        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def spy_acquire(target_model, **kwargs):
            lease_models.append(target_model)
            with orig_acquire(target_model, **kwargs) as lease:
                yield lease

        self.manager.acquire_generation_lease = spy_acquire

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Write a python function."}],
                "mode": "qwen",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]
        self.assertIn("done", types)

        self.assertEqual(len(lease_models), 1)
        self.assertIn("qwen", lease_models[0].lower())
        self.assertEqual(self.manager.reconcile_state()["active_gen_leases"], {})

    @patch("app.get_gemini_api_key", return_value="fake-key")
    @patch("google.genai.Client")
    def test_gemini_route_bypasses_generation_lease(self, mock_genai_client, mock_key):
        """Cloud Gemini route must not acquire any local generation leases."""
        mock_instance = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Hello from Gemini cloud!"
        mock_instance.models.generate_content_stream.return_value = [mock_chunk]
        mock_genai_client.return_value = mock_instance

        acquire_called = []
        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def spy_acquire(target_model, **kwargs):
            acquire_called.append(target_model)
            with orig_acquire(target_model, **kwargs) as lease:
                yield lease

        self.manager.acquire_generation_lease = spy_acquire

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "What is the latest news today?"}],
                "mode": "gemini",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        done_event = next(e for e in events if e.get("type") == "done")
        self.assertEqual(done_event.get("mode"), "gemini")

        # Zero local generation leases acquired!
        self.assertEqual(len(acquire_called), 0)

    @patch("app.http_requests.post")
    def test_pipeline_sequential_leases_never_overlap(self, mock_post):
        """Pipeline route executes Architect then Coder; leases must never overlap."""
        mock_post.side_effect = [
            self._make_mock_response("Architect blueprint specification"),
            self._make_mock_response("Coder complete implementation code"),
        ]

        lifecycle_events = []
        active_leases = set()
        max_concurrent = 0
        lock = threading.Lock()

        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def spy_acquire(target_model, **kwargs):
            nonlocal max_concurrent
            with lock:
                active_leases.add(target_model)
                if len(active_leases) > max_concurrent:
                    max_concurrent = len(active_leases)
                lifecycle_events.append(("acquire", target_model, time.monotonic()))
            try:
                with orig_acquire(target_model, **kwargs) as lease:
                    yield lease
            finally:
                with lock:
                    active_leases.discard(target_model)
                    lifecycle_events.append(("release", target_model, time.monotonic()))

        self.manager.acquire_generation_lease = spy_acquire

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Design an auth service and write the python implementation.",
                    }
                ],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]
        self.assertIn("stage_start", types)
        self.assertIn("done", types)

        # Invariant 1: Concurrency never exceeded 1 model
        self.assertEqual(max_concurrent, 1)

        # Invariant 2: Two acquire events and two release events
        actions = [evt[0] for evt in lifecycle_events]
        self.assertEqual(actions, ["acquire", "release", "acquire", "release"])

        # Invariant 3: Architect release occurred before Coder acquire
        arch_release_time = lifecycle_events[1][2]
        coder_acquire_time = lifecycle_events[2][2]
        self.assertLessEqual(arch_release_time, coder_acquire_time)

        # Invariant 4: No active leases remain after pipeline finishes
        self.assertEqual(self.manager.reconcile_state()["active_gen_leases"], {})

    @patch("app.http_requests.post")
    def test_pipeline_pure_architectural_only_leases_architect(self, mock_post):
        """Pure architectural query in pipeline mode executes Stage 1 only and never acquires Coder lease."""
        mock_post.return_value = self._make_mock_response("System architecture assessment report")

        lease_models = []
        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def spy_acquire(target_model, **kwargs):
            lease_models.append(target_model)
            with orig_acquire(target_model, **kwargs) as lease:
                yield lease

        self.manager.acquire_generation_lease = spy_acquire

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Perform an architectural assessment of our message broker.",
                    }
                ],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("analytical_only"))

        # Only 1 lease acquired: Architect. Coder lease was never requested.
        self.assertEqual(len(lease_models), 1)
        self.assertIn("deepseek", lease_models[0].lower())
        self.assertEqual(self.manager.reconcile_state()["active_gen_leases"], {})

    @patch("app.http_requests.post")
    def test_client_disconnect_during_stream_releases_lease(self, mock_post):
        """When client disconnects (generator.close()), active lease is immediately released."""
        # Simulated generator yielding infinite tokens
        def infinite_token_stream():
            yield json.dumps({"message": {"content": "token1"}, "done": False})
            yield json.dumps({"message": {"content": "token2"}, "done": False})
            while True:
                yield json.dumps({"message": {"content": "..."}, "done": False})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = infinite_token_stream()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        # Invoke chat endpoint via streaming context
        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Stream data"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )

        # Advance until we receive token chunks (confirming generation lease is active)
        tokens_received = []
        for chunk in response.response:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            if '"token"' in chunk_str:
                tokens_received.append(chunk_str)
                break

        self.assertTrue(len(tokens_received) > 0)

        # Confirm lease is active during streaming
        active = self.manager.reconcile_state()["active_gen_leases"]
        self.assertTrue(len(active) > 0)

        # Simulate client closing the stream (e.g. user aborts or navigates away)
        response.close()

        # Confirm lease is released immediately
        active_after = self.manager.reconcile_state()["active_gen_leases"]
        self.assertEqual(active_after, {})

    @patch("app.http_requests.post")
    def test_exception_during_streaming_releases_lease(self, mock_post):
        """When an exception occurs during streaming, lease is released and error event is emitted."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        def fail_mid_stream():
            yield json.dumps({"message": {"content": "start"}, "done": False})
            raise requests.exceptions.Timeout("Read timed out mid-generation")

        mock_resp.iter_lines.return_value = fail_mid_stream()
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Explain sorting"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]
        self.assertIn("error", types)
        self.assertIn("stage_end", types)
        self.assertIn("done", types)

        err_event = next(e for e in events if e.get("type") == "error")
        self.assertIn("timed out", err_event.get("error", "").lower())

        # Lease must be completely released
        self.assertEqual(self.manager.reconcile_state()["active_gen_leases"], {})

    def test_lease_timeout_emits_error_and_terminates(self):
        """When acquire_generation_lease times out, stream emits error and done events cleanly."""
        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def failing_acquire(target_model, **kwargs):
            raise ModelLeaseTimeoutError("Timed out waiting to acquire lease")
            yield  # pragma: no cover

        self.manager.acquire_generation_lease = failing_acquire

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]
        self.assertIn("error", types)
        self.assertIn("done", types)

        err_event = next(e for e in events if e.get("type") == "error")
        self.assertIn("timed out waiting for model resources", err_event.get("error", "").lower())

        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("done"))
        self.assertIsNotNone(done_event.get("error"))

    @patch("app.http_requests.post")
    def test_pipeline_truncation_releases_architect_lease_and_prevents_coder(self, mock_post):
        """When architect is truncated at length limit, lease is released and coder lease is never requested."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "Incomplete architect outline..."}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length"}),
        ]
        mock_resp.__enter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        lease_models = []
        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def spy_acquire(target_model, **kwargs):
            lease_models.append(target_model)
            with orig_acquire(target_model, **kwargs) as lease:
                yield lease

        self.manager.acquire_generation_lease = spy_acquire

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Design an auth service and write the python implementation.",
                    }
                ],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]
        self.assertIn("error", types)
        self.assertIn("done", types)

        err_event = next(e for e in events if e.get("type") == "error")
        self.assertTrue(err_event.get("truncated"))

        # Architect was requested, but Coder was NEVER requested due to truncation
        self.assertEqual(len(lease_models), 1)
        self.assertIn("deepseek", lease_models[0].lower())
        # All leases released
        self.assertEqual(self.manager.reconcile_state()["active_gen_leases"], {})

    @patch("app.http_requests.post")
    def test_concurrent_different_model_requests_serialize(self, mock_post):
        """Concurrent requests for distinct models serialize: model 2 waits for model 1 to release."""
        def make_stream(tokens, delay=0.1):
            def _gen():
                for t in tokens:
                    time.sleep(delay)
                    yield json.dumps({"message": {"content": t}, "done": False})
                yield json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"})
            return _gen()

        # Two distinct model streams
        resp1 = MagicMock()
        resp1.raise_for_status = MagicMock()
        resp1.iter_lines.return_value = make_stream(["gemma1", "gemma2"], delay=0.08)
        resp1.__enter__.return_value = resp1

        resp2 = MagicMock()
        resp2.raise_for_status = MagicMock()
        resp2.iter_lines.return_value = make_stream(["qwen1", "qwen2"], delay=0.08)
        resp2.__enter__.return_value = resp2

        mock_post.side_effect = [resp1, resp2]

        max_concurrent_models = 0
        active_models = set()
        lock = threading.Lock()
        orig_acquire = self.manager.acquire_generation_lease

        @contextlib.contextmanager
        def concurrency_spy(target_model, **kwargs):
            nonlocal max_concurrent_models
            with orig_acquire(target_model, **kwargs) as lease:
                with lock:
                    active_models.add(target_model)
                    if len(active_models) > max_concurrent_models:
                        max_concurrent_models = len(active_models)
                try:
                    yield lease
                finally:
                    with lock:
                        active_models.discard(target_model)

        self.manager.acquire_generation_lease = concurrency_spy

        results = {}

        def send_req1():
            client = self.app.test_client()
            res = client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "hi"}], "mode": "chat", "rag_mode": "off"},
            )
            results["req1"] = res.get_data(as_text=True)

        def send_req2():
            # Small delay so req1 arrives first and acquires gemma
            time.sleep(0.04)
            client = self.app.test_client()
            res = client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "write code"}], "mode": "qwen", "rag_mode": "off"},
            )
            results["req2"] = res.get_data(as_text=True)

        t1 = threading.Thread(target=send_req1)
        t2 = threading.Thread(target=send_req2)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        self.assertIn("req1", results)
        self.assertIn("req2", results)
        # Guarantee: At no point was more than 1 model leased concurrently
        self.assertEqual(max_concurrent_models, 1)
        self.assertEqual(self.manager.reconcile_state()["active_gen_leases"], {})


if __name__ == "__main__":
    unittest.main()
