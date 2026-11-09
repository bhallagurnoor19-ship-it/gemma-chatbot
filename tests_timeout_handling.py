"""
Unit and regression tests for Ollama timeout handling and SSE stream termination.
Verifies that:
1. Ollama timeout constants are defined with fast connect timeout and adequate read timeout.
2. Backend stream terminates cleanly with structured error, stage_end (error=True), and done events.
3. Errors in single-stage (chat, deepseek, qwen) and multi-stage (pipeline) routes are isolated and handled gracefully.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
import requests

import app


class TestTimeoutHandling(unittest.TestCase):
    def setUp(self):
        self.app = app.app
        self.client = self.app.test_client()

    def test_timeout_constants_configured(self):
        """Verify Ollama timeouts are configured with fast connect and adequate read timeouts."""
        self.assertTrue(hasattr(app, "OLLAMA_CONNECT_TIMEOUT"))
        self.assertTrue(hasattr(app, "OLLAMA_MODEL_READ_TIMEOUT"))
        self.assertTrue(hasattr(app, "OLLAMA_TIMEOUT"))

        self.assertEqual(app.OLLAMA_CONNECT_TIMEOUT, 10)
        self.assertGreaterEqual(app.OLLAMA_MODEL_READ_TIMEOUT, 300)
        self.assertEqual(
            app.OLLAMA_TIMEOUT,
            (app.OLLAMA_CONNECT_TIMEOUT, app.OLLAMA_MODEL_READ_TIMEOUT),
        )

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

    @patch("app.http_requests.post")
    def test_chat_route_timeout_emits_structured_events(self, mock_post):
        """When local model times out during chat, verify error, stage_end, and done are emitted."""
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out")

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Explain binary search"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]

        # Must include route, stage_start, error, stage_end, and done
        self.assertIn("route", types)
        self.assertIn("stage_start", types)
        self.assertIn("error", types)
        self.assertIn("stage_end", types)
        self.assertIn("done", types)

        # Verify error event structure
        err_event = next(e for e in events if e.get("type") == "error")
        self.assertEqual(err_event.get("stage"), "chat")
        self.assertIn("timed out", err_event.get("error", "").lower())

        # Verify stage_end with error=True
        stage_end_event = next(e for e in events if e.get("type") == "stage_end")
        self.assertEqual(stage_end_event.get("stage"), "chat")
        self.assertTrue(stage_end_event.get("error"))

        # Verify done event terminates stream
        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("done"))
        self.assertIsNotNone(done_event.get("error"))

        # Verify timeout argument passed to requests.post
        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs.get("timeout"), app.OLLAMA_TIMEOUT)

    @patch("app.http_requests.post")
    def test_chat_route_connection_error_emits_structured_events(self, mock_post):
        """When Ollama daemon is offline/unreachable, verify structured error sequence."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

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
        self.assertIn("stage_end", types)
        self.assertIn("done", types)

        err_event = next(e for e in events if e.get("type") == "error")
        self.assertEqual(err_event.get("stage"), "chat")
        self.assertIn("lost connection", err_event.get("error", "").lower())

    @patch("app.http_requests.post")
    def test_pipeline_architect_timeout(self, mock_post):
        """When architect stage times out in pipeline, coder stage must not run and stream must close."""
        mock_post.side_effect = requests.exceptions.Timeout("Architect timed out")

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Write a python script"}],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]

        # Architect stage should have started and failed
        stage_starts = [e.get("stage") for e in events if e.get("type") == "stage_start"]
        self.assertEqual(stage_starts, ["architect"])

        err_event = next(e for e in events if e.get("type") == "error")
        self.assertEqual(err_event.get("stage"), "architect")

        stage_end = next(e for e in events if e.get("type") == "stage_end")
        self.assertEqual(stage_end.get("stage"), "architect")
        self.assertTrue(stage_end.get("error"))

        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("done"))

    @patch("app.http_requests.post")
    def test_normal_chat_emits_done(self, mock_post):
        """Normal successful chat response should emit route, stage_start, token, and done."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "Hello!"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]

        self.assertIn("route", types)
        self.assertIn("stage_start", types)
        self.assertIn("token", types)
        self.assertIn("done", types)

        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("done"))
        self.assertEqual(done_event.get("mode"), "chat")


if __name__ == "__main__":
    unittest.main()
