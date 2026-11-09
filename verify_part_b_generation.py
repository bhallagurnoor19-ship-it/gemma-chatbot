"""
Verification of Part B: Generation verification with real Ollama calls & pipeline behavior.
Tests:
1. Normal completion: done_reason == 'stop'
2. Generation stopped by token limit: done_reason == 'length'
3. Automatic continuation: continuation appends text cleanly
4. Deduplication: continuation avoids repeated headings/paragraphs
5. Maximum continuation attempts reached: flags truncated and halts pipeline
6. Interrupted network connection: raises connection error, halts gracefully
7. Incomplete output is never marked as successful or handed off to Qwen.
"""

import json
import requests
import unittest
from unittest.mock import MagicMock, patch
import app

OLLAMA_URL = "http://localhost:11434"


class TestGenerationVerification(unittest.TestCase):
    def setUp(self):
        self.app = app.app
        self.client = self.app.test_client()

    def _parse_sse(self, data_str):
        events = []
        for line in data_str.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except Exception:
                    pass
        return events

    def test_real_ollama_normal_completion(self):
        """Test real Ollama response returns done_reason == 'stop' on a short prompt."""
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": "deepseek-coder-v2:16b",
            "messages": [{"role": "user", "content": "Return the single word: OK"}],
            "options": {"num_predict": 10},
            "stream": False
        }, timeout=30).json()
        print("Normal completion done_reason:", resp.get("done_reason"))
        self.assertEqual(resp.get("done_reason"), "stop")
        self.assertTrue(resp.get("done"))

    def test_real_ollama_token_limit_truncation(self):
        """Test real Ollama response returns done_reason == 'length' when token limit is hit."""
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": "deepseek-coder-v2:16b",
            "messages": [{"role": "user", "content": "Write a 500 word architectural analysis of microservices."}],
            "options": {"num_predict": 15},
            "stream": False
        }, timeout=30).json()
        print("Token limit done_reason:", resp.get("done_reason"))
        self.assertEqual(resp.get("done_reason"), "length")
        self.assertTrue(resp.get("done"))

    def test_real_ollama_continuation_and_deduplication(self):
        """Test sending continuation prompt to Ollama continues without duplicating headings."""
        # Step 1: Generate initial truncated part
        resp1 = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": "deepseek-coder-v2:16b",
            "messages": [{"role": "user", "content": "List the numbers 1 to 20 with one number per line formatted as: 'Number X'."}],
            "options": {"num_predict": 20},
            "stream": False
        }, timeout=30).json()
        part1 = resp1.get("message", {}).get("content", "")
        self.assertEqual(resp1.get("done_reason"), "length")

        # Step 2: Auto-continuation prompt
        resp2 = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": "deepseek-coder-v2:16b",
            "messages": [
                {"role": "user", "content": "List the numbers 1 to 20 with one number per line formatted as: 'Number X'."},
                {"role": "assistant", "content": part1},
                {"role": "user", "content": "Please continue directly from where you stopped. Do not repeat any preceding sentences, headings, or thoughts."}
            ],
            "options": {"num_predict": 50},
            "stream": False
        }, timeout=30).json()
        part2 = resp2.get("message", {}).get("content", "")

        # Verify continuation didn't restart from 'Number 1'
        self.assertNotIn("Number 1\n", part2)
        print(f"Part 1 ended with:\n{part1[-40:]}")
        print(f"Part 2 started with:\n{part2[:40]}")

    def test_max_continuation_attempts_halts_handoff(self):
        """When continuation attempts are exhausted, verify coder stage never runs."""
        with patch("app.http_requests.post") as mock_post:
            resp_mock = MagicMock()
            resp_mock.raise_for_status = MagicMock()
            resp_mock.iter_lines.return_value = [
                json.dumps({"message": {"content": "Truncated content chunk"}, "done": False}),
                json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length"}),
            ]
            resp_mock.__enter__.return_value = resp_mock
            mock_post.return_value = resp_mock

            response = self.client.post("/api/chat", json={
                "messages": [{"role": "user", "content": "Implement the complete architecture and code"}],
                "mode": "pipeline",
                "rag_mode": "off"
            })
            events = self._parse_sse(response.get_data(as_text=True))
            stage_starts = [e.get("stage") for e in events if e.get("type") == "stage_start"]
            self.assertEqual(stage_starts, ["architect"])
            self.assertNotIn("coder", stage_starts)

            # Check stage_end for architect has truncated: True
            arch_end = next(e for e in events if e.get("type") == "stage_end")
            self.assertTrue(arch_end.get("truncated"))
            self.assertNotIn("done", arch_end)  # not successful completion

            # Check done event has truncated: True
            done_ev = next(e for e in events if e.get("type") == "done")
            self.assertTrue(done_ev.get("truncated"))
            self.assertTrue(done_ev.get("interrupted"))

    def test_interrupted_connection_handling(self):
        """Simulate dropped network connection to Ollama."""
        with patch("app.http_requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection forcibly closed")
            response = self.client.post("/api/chat", json={
                "messages": [{"role": "user", "content": "Explain system architecture"}],
                "mode": "deepseek",
                "rag_mode": "off"
            })
            events = self._parse_sse(response.get_data(as_text=True))
            err_event = next(e for e in events if e.get("type") == "error")
            self.assertIn("Lost connection", err_event.get("error"))
            done_event = next(e for e in events if e.get("type") == "done")
            self.assertTrue(done_event.get("done"))
            self.assertIn("Lost connection", done_event.get("error"))


if __name__ == "__main__":
    unittest.main()
