"""
Unit and regression tests for incomplete response handling and premature handoffs
in the multi-LLM Aider workflow.

Verifies that:
1. OLLAMA_OPTIONS is configured with expanded context window (8192) and output limit (4096).
2. Intent classifier routes pure architectural assessments to DeepSeek Architect ('deepseek')
   and requests requiring code implementation to the Coding Pipeline ('pipeline').
3. Auto-continuation triggers when Ollama finishes with done_reason == 'length'.
4. Premature handoff to Qwen is prevented when the Architect response remains truncated.
5. Purely analytical queries in 'pipeline' mode complete Stage 1 without triggering Stage 2 (Coder).
6. Full coding requests in 'pipeline' mode execute both Stage 1 and Stage 2 seamlessly.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
import app


class TestAiderIncompleteWorkflow(unittest.TestCase):
    def setUp(self):
        self.app = app.app
        self.client = self.app.test_client()

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

    def test_ollama_options_configured(self):
        """Verify OLLAMA_OPTIONS provides adequate context window and generation limits."""
        self.assertTrue(hasattr(app, "OLLAMA_OPTIONS"))
        self.assertIsInstance(app.OLLAMA_OPTIONS, dict)
        self.assertGreaterEqual(app.OLLAMA_OPTIONS.get("num_ctx", 0), 8192)
        self.assertGreaterEqual(app.OLLAMA_OPTIONS.get("num_predict", 0), 4096)

    def test_intent_classification_architectural_vs_coding(self):
        """Verify pure architectural assessments route to DeepSeek without unsolicited code generation."""
        # 1. Pure architectural assessment -> deepseek
        pure_arch_msg = [
            {"role": "user", "content": "Please conduct an architectural assessment of our microservices system design."}
        ]
        self.assertEqual(app.classify_intent(pure_arch_msg, "auto"), "deepseek")

        # 2. Architectural review -> deepseek
        arch_review_msg = [
            {"role": "user", "content": "Perform an architecture review and evaluate the trade-offs of using Cassandra."}
        ]
        self.assertEqual(app.classify_intent(arch_review_msg, "auto"), "deepseek")

        # 3. Architectural query that explicitly asks to write code -> pipeline
        arch_code_msg = [
            {
                "role": "user",
                "content": "Evaluate our architecture and write the code for the new auth repository service.",
            }
        ]
        self.assertEqual(app.classify_intent(arch_code_msg, "auto"), "pipeline")

        # 4. Standard coding request -> pipeline
        code_msg = [
            {"role": "user", "content": "Write a python script to parse CSV files and insert into SQLite."}
        ]
        self.assertEqual(app.classify_intent(code_msg, "auto"), "pipeline")

        # 5. General conversational question -> chat
        chat_msg = [{"role": "user", "content": "What is the capital of France?"}]
        self.assertEqual(app.classify_intent(chat_msg, "auto"), "chat")

    @patch("app.http_requests.post")
    def test_auto_continuation_on_length_limit(self, mock_post):
        """When Ollama stops with done_reason == 'length', auto-continuation should query for remaining tokens."""
        # First call hits length limit; second call finishes cleanly with 'stop'
        resp_call1 = MagicMock()
        resp_call1.raise_for_status = MagicMock()
        resp_call1.iter_lines.return_value = [
            json.dumps({"message": {"content": "Section 1. Overview\nSection 2. Design"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length"}),
        ]
        resp_call1.__enter__.return_value = resp_call1

        resp_call2 = MagicMock()
        resp_call2.raise_for_status = MagicMock()
        resp_call2.iter_lines.return_value = [
            json.dumps({"message": {"content": "\nSection 2.2 Deep Dive\nSection 3. Conclusion"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}),
        ]
        resp_call2.__enter__.return_value = resp_call2

        mock_post.side_effect = [resp_call1, resp_call2]

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Perform an architectural assessment of our system."}
                ],
                "mode": "deepseek",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        tokens = [e.get("token") for e in events if e.get("type") == "token"]
        full_text = "".join(tokens)

        # Both sections should be present in the final output
        self.assertIn("Section 1. Overview", full_text)
        self.assertIn("Section 2.2 Deep Dive", full_text)
        self.assertIn("Section 3. Conclusion", full_text)

        # Ollama should have been called twice (initial + continuation)
        self.assertEqual(mock_post.call_count, 2)

        # Check options were passed in the payload
        first_call_payload = mock_post.call_args_list[0][1]["json"]
        self.assertEqual(first_call_payload.get("options"), app.OLLAMA_OPTIONS)

        # Done event should indicate clean completion without error/truncation
        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("done"))
        self.assertIsNone(done_event.get("error"))
        self.assertFalse(done_event.get("truncated", False))

    @patch("app.http_requests.post")
    def test_premature_handoff_prevention_on_unresolved_truncation(self, mock_post):
        """When architect is truncated and exceeds continuation limit, Stage 2 (coder) must not start."""
        # Initial call and all continuation attempts hit 'length' limit
        def make_length_mock(part_text):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.iter_lines.return_value = [
                json.dumps({"message": {"content": part_text}, "done": False}),
                json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length"}),
            ]
            resp.__enter__.return_value = resp
            return resp

        mock_post.side_effect = [
            make_length_mock("Part 1: Incomplete blueprint..."),
            make_length_mock("Part 2: Still incomplete..."),
            make_length_mock("Part 3: Still incomplete at limit..."),
        ]

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Evaluate our architecture and write the code for our service.",
                    }
                ],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        types = [e.get("type") for e in events]

        # Stage starts should ONLY contain architect, NOT coder
        stage_starts = [e.get("stage") for e in events if e.get("type") == "stage_start"]
        self.assertEqual(stage_starts, ["architect"])
        self.assertNotIn("coder", stage_starts)

        # Stage end for architect should be flagged as truncated
        arch_stage_end = next(
            e for e in events if e.get("type") == "stage_end" and e.get("stage") == "architect"
        )
        self.assertTrue(arch_stage_end.get("truncated"))
        self.assertEqual(arch_stage_end.get("done_reason"), "length")

        # Error and done events should clearly announce truncation and prevent handoff
        err_event = next(e for e in events if e.get("type") == "error")
        self.assertTrue(err_event.get("truncated"))
        self.assertIn("paused to prevent incomplete code", err_event.get("error", ""))

        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("truncated"))
        self.assertTrue(done_event.get("interrupted"))

    @patch("app.http_requests.post")
    def test_pipeline_pure_architectural_skips_coder(self, mock_post):
        """When an architectural assessment is submitted in pipeline mode, Stage 2 (coder) is skipped."""
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "Comprehensive architectural assessment..."}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}),
        ]
        resp.__enter__.return_value = resp
        mock_post.return_value = resp

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Provide a complete architectural assessment and trade-off analysis of our database.",
                    }
                ],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        stage_starts = [e.get("stage") for e in events if e.get("type") == "stage_start"]

        # Only architect stage should have run
        self.assertEqual(stage_starts, ["architect"])
        self.assertNotIn("coder", stage_starts)

        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("done"))
        self.assertTrue(done_event.get("analytical_only"))
        # Aider pairing command should target architect mode
        self.assertIn("--architect", done_event.get("aider_cmd", ""))
        self.assertNotIn("--editor-model", done_event.get("aider_cmd", ""))

    @patch("app.http_requests.post")
    def test_successful_pipeline_runs_both_stages(self, mock_post):
        """When an implementation request is processed without truncation, both stages run."""
        resp_arch = MagicMock()
        resp_arch.raise_for_status = MagicMock()
        resp_arch.iter_lines.return_value = [
            json.dumps({"message": {"content": "Architect blueprint: Class structure..."}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}),
        ]
        resp_arch.__enter__.return_value = resp_arch

        resp_coder = MagicMock()
        resp_coder.raise_for_status = MagicMock()
        resp_coder.iter_lines.return_value = [
            json.dumps({"message": {"content": "class AuthRepository:\n    pass"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}),
        ]
        resp_coder.__enter__.return_value = resp_coder

        mock_post.side_effect = [resp_arch, resp_coder]

        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Write the code to implement the authentication repository class.",
                    }
                ],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(response.status_code, 200)

        events = self._parse_sse_events(response.get_data(as_text=True))
        stage_starts = [e.get("stage") for e in events if e.get("type") == "stage_start"]

        # Both architect and coder should have started
        self.assertEqual(stage_starts, ["architect", "coder"])

        # Both stages should have completed cleanly
        stage_ends = [e.get("stage") for e in events if e.get("type") == "stage_end"]
        self.assertEqual(stage_ends, ["architect", "coder"])

        done_event = next(e for e in events if e.get("type") == "done")
        self.assertTrue(done_event.get("done"))
        self.assertFalse(done_event.get("truncated", False))
        self.assertIn("--editor-model", done_event.get("aider_cmd", ""))


if __name__ == "__main__":
    unittest.main()
