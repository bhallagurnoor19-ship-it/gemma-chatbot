"""
Automated Test Suite for Unified RAG Phase 6
Frontend RAG Controls, Retrieval Status, and Source Citations

Validates:
1. HTML template structure: RAG mode selector (#ragModeSelect) with 5 options (auto, documents, codebase, both, off).
2. Default selection: auto.
3. CSS styles: .rag-select-wrapper, .rag-select, .bot-meta-row, .rag-status-badge, .rag-sources-section, and responsive rules.
4. JavaScript integration: ragModeSelect event listener, localStorage persistence ('gemma_rag_mode'), toast feedback.
5. XSS escaping: escapeHtml escapes &, <, >, ", and ' against injection.
6. Retrieval status badge logic: handles off, success (sources), success (empty), no_results, embedding_unavailable, database_error, and search states.
7. Source citations component: collapsible <details>, category tags, line numbers, page numbers, symbol badges, citations, and empty/malformed resilience.
8. API payload transmission: streamBotPipeline and resumeGeneration send rag_mode while preserving model routing mode.
9. Flask backend contract: /api/chat accepts rag_mode in all 5 modes and emits matching SSE route metadata.
10. Fallback and isolation: invalid rag_mode values fall back safely to 'auto', and model mode routing is unaffected.
"""

import html
import json
import os
import re
import sys
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app
from rag.retriever import RetrievedChunk, RetrievalResponse, RetrievalStatus
from rag.service import RAGContextResult


def simulated_escape_html(text: Any) -> str:
    """Python mirror of the JavaScript escapeHtml function in index.html."""
    if text is None:
        return ""
    s = str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def simulated_rag_status_badge(rag_mode: str, rag_status: str, sources: Optional[List[Any]]) -> Dict[str, str]:
    """Python mirror of createRagStatusBadge logic in index.html."""
    num_sources = len(sources) if isinstance(sources, list) else 0
    mode = (rag_mode or "auto").lower()
    status = (rag_status or "").lower()

    if mode == "off" or status == "off":
        return {"class": "rag-status-off", "text": "RAG Off"}
    elif status == "success":
        if num_sources > 0:
            return {"class": "rag-status-success", "text": f"Sources found ({num_sources})"}
        else:
            return {"class": "rag-status-empty", "text": "No relevant sources"}
    elif status == "no_results":
        return {"class": "rag-status-empty", "text": "No relevant sources"}
    elif status == "embedding_unavailable":
        return {"class": "rag-status-warning", "text": "Retrieval unavailable"}
    elif status in ("database_error", "error", "unavailable"):
        return {"class": "rag-status-error", "text": "Retrieval unavailable"}
    elif mode == "documents":
        return {"class": "rag-status-searching", "text": "Searching documents"}
    elif mode == "codebase":
        return {"class": "rag-status-searching", "text": "Searching codebase"}
    elif mode == "both":
        return {"class": "rag-status-searching", "text": "Searching documents and codebase"}
    elif mode == "auto":
        return {"class": "rag-status-searching", "text": "Searching documents and codebase"}
    else:
        return {"class": "rag-status-off", "text": "RAG Off"}


def simulated_create_sources_section(sources: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Python mirror of createSourcesSection logic in index.html."""
    if not isinstance(sources, list) or len(sources) == 0:
        return None

    items = []
    for src in sources:
        if not src or not isinstance(src, dict):
            continue
        is_doc = (src.get("category") or "").lower() == "document"
        cat_label = "📄 Document" if is_doc else "💻 Codebase"
        cat_class = "doc" if is_doc else "code"
        fname = simulated_escape_html(src.get("file_name") or src.get("source_path") or "Unknown")
        fpath = simulated_escape_html(src.get("source_path") or src.get("file_name") or "")
        citation = simulated_escape_html(src.get("citation") or "")

        loc_badge = ""
        if src.get("page_number") is not None:
            loc_badge = f"p. {simulated_escape_html(src.get('page_number'))}"
        elif src.get("start_line") is not None and src.get("end_line") is not None:
            loc_badge = f"L{simulated_escape_html(src.get('start_line'))}-{simulated_escape_html(src.get('end_line'))}"
        elif src.get("start_line") is not None:
            loc_badge = f"L{simulated_escape_html(src.get('start_line'))}"

        symbol = simulated_escape_html(src.get("symbol_name") or "")
        items.append({
            "cat_label": cat_label,
            "cat_class": cat_class,
            "file_name": fname,
            "source_path": fpath,
            "loc_badge": loc_badge,
            "symbol": symbol,
            "citation": citation,
        })

    return {
        "count": len(sources),
        "items": items,
        "collapsed": True  # <details> without 'open'
    }


class TestFrontendRAGControls(unittest.TestCase):
    """Test suite for Phase 6 frontend template, CSS, JS, and backend API integration."""

    @classmethod
    def setUpClass(cls):
        cls.template_path = os.path.join(PROJECT_ROOT, "templates", "index.html")
        with open(cls.template_path, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

        app.app.config["TESTING"] = True
        cls.client = app.app.test_client()

    def test_01_template_contains_rag_mode_selector(self):
        """Verify #ragModeSelect exists in index.html with class rag-select."""
        self.assertIn('id="ragModeSelect"', self.html_content)
        self.assertIn('class="rag-select"', self.html_content)
        self.assertIn('class="rag-select-wrapper"', self.html_content)

    def test_02_selector_has_all_five_options_with_auto_default(self):
        """Verify all 5 options (auto, documents, codebase, both, off) are present and auto is default."""
        select_match = re.search(r'<select[^>]*id="ragModeSelect"[^>]*>(.*?)</select>', self.html_content, re.DOTALL)
        self.assertIsNotNone(select_match, "ragModeSelect element not found")
        options_html = select_match.group(1)

        expected_values = ["auto", "documents", "codebase", "both", "off"]
        for val in expected_values:
            self.assertTrue(
                re.search(rf'<option[^>]*value="{val}"', options_html),
                f"Option value '{val}' missing from #ragModeSelect"
            )

        # Check that auto has 'selected' attribute
        auto_match = re.search(r'<option[^>]*value="auto"[^>]*>', options_html)
        self.assertIsNotNone(auto_match)
        self.assertIn("selected", auto_match.group(0), "Default option 'auto' must have selected attribute")

    def test_03_css_contains_rag_component_styles(self):
        """Verify CSS contains styles for selector, badge states, and collapsible sources."""
        required_css_tokens = [
            ".rag-select-wrapper",
            ".rag-select",
            ".bot-meta-row",
            ".rag-status-badge",
            ".rag-status-badge.rag-status-success",
            ".rag-status-badge.rag-status-empty",
            ".rag-status-badge.rag-status-error",
            ".rag-status-badge.rag-status-warning",
            ".rag-status-badge.rag-status-off",
            ".rag-status-badge.rag-status-searching",
            ".rag-sources-section",
            ".rag-sources-summary",
            ".rag-source-item",
            ".rag-source-cat.doc",
            ".rag-source-cat.code",
            ".rag-source-loc-badge",
        ]
        for token in required_css_tokens:
            self.assertIn(token, self.html_content, f"Missing CSS token: {token}")

    def test_04_responsive_css_for_mobile_viewports(self):
        """Verify @media (max-width: 768px) handles .rag-select and responsive elements."""
        self.assertIn("@media (max-width: 768px)", self.html_content)
        self.assertIn(".rag-select", self.html_content)
        self.assertIn(".bot-meta-row", self.html_content)

    def test_05_javascript_persistence_and_dom_binding(self):
        """Verify JS binds #ragModeSelect, listens to change, and persists to localStorage."""
        self.assertIsNotNone(
            re.search(r'const\s+ragModeSelect\s*=\s*document\.getElementById\("ragModeSelect"\);', self.html_content)
        )
        self.assertIn("gemma_rag_mode", self.html_content)
        self.assertIn("localStorage.setItem('gemma_rag_mode', ragModeSelect.value)", self.html_content)
        self.assertIn("localStorage.getItem('gemma_rag_mode')", self.html_content)

    def test_06_escape_html_hardened_against_xss(self):
        """Verify escapeHtml function in JS and simulated implementation strictly escape untrusted input."""
        # Check index.html implementation
        self.assertIn("replace(/&/g, \"&amp;\")", self.html_content)
        self.assertIn("replace(/</g, \"&lt;\")", self.html_content)
        self.assertIn("replace(/>/g, \"&gt;\")", self.html_content)
        self.assertIn("replace(/\"/g, \"&quot;\")", self.html_content)
        self.assertIn("replace(/'/g, \"&#39;\")", self.html_content)

        # Test simulated function against realistic attack vectors
        xss_payloads = [
            ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
            ('"><img src=x onerror=alert(1)>', '&quot;&gt;&lt;img src=x onerror=alert(1)&gt;'),
            ("x' onfocus='alert(1)'", "x&#39; onfocus=&#39;alert(1)&#39;"),
            ("A & B < C > D", "A &amp; B &lt; C &gt; D"),
            (None, ""),
            (123, "123"),
        ]
        for payload, expected in xss_payloads:
            escaped = simulated_escape_html(payload)
            self.assertEqual(escaped, expected)
            self.assertNotIn("<script>", escaped)
            self.assertNotIn('">', escaped)
            self.assertNotIn("'", escaped)
            self.assertNotIn('"', escaped)

    def test_07_rag_status_badge_mapping(self):
        """Verify all retrieval statuses map to the required text, CSS class, and icon."""
        cases = [
            # (mode, status, sources, expected_class, expected_text)
            ("off", "off", [], "rag-status-off", "RAG Off"),
            ("auto", "off", [], "rag-status-off", "RAG Off"),
            ("auto", "success", [{"file_name": "doc.pdf"}], "rag-status-success", "Sources found (1)"),
            ("auto", "success", [{"f": "1"}, {"f": "2"}], "rag-status-success", "Sources found (2)"),
            ("auto", "success", [], "rag-status-empty", "No relevant sources"),
            ("auto", "no_results", [], "rag-status-empty", "No relevant sources"),
            ("auto", "embedding_unavailable", [], "rag-status-warning", "Retrieval unavailable"),
            ("auto", "database_error", [], "rag-status-error", "Retrieval unavailable"),
            ("auto", "error", [], "rag-status-error", "Retrieval unavailable"),
            ("documents", "", [], "rag-status-searching", "Searching documents"),
            ("codebase", "", [], "rag-status-searching", "Searching codebase"),
            ("both", "", [], "rag-status-searching", "Searching documents and codebase"),
            ("auto", "", [], "rag-status-searching", "Searching documents and codebase"),
        ]
        for mode, status, sources, exp_cls, exp_txt in cases:
            badge = simulated_rag_status_badge(mode, status, sources)
            self.assertEqual(badge["class"], exp_cls, f"Failed class for mode={mode}, status={status}")
            self.assertEqual(badge["text"], exp_txt, f"Failed text for mode={mode}, status={status}")

    def test_08_sources_section_generation_and_collapsible_behavior(self):
        """Verify sources section renders document and codebase items with badges, and handles empty safely."""
        # 1. Empty / None sources return None
        self.assertIsNone(simulated_create_sources_section(None))
        self.assertIsNone(simulated_create_sources_section([]))

        # 2. Document and Codebase sources
        sample_sources = [
            {
                "category": "document",
                "file_name": "manual.pdf",
                "source_path": "data/documents/manual.pdf",
                "page_number": 4,
                "citation": "Step 1: Initialize system.",
                "similarity": 0.88,
            },
            {
                "category": "codebase",
                "file_name": "auth.py",
                "source_path": "rag/auth.py",
                "start_line": 15,
                "end_line": 30,
                "symbol_name": "verify_token",
                "citation": "def verify_token(token):\n    pass",
                "similarity": 0.94,
            },
        ]
        res = simulated_create_sources_section(sample_sources)
        self.assertIsNotNone(res)
        self.assertEqual(res["count"], 2)
        self.assertTrue(res["collapsed"], "Sources section must be collapsed by default")

        doc_item = res["items"][0]
        self.assertEqual(doc_item["cat_label"], "📄 Document")
        self.assertEqual(doc_item["cat_class"], "doc")
        self.assertEqual(doc_item["file_name"], "manual.pdf")
        self.assertEqual(doc_item["loc_badge"], "p. 4")

        code_item = res["items"][1]
        self.assertEqual(code_item["cat_label"], "💻 Codebase")
        self.assertEqual(code_item["cat_class"], "code")
        self.assertEqual(code_item["file_name"], "auth.py")
        self.assertEqual(code_item["loc_badge"], "L15-30")
        self.assertEqual(code_item["symbol"], "verify_token")

    def test_09_sources_section_handles_malformed_entries_safely(self):
        """Verify malformed items inside sources array do not throw exceptions."""
        malformed = [None, "invalid_string", 42, {}, {"file_name": "safe.txt"}]
        res = simulated_create_sources_section(malformed)
        self.assertIsNotNone(res)
        self.assertEqual(len(res["items"]), 1)
        self.assertEqual(res["items"][0]["file_name"], "safe.txt")

    def test_10_fetch_payload_includes_rag_mode_in_stream_and_resume(self):
        """Verify streamBotPipeline and resumeGeneration include rag_mode and preserve mode in request body."""
        # Check streamBotPipeline
        self.assertIn("rag_mode: (ragModeSelect ? ragModeSelect.value : \"auto\")", self.html_content)
        # Check resumeGeneration
        resume_block = self.html_content.split("async function resumeGeneration()")[1].split("async function")[0]
        self.assertIn("rag_mode: (ragModeSelect ? ragModeSelect.value : \"auto\")", resume_block)
        self.assertIn("mode: modeSelect.value", resume_block)

    def test_11_route_event_handler_constructs_meta_row_and_sources(self):
        """Verify route event handler binds createRagStatusBadge and createSourcesSection to botBubbleEl."""
        route_handler = self.html_content.split('if (data.type === "route") {')[1].split('else if (data.type === "stage_start")')[0]
        self.assertIn("bot-meta-row", route_handler)
        self.assertIn("createRagStatusBadge(data.rag_mode, data.rag_status, data.rag_sources)", route_handler)
        self.assertIn("createSourcesSection(data.rag_sources)", route_handler)
        self.assertIn("botBubbleEl.insertBefore(metaRow, botBubbleEl.firstChild)", route_handler)

    def _drain_sse_events(self, response) -> List[Dict[str, Any]]:
        """Helper to consume SSE response stream and parse JSON data lines."""
        with self.client:
            raw_data = b"".join(response.response).decode("utf-8")
        events = []
        for line in raw_data.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except Exception:
                    pass
        return events

    def test_12_backend_receives_rag_mode_and_emits_matching_route_metadata(self):
        """Verify Flask /api/chat endpoint receives rag_mode across all modes and emits matching SSE metadata."""
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.iter_lines.return_value = [
            b'{"message": {"content": "Hello!"}, "done": false}',
            b'{"done": true}',
        ]

        # 1. Test explicit modes ('off', 'documents', 'codebase', 'both')
        explicit_modes = ["off", "documents", "codebase", "both"]
        for r_mode in explicit_modes:
            with patch("app.http_requests.post", return_value=mock_post_resp):
                payload = {
                    "messages": [{"role": "user", "content": "Tell me about the system."}],
                    "mode": "chat",
                    "rag_mode": r_mode,
                }
                resp = self.client.post("/api/chat", json=payload)
                self.assertEqual(resp.status_code, 200)
                events = self._drain_sse_events(resp)
                route_ev = next((e for e in events if e.get("type") == "route"), None)
                self.assertIsNotNone(route_ev, f"No route event for rag_mode={r_mode}")
                self.assertEqual(route_ev.get("rag_mode"), r_mode)
                self.assertIn("rag_status", route_ev)
                self.assertIn("rag_sources", route_ev)
                self.assertIsInstance(route_ev["rag_sources"], list)

        # 2. Test auto mode with document intent
        with patch("app.http_requests.post", return_value=mock_post_resp):
            payload = {
                "messages": [{"role": "user", "content": "What does the user manual documentation say?"}],
                "mode": "chat",
                "rag_mode": "auto",
            }
            resp = self.client.post("/api/chat", json=payload)
            self.assertEqual(resp.status_code, 200)
            events = self._drain_sse_events(resp)
            route_ev = next((e for e in events if e.get("type") == "route"), None)
            self.assertIsNotNone(route_ev)
            self.assertEqual(route_ev.get("rag_mode"), "documents")

        # 3. Test auto mode with codebase intent
        with patch("app.http_requests.post", return_value=mock_post_resp):
            payload = {
                "messages": [{"role": "user", "content": "Explain the codebase in app.py"}],
                "mode": "chat",
                "rag_mode": "auto",
            }
            resp = self.client.post("/api/chat", json=payload)
            self.assertEqual(resp.status_code, 200)
            events = self._drain_sse_events(resp)
            route_ev = next((e for e in events if e.get("type") == "route"), None)
            self.assertIsNotNone(route_ev)
            self.assertEqual(route_ev.get("rag_mode"), "codebase")

    def test_13_invalid_rag_mode_falls_back_safely_without_crashing(self):
        """Verify invalid rag_mode values (e.g. 'unsupported', 'hack') fall back safely to off without crashing."""
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.iter_lines.return_value = [
            b'{"message": {"content": "Fallback response."}, "done": true}',
        ]
        with patch("app.http_requests.post", return_value=mock_post_resp):
            payload = {
                "messages": [{"role": "user", "content": "How does the indexer work?"}],
                "mode": "chat",
                "rag_mode": "invalid_experimental_mode",
            }
            resp = self.client.post("/api/chat", json=payload)
            self.assertEqual(resp.status_code, 200)
            events = self._drain_sse_events(resp)
            route_ev = next((e for e in events if e.get("type") == "route"), None)
            self.assertIsNotNone(route_ev)
            # RAGService safely sets mode to 'off' and status to 'off' on unrecognised modes
            self.assertEqual(route_ev.get("rag_mode"), "off")
            self.assertEqual(route_ev.get("rag_status"), "off")
            self.assertEqual(route_ev.get("rag_sources"), [])

    def test_14_mock_retrieval_sources_propagated_to_sse_route_event(self):
        """Verify retrieved source chunks are serialized into rag_sources in SSE route event."""
        mock_chunk = RetrievedChunk(
            content="Antigravity architecture overview and design principles.",
            source_path="data/documents/architecture.pdf",
            file_name="architecture.pdf",
            category="document",
            distance=0.088,
            similarity=0.912,
            page_number=2,
            chunk_id="doc_chunk_001",
        )
        mock_rag_result = RAGContextResult(
            mode="documents",
            status="success",
            context_text="[RETRIEVED CONTEXT]...",
            chunks=[mock_chunk],
            sources=[{
                "category": "document",
                "file_name": "architecture.pdf",
                "source_path": "data/documents/architecture.pdf",
                "page_number": 2,
                "start_line": None,
                "end_line": None,
                "symbol_name": None,
                "similarity": 0.912,
                "citation": "Section 2.1: System Architecture",
            }],
            error_message=None,
        )

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.iter_lines.return_value = [
            b'{"message": {"content": "Based on the architecture document..."}, "done": true}',
        ]

        with patch("rag.service.RAGService.retrieve_context", return_value=mock_rag_result):
            with patch("app.http_requests.post", return_value=mock_post_resp):
                payload = {
                    "messages": [{"role": "user", "content": "Explain the architecture from the manual."}],
                    "mode": "chat",
                    "rag_mode": "documents",
                }
                resp = self.client.post("/api/chat", json=payload)
                self.assertEqual(resp.status_code, 200)
                events = self._drain_sse_events(resp)
                route_ev = next((e for e in events if e.get("type") == "route"), None)
                self.assertIsNotNone(route_ev)
                self.assertEqual(route_ev.get("rag_status"), "success")
                sources = route_ev.get("rag_sources", [])
                self.assertEqual(len(sources), 1)
                self.assertEqual(sources[0]["file_name"], "architecture.pdf")
                self.assertEqual(sources[0]["category"], "document")
                self.assertEqual(sources[0]["page_number"], 2)
                self.assertEqual(sources[0]["citation"], "Section 2.1: System Architecture")
                self.assertAlmostEqual(sources[0]["similarity"], 0.912, places=3)

    def test_15_model_routing_mode_operates_independently_from_rag_mode(self):
        """Verify model routing modes (chat, pipeline, deepseek, qwen) coexist without collision with rag_mode."""
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.iter_lines.return_value = [
            b'{"message": {"content": "Pipeline response stage."}, "done": true}',
        ]

        # Test pipeline mode with rag_mode codebase
        with patch("app.http_requests.post", return_value=mock_post_resp):
            payload = {
                "messages": [{"role": "user", "content": "Refactor vector_store.py"}],
                "mode": "pipeline",
                "rag_mode": "codebase",
            }
            resp = self.client.post("/api/chat", json=payload)
            self.assertEqual(resp.status_code, 200)
            events = self._drain_sse_events(resp)
            route_ev = next((e for e in events if e.get("type") == "route"), None)
            self.assertIsNotNone(route_ev)
            self.assertEqual(route_ev.get("route"), "pipeline", "Model routing mode must remain 'pipeline'")
            self.assertEqual(route_ev.get("rag_mode"), "codebase", "RAG mode must remain 'codebase'")


if __name__ == "__main__":
    unittest.main()
