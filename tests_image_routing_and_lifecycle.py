"""
Focused Verification Test Suite: Image-Aware Routing & Lifecycle Integration
Tests:
1. Local vision normal completion (Ollama payload, base64 images, route announcement, lease release).
2. Local vision cancellation while worker is alive (lease preserved under worker ownership).
3. Lease remains unavailable while worker is alive (subsequent acquisition times out).
4. Lease becomes releasable after actual worker termination (subsequent acquisition succeeds).
5. Cloud-disabled image requests never invoke Gemini (both explicit 'gemini' and 'auto' mode).
6. Text-only Gemini requests still work when cloud multimodal is disabled.
7. Strict rejection (HTTP 400) of incompatible text modes (pipeline, deepseek, qwen, chat) with images.
8. Image quota and format security enforcement (count, per-image 10MB, aggregate 25MB, magic sniffing).
9. /api/upload endpoint security (magic byte sniffing, 10MB limit, mime_type reporting).
10. Settings persistence for ALLOW_CLOUD_MULTIMODAL.
"""

import base64
import json
import os
import queue
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import MagicMock, patch

from app import (
    app,
    cancellation_registry,
    classify_intent,
    is_cloud_multimodal_enabled,
    save_env_config,
    sniff_image_mime,
    validate_and_decode_images,
    extract_attachment_summary,
    STATIC_CAPABILITY_REGISTRY,
    is_model_vision_capable,
    resolve_vision_model,
)
from model_lifecycle import (
    EmbeddingPolicy,
    ModelLease,
    ModelLeaseType,
    ModelLeaseTimeoutError,
    ModelLeaseCancelledError,
    OllamaLifecycleManager,
    _LeaseState,
    reset_model_manager_for_testing,
)

# 1x1 valid PNG (68 bytes)
TINY_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01Haf\xa0\x00\x00\x00\x00IEND\xaeB`\x82"
TINY_PNG_B64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")


class TestImageRoutingAndLifecycle(unittest.TestCase):
    def setUp(self):
        self.manager = OllamaLifecycleManager(
            default_timeout=5.0,
            eviction_timeout=0.3,
            embedding_policy=EmbeddingPolicy.PERSISTENT,
        )
        self.manager._query_ollama_ps = MagicMock(return_value=[])
        self.manager._evict_model_http = MagicMock(return_value=True)
        reset_model_manager_for_testing(self.manager)
        cancellation_registry.clear()
        self.client = app.test_client()

        # Disable RAG context retrieval
        self.rag_patch = patch("app.get_rag_service")
        self.mock_rag = self.rag_patch.start()
        self.mock_rag.return_value.retrieve_context.return_value = MagicMock(
            has_context=False,
            context_text="",
            mode="off",
            status="disabled",
            sources=[],
        )

        # Snapshot environment for strict test isolation
        self._orig_env = dict(os.environ)
        os.environ["ALLOW_CLOUD_MULTIMODAL"] = "false"

    def tearDown(self):
        self.rag_patch.stop()
        cancellation_registry.clear()
        reset_model_manager_for_testing(None)
        # Restore original environment
        os.environ.clear()
        os.environ.update(self._orig_env)

    # ── Test 1: Local Vision Normal Completion ─────────────────────────────

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["llama3.2-vision:latest"])
    def test_01_local_vision_normal_completion(self, mock_models, mock_post):
        """
        Local vision normal completion:
        - Correct model 'llama3.2-vision:latest' is selected.
        - Ollama payload contains cleaned base64 image data in user message.
        - Generation lease is acquired and released.
        - Stream emits route, stage_start(vision), token, and done.
        """
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'{"message": {"content": "I see a blue dot."}, "done": false}',
            b'{"message": {"content": ""}, "done": true, "done_reason": "stop"}',
        ]
        mock_resp.close = MagicMock()
        mock_post.return_value = mock_resp

        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Describe this image",
                        "images": [f"data:image/png;base64,{TINY_PNG_B64}"],
                    }
                ],
                "mode": "auto",
            },
        )
        self.assertEqual(resp.status_code, 200)

        raw_events = [c.decode("utf-8") if isinstance(c, bytes) else c for c in resp.response]
        full_stream = "".join(raw_events)

        self.assertIn('"route": "local_vision"', full_stream)
        self.assertIn('"stage": "vision"', full_stream)
        self.assertIn("I see a blue dot.", full_stream)
        self.assertIn('"done": true', full_stream)

        # Verify Ollama request payload
        ollama_chat_call = None
        for call in mock_post.call_args_list:
            if call[0][0].endswith("/api/chat"):
                ollama_chat_call = call
                break

        self.assertIsNotNone(ollama_chat_call, "Ollama /api/chat must be called")
        sent_json = ollama_chat_call[1]["json"]
        self.assertEqual(sent_json["model"], "llama3.2-vision:latest")
        
        # Verify cleaned base64 (without data URI prefix)
        user_msg = [m for m in sent_json["messages"] if m.get("role") == "user"][-1]
        self.assertIn("images", user_msg)
        self.assertEqual(user_msg["images"], [TINY_PNG_B64])

        # Verify lease cleanup
        active_leases = self.manager._active_gen_leases.get("llama3.2-vision:latest", set())
        self.assertEqual(len(active_leases), 0, "Vision lease must be released after completion")
        self.assertFalse(cancellation_registry.is_active(req_id), "Registry entry must be cleared")

    # ── Test 2, 3, 4: Worker Lifecycle, Cancellation, and Mutual Exclusion ──

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["llama3.2-vision:latest"])
    def test_02_to_04_local_vision_cancellation_and_worker_lease_preservation(self, mock_models, mock_post):
        """
        Tests 2, 3, 4 in single coherent lifecycle flow:
        Test 2: Cancellation while worker is alive preserves lease under worker ownership.
        Test 3: Subsequent generation request times out while worker remains alive.
        Test 4: Once worker terminates, lease is released and subsequent request succeeds.
        """
        worker_started = threading.Event()
        worker_unblock = threading.Event()
        worker_finished = threading.Event()

        def stream_generator(*args, **kwargs):
            worker_started.set()
            yield b'{"message": {"content": "Analyzing..."}, "done": false}'
            # Worker blocks until explicitly unblocked
            worker_unblock.wait(timeout=10.0)
            yield b'{"message": {"content": "done"}, "done": true, "done_reason": "stop"}'
            worker_finished.set()

        mock_resp = MagicMock()
        mock_resp.iter_lines.side_effect = stream_generator
        mock_resp.close = MagicMock()
        mock_post.return_value = mock_resp

        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Analyze image",
                        "images": [TINY_PNG_B64],
                    }
                ],
                "mode": "auto",
            },
        )
        self.assertEqual(resp.status_code, 200)

        # Trigger abort in a separate thread once streaming worker begins
        def do_abort():
            worker_started.wait(timeout=2.0)
            time.sleep(0.05)
            abort_resp = self.client.post("/api/chat/abort", json={"request_id": req_id})
            self.assertEqual(abort_resp.status_code, 200)
            self.assertEqual(abort_resp.get_json()["status"], "aborted")

        abort_thr = threading.Thread(target=do_abort)
        abort_thr.start()

        # Consume stream in main thread (where request context lives) until cancelled
        chunks = [c.decode("utf-8") if isinstance(c, bytes) else c for c in resp.response]
        abort_thr.join(timeout=3.0)
        full_stream = "".join(chunks)
        self.assertIn("cancelled", full_stream.lower())

        # ── TEST 2 VERIFICATION: Lease preserved while worker alive ───────
        active_leases = self.manager._active_gen_leases.get("llama3.2-vision:latest", set())
        self.assertEqual(len(active_leases), 1, "Vision lease must remain active under worker ownership")
        vision_lease = next(iter(active_leases))
        self.assertTrue(vision_lease.is_active, "Lease must remain active while worker is alive")

        # ── TEST 3 VERIFICATION: Subsequent request times out ─────────────
        with self.assertRaises(ModelLeaseTimeoutError):
            with self.manager.acquire_generation_lease("llama3.2-vision:latest", timeout=0.1):
                pass

        # ── TEST 4 VERIFICATION: Releasable after worker termination ──────
        # Unblock worker thread
        worker_unblock.set()
        worker_finished.wait(timeout=2.0)
        time.sleep(0.1)  # Allow worker thread finally block to execute

        # Verify worker thread has finished and lease is released
        self.assertFalse(vision_lease.is_active, "Lease must transition to RELEASED once worker finishes")
        self.assertEqual(len(self.manager._active_gen_leases.get("llama3.2-vision:latest", set())), 0)

        # Subsequent acquisition now succeeds immediately
        with self.manager.acquire_generation_lease("llama3.2-vision:latest", timeout=0.5) as new_lease:
            self.assertIsNotNone(new_lease)
            self.assertEqual(new_lease.model_name, "llama3.2-vision:latest")

    # ── Test 5: Cloud-Disabled Image Requests Never Invoke Gemini ──────────

    @patch("google.genai.Client")
    @patch("app.get_gemini_api_key", return_value="fake_gemini_key_12345")
    @patch("app.get_available_ollama_models", return_value=["gemma4:12b"])
    def test_05_cloud_disabled_image_requests_never_invoke_gemini(self, mock_models, mock_key, mock_gemini_client):
        """
        When ALLOW_CLOUD_MULTIMODAL is false:
        - Explicit 'gemini' mode with images returns HTTP 400.
        - 'auto' mode with images (and no local vision model) returns HTTP 400.
        - Google Gemini SDK Client is NEVER instantiated.
        """
        os.environ["ALLOW_CLOUD_MULTIMODAL"] = "false"
        self.assertFalse(is_cloud_multimodal_enabled())

        # Case A: Explicit mode='gemini'
        req_id_a = str(uuid.uuid4())
        resp_a = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id_a,
                "messages": [{"role": "user", "content": "Look at this", "images": [TINY_PNG_B64]}],
                "mode": "gemini",
            },
        )
        self.assertEqual(resp_a.status_code, 400)
        self.assertIn("Cloud image processing is disabled", resp_a.get_json()["error"])
        mock_gemini_client.assert_not_called()

        # Case B: Mode='auto' without local vision model
        req_id_b = str(uuid.uuid4())
        resp_b = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id_b,
                "messages": [{"role": "user", "content": "Look at this", "images": [TINY_PNG_B64]}],
                "mode": "auto",
            },
        )
        self.assertEqual(resp_b.status_code, 400)
        self.assertIn("No verified local vision model", resp_b.get_json()["error"])
        mock_gemini_client.assert_not_called()

    # ── Test 6: Text-Only Gemini Requests Still Work ───────────────────────

    @patch("google.genai.Client")
    @patch("app.get_gemini_api_key", return_value="fake_gemini_key_12345")
    @patch("app.get_available_ollama_models", return_value=["gemma4:12b"])
    def test_06_text_only_gemini_requests_still_work(self, mock_models, mock_key, mock_gemini_client):
        """
        When ALLOW_CLOUD_MULTIMODAL is false, text-only requests routed to Gemini
        must function normally without restriction.
        """
        os.environ["ALLOW_CLOUD_MULTIMODAL"] = "false"

        mock_instance = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Paris is the capital of France."
        mock_instance.models.generate_content_stream.return_value = [mock_chunk]
        mock_gemini_client.return_value = mock_instance

        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "mode": "gemini",
            },
        )
        self.assertEqual(resp.status_code, 200)

        chunks = [c.decode("utf-8") if isinstance(c, bytes) else c for c in resp.response]
        full_stream = "".join(chunks)

        self.assertIn('"route": "gemini"', full_stream)
        self.assertIn("Paris is the capital of France.", full_stream)
        self.assertIn('"done": true', full_stream)

    # ── Test 7: Strict Rejection of Incompatible Modes with Images ──────────

    @patch("app.get_available_ollama_models", return_value=["gemma4:12b", "deepseek-coder-v2:16b", "qwen2.5-coder:7b"])
    def test_07_strict_rejection_of_incompatible_modes_with_images(self, mock_models):
        """
        Explicitly selecting 'pipeline', 'deepseek', 'qwen', or text-only 'chat'
        with images attached must return HTTP 400 Bad Request with an explanation.
        """
        for mode in ("pipeline", "deepseek", "qwen", "chat"):
            req_id = str(uuid.uuid4())
            resp = self.client.post(
                "/api/chat",
                json={
                    "request_id": req_id,
                    "messages": [{"role": "user", "content": "Analyze image", "images": [TINY_PNG_B64]}],
                    "mode": mode,
                },
            )
            self.assertEqual(resp.status_code, 400, f"Mode '{mode}' with images must return 400")
            err_msg = resp.get_json()["error"]
            self.assertIn(f"selected mode '{mode}'", err_msg)
            self.assertIn("cannot process image attachments", err_msg)

    # ── Test 8: Image Quota and Format Security Enforcement ─────────────────

    def test_08_image_quotas_and_magic_sniffing(self):
        """
        validate_and_decode_images() must enforce:
        - Image count <= 10.
        - Single image size <= 10 MB raw bytes.
        - Aggregate image size <= 25 MB raw bytes.
        - Rejection of corrupted base64.
        - Rejection of SVG/HTML (magic bytes).
        """
        # A. Count limit (> 10)
        eleven_images = [TINY_PNG_B64] * 11
        _, err = validate_and_decode_images(eleven_images)
        self.assertIn("maximum allowed is 10", err)

        # B. Corrupted base64
        _, err = validate_and_decode_images(["not_valid_base64!!!"])
        self.assertIn("corrupted or invalid base64", err)

        # C. Non-image payload (SVG/HTML disguised)
        fake_svg_b64 = base64.b64encode(b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>").decode("utf-8")
        _, err = validate_and_decode_images([fake_svg_b64])
        self.assertIn("unsupported format or corrupted header", err)

        # D. Single image size > 10 MB
        # 11 MB of zeros with PNG header
        huge_raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)
        huge_b64 = base64.b64encode(huge_raw).decode("utf-8")
        _, err = validate_and_decode_images([huge_b64])
        self.assertIn("exceeds 10 MB limit", err)

        # E. Aggregate payload > 25 MB (3 x 9 MB = 27 MB)
        nine_mb_raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * (9 * 1024 * 1024)
        nine_mb_b64 = base64.b64encode(nine_mb_raw).decode("utf-8")
        _, err = validate_and_decode_images([nine_mb_b64, nine_mb_b64, nine_mb_b64])
        self.assertIn("exceeds 25 MB aggregate limit", err)

        # F. Valid image succeeds
        validated, err = validate_and_decode_images([TINY_PNG_B64])
        self.assertIsNone(err)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["mime_type"], "image/png")
        self.assertEqual(validated[0]["size"], len(TINY_PNG_BYTES))

    # ── Test 9: /api/upload Endpoint Validation ────────────────────────────

    def test_09_upload_endpoint_validation(self):
        """
        /api/upload must:
        - Return base64 and detected mime_type for valid image.
        - Reject disallowed extension.
        - Reject file with valid extension but invalid magic bytes (e.g. SVG named evil.png).
        - Reject file > 10 MB.
        """
        import io

        # A. Valid upload
        buf_a = io.BytesIO(TINY_PNG_BYTES)
        data = {"image": (buf_a, "test.png")}
        resp = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertEqual(json_data["base64"], TINY_PNG_B64)
        self.assertEqual(json_data["mime_type"], "image/png")
        resp.close()
        buf_a.close()

        # B. Invalid extension
        buf_b = io.BytesIO(b"hello")
        data_bad_ext = {"image": (buf_b, "file.txt")}
        resp = self.client.post("/api/upload", data=data_bad_ext, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported format", resp.get_json()["error"])
        resp.close()
        buf_b.close()

        # C. Disguised SVG named image.png (magic bytes mismatch)
        buf_c = io.BytesIO(b"<svg>script</svg>")
        data_fake = {"image": (buf_c, "image.png")}
        resp = self.client.post("/api/upload", data=data_fake, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported image format or corrupted header", resp.get_json()["error"])
        resp.close()
        buf_c.close()

        # D. Size > 10 MB
        oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)
        buf_d = io.BytesIO(oversized)
        data_large = {"image": (buf_d, "large.png")}
        resp = self.client.post("/api/upload", data=data_large, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("exceeds 10 MB limit", resp.get_json()["error"])
        resp.close()
        buf_d.close()

    # ── Test 10: Settings Persistence for ALLOW_CLOUD_MULTIMODAL ──────────

    def test_10_settings_persistence(self):
        """
        GET /api/settings reports allow_cloud_multimodal.
        POST /api/settings updates allow_cloud_multimodal and persists to .env.
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".env")
        with open(fd, "w", encoding="utf-8") as f:
            f.write("# Initial\n")

        try:
            with patch("app.ENV_FILE", tmp_path):
                # Update setting to True
                resp = self.client.post("/api/settings", json={"allow_cloud_multimodal": True})
                self.assertEqual(resp.status_code, 200)
                resp.close()

                # Verify GET returns True
                get_resp = self.client.get("/api/settings")
                self.assertEqual(get_resp.status_code, 200)
                self.assertTrue(get_resp.get_json()["allow_cloud_multimodal"])
                get_resp.close()

                # Verify file content
                with open(tmp_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertIn("ALLOW_CLOUD_MULTIMODAL=true", content)

                # Update setting to False
                resp2 = self.client.post("/api/settings", json={"allow_cloud_multimodal": False})
                self.assertEqual(resp2.status_code, 200)
                resp2.close()

                get_resp2 = self.client.get("/api/settings")
                self.assertEqual(get_resp2.status_code, 200)
                self.assertFalse(get_resp2.get_json()["allow_cloud_multimodal"])
                get_resp2.close()

                with open(tmp_path, "r", encoding="utf-8") as f:
                    content2 = f.read()
                self.assertIn("ALLOW_CLOUD_MULTIMODAL=false", content2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
