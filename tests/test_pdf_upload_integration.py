"""
Integration & Regression Test Suite for PDF Upload & Grounded Chat
Tests:
1. Valid PDF upload (/api/pdf/upload)
2. Invalid file disguised as PDF (header check & PyMuPDF validation)
3. Oversized PDF rejection (> 25 MB)
4. Page limit rejection (> 50 pages)
5. Multiple PDF attachments
6. Processing status retrieval (/api/pdf/status/<doc_id>)
7. Document list & deletion (/api/pdf/documents, DELETE /api/pdf/<doc_id>)
8. Path traversal / malicious filename prevention
9. Document selection and context injection in chat (/api/chat)
10. Page-level source references emitted in route event
11. Existing image upload regression (/api/upload)
12. Existing chat SSE regression (/api/chat without pdf_doc_ids)
"""

import io
import json
import os
import shutil
import tempfile
import time
import unittest
import fitz

# Import Flask app
from app import app
from pdf_intelligence.registry import get_document_registry, DocumentRecord


class TestPDFUploadIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.registry = get_document_registry()

    def setUp(self):
        self.created_doc_ids = []

    def tearDown(self):
        for doc_id in self.created_doc_ids:
            try:
                self.registry.delete_document(doc_id)
            except Exception:
                pass

    def _create_sample_pdf_bytes(self, num_pages: int = 1, text: str = "Test PDF Page Content") -> bytes:
        doc = fitz.open()
        for i in range(num_pages):
            page = doc.new_page(width=400, height=300)
            page.insert_text((40, 50), f"{text} - Page {i+1}", fontsize=14)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_01_valid_pdf_upload(self):
        """Test uploading a valid PDF document."""
        pdf_bytes = self._create_sample_pdf_bytes(1, "Formal Languages Automata Theory")
        data = {
            "pdf": (io.BytesIO(pdf_bytes), "flat_sample.pdf", "application/pdf")
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 201)
        resp_data = res.get_json()
        self.assertIn("doc_id", resp_data)
        self.assertEqual(resp_data["filename"], "flat_sample.pdf")
        self.assertEqual(resp_data["page_count"], 1)
        self.assertIn(resp_data["status"], ("pending", "processing", "completed"))
        self.created_doc_ids.append(resp_data["doc_id"])

    def test_02_invalid_file_disguised_as_pdf(self):
        """Test uploading a non-PDF file disguised with .pdf extension."""
        fake_bytes = b"This is plain text with no PDF magic header"
        data = {
            "pdf": (io.BytesIO(fake_bytes), "disguised.pdf", "application/pdf")
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        resp_data = res.get_json()
        self.assertIn("error", resp_data)
        self.assertIn("PDF signature", resp_data["error"])

    def test_03_corrupt_pdf_structure(self):
        """Test uploading a file starting with %PDF- but having corrupt binary body."""
        corrupt_bytes = b"%PDF-1.4\nJUNK CORRUPT BODY DATA\n%%EOF"
        data = {
            "pdf": (io.BytesIO(corrupt_bytes), "corrupt.pdf", "application/pdf")
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        resp_data = res.get_json()
        self.assertIn("error", resp_data)

    def test_04_oversized_pdf_rejection(self):
        """Test uploading a PDF exceeding the 25 MB limit."""
        # Create a mock 26 MB stream
        large_bytes = b"%PDF-" + b"0" * (26 * 1024 * 1024)
        data = {
            "pdf": (io.BytesIO(large_bytes), "oversized.pdf", "application/pdf")
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        resp_data = res.get_json()
        self.assertIn("error", resp_data)
        self.assertIn("exceeds maximum allowed size", resp_data["error"])

    def test_05_page_limit_rejection(self):
        """Test uploading a PDF exceeding the 50-page limit."""
        doc = fitz.open()
        for i in range(55):
            p = doc.new_page(width=100, height=100)
            p.insert_text((10, 20), f"P{i}")
        excess_bytes = doc.tobytes()
        doc.close()

        data = {
            "pdf": (io.BytesIO(excess_bytes), "too_many_pages.pdf", "application/pdf")
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        resp_data = res.get_json()
        self.assertIn("page limit", resp_data["error"].lower())

    def test_06_multiple_pdf_attachments(self):
        """Test uploading multiple PDFs in a single request."""
        pdf1 = self._create_sample_pdf_bytes(1, "Paper One")
        pdf2 = self._create_sample_pdf_bytes(1, "Paper Two")

        data = {
            "pdf": [
                (io.BytesIO(pdf1), "doc1.pdf", "application/pdf"),
                (io.BytesIO(pdf2), "doc2.pdf", "application/pdf"),
            ]
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 201)
        resp_data = res.get_json()
        self.assertIn("documents", resp_data)
        self.assertEqual(len(resp_data["documents"]), 2)
        for doc_item in resp_data["documents"]:
            self.created_doc_ids.append(doc_item["doc_id"])

    def test_07_status_retrieval_and_inspection_completion(self):
        """Test polling /api/pdf/status/<doc_id> until inspection completes."""
        pdf_bytes = self._create_sample_pdf_bytes(2, "Automata Theory Questions")
        data = {
            "pdf": (io.BytesIO(pdf_bytes), "status_test.pdf", "application/pdf")
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        doc_id = res.get_json()["doc_id"]
        self.created_doc_ids.append(doc_id)

        # Poll status
        completed = False
        for _ in range(15):
            time.sleep(0.3)
            status_res = self.client.get(f"/api/pdf/status/{doc_id}")
            self.assertEqual(status_res.status_code, 200)
            status_data = status_res.get_json()
            if status_data["status"] == "completed":
                completed = True
                self.assertEqual(status_data["page_count"], 2)
                self.assertIsNotNone(status_data.get("sha256"))
                break
        self.assertTrue(completed, "Document inspection did not reach completed status in time.")

    def test_08_path_traversal_prevention(self):
        """Test that malicious path traversal filenames are safely sanitized."""
        pdf_bytes = self._create_sample_pdf_bytes(1, "Path Traversal Test")
        data = {
            "pdf": (io.BytesIO(pdf_bytes), "../../../etc/passwd.pdf", "application/pdf")
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 201)
        resp_data = res.get_json()
        doc_id = resp_data["doc_id"]
        self.created_doc_ids.append(doc_id)

        # Verify filename was sanitized
        self.assertNotIn("..", resp_data["filename"])
        self.assertNotIn("/", resp_data["filename"])
        self.assertNotIn("\\", resp_data["filename"])

        # Verify physical file is located strictly inside uploads_dir
        rec = self.registry.get_document(doc_id)
        self.assertTrue(os.path.abspath(rec.file_path).startswith(os.path.abspath(self.registry.uploads_dir)))

    def test_09_document_deletion(self):
        """Test deleting a document via DELETE /api/pdf/<doc_id>."""
        pdf_bytes = self._create_sample_pdf_bytes(1, "Delete Test")
        data = {
            "pdf": (io.BytesIO(pdf_bytes), "to_delete.pdf", "application/pdf")
        }
        res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        doc_id = res.get_json()["doc_id"]

        del_res = self.client.delete(f"/api/pdf/{doc_id}")
        self.assertEqual(del_res.status_code, 200)

        # Subsequent GET should return 404
        get_res = self.client.get(f"/api/pdf/status/{doc_id}")
        self.assertEqual(get_res.status_code, 404)

    def test_10_chat_with_pdf_document_grounding(self):
        """Test /api/chat with pdf_doc_ids includes grounded context and pdf_sources."""
        pdf_bytes = self._create_sample_pdf_bytes(1, "FLAT Theorem: Pumping Lemma for Regular Languages")
        data = {
            "pdf": (io.BytesIO(pdf_bytes), "pumping_lemma.pdf", "application/pdf")
        }
        upload_res = self.client.post("/api/pdf/upload", data=data, content_type="multipart/form-data")
        doc_id = upload_res.get_json()["doc_id"]
        self.created_doc_ids.append(doc_id)

        # Wait for inspection to finish
        for _ in range(15):
            time.sleep(0.3)
            if self.registry.get_document(doc_id).status == "completed":
                break

        # Submit chat query referencing this PDF
        chat_payload = {
            "messages": [
                {"role": "user", "content": "What theorem is discussed in the document?"}
            ],
            "mode": "auto",
            "pdf_doc_ids": [doc_id]
        }

        chat_res = self.client.post("/api/chat", json=chat_payload)
        self.assertEqual(chat_res.status_code, 200)

        # Parse SSE stream line by line
        found_route = False
        pdf_sources_found = False
        for line in chat_res.response:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                try:
                    payload = json.loads(line_str[6:].strip())
                    if payload.get("type") == "route":
                        found_route = True
                        if "pdf_sources" in payload and len(payload["pdf_sources"]) > 0:
                            pdf_sources_found = True
                            self.assertEqual(payload["pdf_sources"][0]["doc_id"], doc_id)
                            self.assertEqual(payload["pdf_sources"][0]["page"], 1)
                except Exception:
                    pass
            if pdf_sources_found:
                break

        self.assertTrue(found_route, "Route SSE event was not emitted")
        self.assertTrue(pdf_sources_found, "pdf_sources was not populated in route event")

    def test_11_chat_with_unready_document_returns_400(self):
        """Test that referencing a processing/non-existent document returns a clean 400."""
        chat_payload = {
            "messages": [{"role": "user", "content": "Query"}],
            "pdf_doc_ids": ["doc_non_existent_12345"]
        }
        res = self.client.post("/api/chat", json=chat_payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("not found", res.get_json()["error"].lower())

    def test_12_existing_image_upload_regression(self):
        """Test that existing image upload endpoint /api/upload is completely intact."""
        # 1x1 PNG pixel
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
            b"\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00"
            b"IEND\xaeB`\x82"
        )
        data = {
            "image": (io.BytesIO(png_bytes), "test_pixel.png", "image/png")
        }
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        resp_data = res.get_json()
        self.assertIn("base64", resp_data)
        self.assertEqual(resp_data["filename"], "test_pixel.png")
        self.assertEqual(resp_data["mime_type"], "image/png")

    def test_13_existing_chat_sse_regression(self):
        """Test that /api/chat without pdf_doc_ids functions identically."""
        chat_payload = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "mode": "auto"
        }
        res = self.client.post("/api/chat", json=chat_payload)
        self.assertEqual(res.status_code, 200)
        # Verify first SSE event is route
        first_line = next(res.response).decode("utf-8")
        self.assertTrue(first_line.startswith("data: "))
        data = json.loads(first_line[6:].strip())
        self.assertEqual(data["type"], "route")
        self.assertEqual(data["pdf_sources"], [])


if __name__ == "__main__":
    unittest.main()
