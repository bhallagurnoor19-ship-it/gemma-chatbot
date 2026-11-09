"""
Automated Test Suite for PDF Intelligence Subsystem — Milestone 1
Tests:
- DIGITAL_TEXT page classification and extraction
- SCANNED_IMAGE page classification and OCR extraction
- MIXED_CONTENT page classification and dual provenance extraction
- INTENTIONALLY_BLANK page classification (empty & blank statement)
- UNCERTAIN disposition fallback for low-density/unmatched cases
- Corrupted/unreadable/missing PDF error handling
- OCR-unavailable fallback when tessdata is missing
- Real FLAT exam PDF validation
"""

import os
import shutil
import tempfile
import unittest
import fitz

from pdf_intelligence.classifier import (
    PageDisposition,
    calculate_rectangles_union_area,
    classify_page,
)
from pdf_intelligence.config import (
    CHAR_COUNT_MIN,
    IMAGE_COVERAGE_THRESHOLD,
    DEFAULT_TESSDATA_DIR,
)
from pdf_intelligence.inspector import (
    DocumentInspectionResult,
    PDFInspector,
    compute_file_sha256,
)
from pdf_intelligence.ocr_engine import OCREngine


class TestPDFIntelligenceMilestone1(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_image_bytes(self, width=200, height=200, color=(100, 150, 200)) -> bytes:
        """Creates a simple raw pixmap PNG in memory."""
        pix = fitz.Pixmap(fitz.csRGB, (0, 0, width, height), 0)
        pix.set_rect(pix.irect, color)
        return pix.tobytes("png")

    def test_rectangles_union_area_calculation(self):
        """Verifies exact union area calculation with overlaps and boundary clipping."""
        clip_rect = fitz.Rect(0, 0, 100, 100)
        
        # Test 1: Single non-overlapping rect
        r1 = fitz.Rect(10, 10, 50, 50)  # area = 40 * 40 = 1600
        area = calculate_rectangles_union_area([r1], clip_rect)
        self.assertAlmostEqual(area, 1600.0, places=2)

        # Test 2: Overlapping rects
        r2 = fitz.Rect(30, 30, 70, 70)  # area = 40 * 40 = 1600, overlap = 20 * 20 = 400
        # Expected union = 1600 + 1600 - 400 = 2800
        area_overlap = calculate_rectangles_union_area([r1, r2], clip_rect)
        self.assertAlmostEqual(area_overlap, 2800.0, places=2)

        # Test 3: Rect exceeding page boundary (must be clipped)
        r_bleed = fitz.Rect(80, 80, 150, 150)  # clipped to (80, 80, 100, 100) -> area 400
        area_bleed = calculate_rectangles_union_area([r_bleed], clip_rect)
        self.assertAlmostEqual(area_bleed, 400.0, places=2)

    def test_page_classifier_states(self):
        """Verifies all 5 triage states and their transparent reasons."""
        page_rect = fitz.Rect(0, 0, 500, 800)  # Area = 400,000

        # 1. DIGITAL_TEXT
        disp, cov, reasons = classify_page(
            native_text="This is a long sample text with more than fifty characters to qualify as digital.",
            image_rects=[],
            page_rect=page_rect,
        )
        self.assertEqual(disp, PageDisposition.DIGITAL_TEXT)
        self.assertAlmostEqual(cov, 0.0)
        self.assertTrue(any("Sufficient native digital text" in r for r in reasons))

        # 2. SCANNED_IMAGE
        full_img = [fitz.Rect(0, 0, 500, 800)]
        disp, cov, reasons = classify_page(
            native_text="Watermark",  # < 50 chars
            image_rects=full_img,
            page_rect=page_rect,
        )
        self.assertEqual(disp, PageDisposition.SCANNED_IMAGE)
        self.assertAlmostEqual(cov, 1.0)
        self.assertTrue(any("scanned document" in r for r in reasons))

        # 3. MIXED_CONTENT
        disp, cov, reasons = classify_page(
            native_text="This is a substantial amount of native text that clearly exceeds fifty characters.",
            image_rects=[fitz.Rect(0, 0, 500, 500)],  # Area 250,000 / 400,000 = 62.5% (>40%)
            page_rect=page_rect,
        )
        self.assertEqual(disp, PageDisposition.MIXED_CONTENT)
        self.assertGreaterEqual(cov, IMAGE_COVERAGE_THRESHOLD)

        # 4. INTENTIONALLY_BLANK (empty)
        disp, cov, reasons = classify_page(
            native_text="",
            image_rects=[],
            page_rect=page_rect,
        )
        self.assertEqual(disp, PageDisposition.INTENTIONALLY_BLANK)

        # 4b. INTENTIONALLY_BLANK (statement)
        disp, cov, reasons = classify_page(
            native_text="This page is intentionally left blank.",
            image_rects=[],
            page_rect=page_rect,
        )
        self.assertEqual(disp, PageDisposition.INTENTIONALLY_BLANK)
        self.assertTrue(any("intentionally blank statement" in r for r in reasons))

        # 5. UNCERTAIN
        disp, cov, reasons = classify_page(
            native_text="Draft v1.0",  # < 50 chars, no blank statement, no high image
            image_rects=[fitz.Rect(10, 10, 50, 50)],  # very small image
            page_rect=page_rect,
        )
        self.assertEqual(disp, PageDisposition.UNCERTAIN)
        self.assertTrue(any("Low native text density" in r for r in reasons))

    def test_inspection_pipeline_digital_pdf(self):
        """Tests ingestion and inspection of a purely digital PDF document."""
        pdf_path = os.path.join(self.temp_dir, "digital_sample.pdf")
        doc = fitz.open()
        page = doc.new_page(width=600, height=800)
        sample_text = (
            "Formal Languages and Automata Theory. Chapter 1: Introduction to Deterministic Finite Automata. "
            "A DFA consists of a 5-tuple (Q, Sigma, delta, q0, F). It processes strings symbol by symbol."
        )
        page.insert_text((50, 100), sample_text, fontsize=12)
        doc.save(pdf_path)
        doc.close()

        inspector = PDFInspector()
        result = inspector.inspect_document(pdf_path)

        self.assertEqual(result.page_count, 1)
        self.assertFalse(result.is_encrypted)
        self.assertEqual(len(result.sha256), 64)
        self.assertEqual(result.page_classifications_summary["DIGITAL_TEXT"], 1)

        p0 = result.pages[0]
        self.assertEqual(p0.classification, "DIGITAL_TEXT")
        self.assertEqual(p0.provenance, "native")
        self.assertFalse(p0.ocr_executed)
        self.assertIn("Deterministic Finite Automata", p0.extracted_text)

    def test_inspection_pipeline_scanned_pdf_with_ocr(self):
        """Tests ingestion, triage, and local OCR execution on a scanned page."""
        pdf_path = os.path.join(self.temp_dir, "scanned_sample.pdf")
        doc = fitz.open()
        
        # Synthesize an image containing rendered text
        img_doc = fitz.open()
        img_page = img_doc.new_page(width=400, height=200)
        img_page.insert_text((20, 60), "REGULAR EXPRESSION PARSING TEST", fontsize=16)
        pix = img_page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        img_doc.close()

        # Insert as full-page image in target PDF
        page = doc.new_page(width=400, height=200)
        page.insert_image(fitz.Rect(0, 0, 400, 200), stream=img_bytes)
        doc.save(pdf_path)
        doc.close()

        inspector = PDFInspector()
        result = inspector.inspect_document(pdf_path)

        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.page_classifications_summary["SCANNED_IMAGE"], 1)

        p0 = result.pages[0]
        self.assertEqual(p0.classification, "SCANNED_IMAGE")
        self.assertTrue(p0.ocr_executed)
        self.assertEqual(p0.provenance, "ocr")
        self.assertGreater(len(p0.extracted_text), 0)
        # Verify OCR detected words from the image
        self.assertTrue("PARSING" in p0.extracted_text.upper() or "EXPRESSION" in p0.extracted_text.upper())

    def test_ocr_unavailable_fallback(self):
        """Verifies that missing tessdata triggers graceful fallback without crashing."""
        pdf_path = os.path.join(self.temp_dir, "fallback_sample.pdf")
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        page.insert_image(
            fitz.Rect(0, 0, 400, 200),
            stream=self._create_sample_image_bytes(400, 200),
        )
        doc.save(pdf_path)
        doc.close()

        # Point inspector to non-existent tessdata directory
        fake_tessdata = os.path.join(self.temp_dir, "non_existent_tessdata")
        inspector = PDFInspector(tessdata_dir=fake_tessdata)

        self.assertFalse(inspector.ocr_engine.is_available)
        result = inspector.inspect_document(pdf_path)

        p0 = result.pages[0]
        self.assertEqual(p0.classification, "SCANNED_IMAGE")
        self.assertFalse(p0.ocr_executed)
        self.assertIsNotNone(p0.ocr_error)
        self.assertTrue(
            any(k in p0.ocr_error.lower() for k in ("missing", "disabled", "unavailable"))
        )

    def test_corrupted_or_missing_file_handling(self):
        """Verifies proper error handling on missing or corrupt files."""
        inspector = PDFInspector()
        
        # Missing file
        with self.assertRaises(FileNotFoundError):
            inspector.inspect_document("C:/non_existent_path_file.pdf")

        # Corrupt file
        corrupt_path = os.path.join(self.temp_dir, "corrupt.pdf")
        with open(corrupt_path, "wb") as f:
            f.write(b"%PDF-1.4 completely corrupted binary junk that cannot be parsed")
        
        with self.assertRaises(Exception):
            inspector.inspect_document(corrupt_path)

    def test_real_flat_pdf_if_available(self):
        """Verifies execution against real FLAT exam papers discovered on disk."""
        real_flat_path = r"C:\Users\Gurnoor\Downloads\FLAT-(5th)-Dec2025-2.pdf"
        if not os.path.exists(real_flat_path):
            self.skipTest(f"Real FLAT paper not found at {real_flat_path}")

        inspector = PDFInspector()
        result = inspector.inspect_document(real_flat_path)

        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.page_classifications_summary["SCANNED_IMAGE"], 2)
        for page in result.pages:
            self.assertEqual(page.classification, "SCANNED_IMAGE")
            self.assertEqual(page.provenance, "ocr")
            self.assertGreater(len(page.extracted_text), 100)
            self.assertTrue(page.ocr_executed)


if __name__ == "__main__":
    unittest.main()
