"""
Local PyMuPDF OCR Engine Wrapper
Utilizes PyMuPDF's built-in Tesseract C-runtime (fitz.Pixmap.pdfocr_tobytes)
with project-local tessdata language models.
"""

import os
import time
from typing import Optional, Tuple
import fitz

from pdf_intelligence.config import (
    DEFAULT_TESSDATA_DIR,
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_OCR_DPI,
)


class OCREngine:
    def __init__(self, tessdata_dir: Optional[str] = None):
        self.tessdata_dir = tessdata_dir or DEFAULT_TESSDATA_DIR
        self._check_availability()

    def _check_availability(self) -> None:
        """Verifies if the required tessdata files exist on disk."""
        lang_file = os.path.join(self.tessdata_dir, f"{DEFAULT_OCR_LANGUAGE}.traineddata")
        if os.path.isdir(self.tessdata_dir) and os.path.isfile(lang_file):
            self.is_available = True
            self.status_message = f"Tessdata found at {self.tessdata_dir}"
        else:
            self.is_available = False
            self.status_message = (
                f"Tessdata missing: '{lang_file}' not found. "
                f"OCR execution is disabled until traineddata is provided."
            )

    def get_engine_info(self) -> dict:
        return {
            "is_available": self.is_available,
            "tessdata_dir": self.tessdata_dir,
            "status": self.status_message,
            "pymupdf_version": getattr(fitz, "__version__", "unknown"),
            "mupdf_version": getattr(fitz, "version", ("unknown",))[1]
            if hasattr(fitz, "version")
            else "unknown",
        }

    def ocr_page(
        self,
        page: fitz.Page,
        dpi: int = DEFAULT_OCR_DPI,
        language: str = DEFAULT_OCR_LANGUAGE,
    ) -> Tuple[str, float, Optional[str]]:
        """
        Renders a PyMuPDF page to a pixmap and performs local OCR.
        Returns: (extracted_ocr_text, duration_ms, error_message)
        """
        if not self.is_available:
            return "", 0.0, f"OCR unavailable: {self.status_message}"

        t0 = time.perf_counter()
        try:
            # 1. Render page to raster pixmap
            pix = page.get_pixmap(dpi=dpi)
            
            # 2. Invoke local PyMuPDF OCR C-engine
            ocr_pdf_bytes = pix.pdfocr_tobytes(
                language=language,
                tessdata=self.tessdata_dir,
            )

            # 3. Open synthesized 1-page PDF and extract text
            with fitz.open("pdf", ocr_pdf_bytes) as ocr_doc:
                if len(ocr_doc) > 0:
                    ocr_text = ocr_doc[0].get_text("text")
                else:
                    ocr_text = ""

            duration_ms = (time.perf_counter() - t0) * 1000.0
            return ocr_text, duration_ms, None

        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return "", duration_ms, f"OCR execution failed: {type(e).__name__}: {str(e)}"
