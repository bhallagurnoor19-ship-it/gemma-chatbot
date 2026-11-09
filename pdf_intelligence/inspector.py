"""
PDF Inspector Pipeline — Milestone 1
Coordinates document loading, hashing, geometry extraction,
5-state page triage classification, local OCR execution, and structured reporting.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import os
import time
from typing import Any, Dict, List, Optional
import fitz

from pdf_intelligence.classifier import (
    PageDisposition,
    calculate_rectangles_union_area,
    classify_page,
)
from pdf_intelligence.config import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    DEFAULT_TESSDATA_DIR,
)
from pdf_intelligence.ocr_engine import OCREngine


@dataclass
class PageInspectionResult:
    page_number: int              # 0-indexed
    display_page_number: int      # 1-indexed
    width: float
    height: float
    rotation: int
    native_char_count: int
    native_word_count: int
    image_count: int
    image_coverage_ratio: float
    classification: str           # DIGITAL_TEXT, SCANNED_IMAGE, MIXED_CONTENT, INTENTIONALLY_BLANK, UNCERTAIN
    classification_reasons: List[str]
    provenance: str               # "native", "ocr", "native+ocr", "none"
    extracted_text: str
    ocr_executed: bool = False
    ocr_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    ocr_error: Optional[str] = None


@dataclass
class DocumentInspectionResult:
    file_path: str
    file_name: str
    file_size_bytes: int
    sha256: str
    page_count: int
    is_encrypted: bool
    metadata: Dict[str, Any]
    page_classifications_summary: Dict[str, int]
    ocr_available: bool
    ocr_engine_info: Dict[str, Any]
    total_inspection_duration_ms: float
    pages: List[PageInspectionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_sha256(filepath: str, block_size: int = 65536) -> str:
    """Computes the SHA-256 checksum of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


class PDFInspector:
    def __init__(
        self,
        tessdata_dir: Optional[str] = None,
        ocr_dpi: int = DEFAULT_OCR_DPI,
        ocr_language: str = DEFAULT_OCR_LANGUAGE,
    ):
        self.ocr_dpi = ocr_dpi
        self.ocr_language = ocr_language
        self.ocr_engine = OCREngine(tessdata_dir=tessdata_dir)

    def inspect_document(
        self,
        pdf_path: str,
        force_ocr: bool = False,
    ) -> DocumentInspectionResult:
        """
        Inspects an entire PDF document, triages every page,
        executes OCR where appropriate, and produces structured inspection results.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc_t0 = time.perf_counter()
        file_size = os.path.getsize(pdf_path)
        sha256 = compute_file_sha256(pdf_path)
        file_name = os.path.basename(pdf_path)

        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
            is_encrypted = doc.is_encrypted
            doc_metadata = dict(doc.metadata) if doc.metadata else {}

            page_results: List[PageInspectionResult] = []
            class_counts: Dict[str, int] = {
                d.value: 0 for d in PageDisposition
            }

            for page_idx in range(page_count):
                page_t0 = time.perf_counter()
                page = doc[page_idx]

                # Geometry & native text extraction
                page_rect = page.rect
                width = float(page_rect.width)
                height = float(page_rect.height)
                rotation = int(page.rotation)

                native_text = page.get_text("text")
                clean_native = native_text.strip()
                native_chars = len(clean_native)
                native_words = len(clean_native.split()) if native_chars > 0 else 0

                # Extract image bounding boxes
                image_info_list = page.get_images(full=True)
                image_rects: List[fitz.Rect] = []
                for img_info in image_info_list:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    image_rects.extend(rects)

                # Page triage classification
                disposition, coverage_ratio, reasons = classify_page(
                    native_text=native_text,
                    image_rects=image_rects,
                    page_rect=page_rect,
                )
                class_counts[disposition.value] = class_counts.get(disposition.value, 0) + 1

                # Determine OCR execution
                should_run_ocr = force_ocr or (
                    disposition in (
                        PageDisposition.SCANNED_IMAGE,
                        PageDisposition.MIXED_CONTENT,
                    )
                )

                ocr_executed = False
                ocr_duration_ms = 0.0
                ocr_error: Optional[str] = None
                extracted_text = clean_native
                provenance = "native" if native_chars > 0 else "none"

                if should_run_ocr:
                    if self.ocr_engine.is_available:
                        ocr_text, ocr_dur, err = self.ocr_engine.ocr_page(
                            page=page,
                            dpi=self.ocr_dpi,
                            language=self.ocr_language,
                        )
                        ocr_executed = True
                        ocr_duration_ms = ocr_dur

                        if err:
                            ocr_error = err
                            reasons.append(f"OCR warning: {err}")
                        else:
                            clean_ocr = ocr_text.strip()
                            if disposition == PageDisposition.SCANNED_IMAGE:
                                extracted_text = clean_ocr
                                provenance = "ocr" if len(clean_ocr) > 0 else "none"
                            elif disposition == PageDisposition.MIXED_CONTENT:
                                extracted_text = (
                                    clean_native
                                    + "\n\n--- [OCR Content] ---\n"
                                    + clean_ocr
                                )
                                provenance = "native+ocr"
                    else:
                        ocr_error = self.ocr_engine.status_message
                        reasons.append(f"OCR skipped: {self.ocr_engine.status_message}")

                page_dur_ms = (time.perf_counter() - page_t0) * 1000.0

                page_results.append(
                    PageInspectionResult(
                        page_number=page_idx,
                        display_page_number=page_idx + 1,
                        width=width,
                        height=height,
                        rotation=rotation,
                        native_char_count=native_chars,
                        native_word_count=native_words,
                        image_count=len(image_rects),
                        image_coverage_ratio=coverage_ratio,
                        classification=disposition.value,
                        classification_reasons=reasons,
                        provenance=provenance,
                        extracted_text=extracted_text,
                        ocr_executed=ocr_executed,
                        ocr_duration_ms=ocr_duration_ms,
                        total_duration_ms=page_dur_ms,
                        ocr_error=ocr_error,
                    )
                )

        doc_dur_ms = (time.perf_counter() - doc_t0) * 1000.0

        return DocumentInspectionResult(
            file_path=os.path.abspath(pdf_path),
            file_name=file_name,
            file_size_bytes=file_size,
            sha256=sha256,
            page_count=page_count,
            is_encrypted=is_encrypted,
            metadata=doc_metadata,
            page_classifications_summary=class_counts,
            ocr_available=self.ocr_engine.is_available,
            ocr_engine_info=self.ocr_engine.get_engine_info(),
            total_inspection_duration_ms=doc_dur_ms,
            pages=page_results,
        )
