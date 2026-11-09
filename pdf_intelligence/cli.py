"""
CLI Entrypoint for PDF Intelligence Ingestion & Inspection Pipeline
Usage:
    python -m pdf_intelligence.cli <pdf_path> [--output-dir <dir>] [--dpi <dpi>]
"""

import argparse
import sys

from pdf_intelligence.config import DEFAULT_OCR_DPI, DEFAULT_OUTPUT_DIR
from pdf_intelligence.inspector import PDFInspector
from pdf_intelligence.reporter import (
    save_json_report,
    save_markdown_report,
    save_text_dumps,
)


def main():
    parser = argparse.ArgumentParser(
        description="PDF Intelligence Ingestion and Page Inspection Pipeline"
    )
    parser.add_argument("pdf_path", help="Path to the PDF document to inspect")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save inspection reports",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_OCR_DPI,
        help="DPI for rasterization when OCR is required",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR execution on all pages regardless of classification",
    )

    args = parser.parse_args()

    try:
        print(f"[*] Initializing PDF Inspector (OCR DPI: {args.dpi})...")
        inspector = PDFInspector(ocr_dpi=args.dpi)
        print(f"[*] Inspecting document: {args.pdf_path}...")
        result = inspector.inspect_document(args.pdf_path, force_ocr=args.force_ocr)

        print(f"[+] Document inspection completed in {result.total_inspection_duration_ms:.2f} ms")
        print(f"    - Pages: {result.page_count}")
        print(f"    - SHA-256: {result.sha256[:16]}...")
        print(f"    - Classifications: {result.page_classifications_summary}")

        json_path = save_json_report(result, args.output_dir)
        md_path = save_markdown_report(result, args.output_dir)
        text_dumps = save_text_dumps(result, args.output_dir)

        print(f"[+] Outputs saved to: {args.output_dir}")
        print(f"    - JSON: {json_path}")
        print(f"    - Markdown: {md_path}")
        print(f"    - Text dumps: {len(text_dumps)} pages saved")

    except Exception as e:
        print(f"[!] Error inspecting document: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
