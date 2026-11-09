"""
PDF Intelligence Reporting Utilities
Outputs inspection results into JSON, Markdown summaries, and per-page text files.
"""

import json
import os
from typing import Dict, Optional

from pdf_intelligence.config import DEFAULT_OUTPUT_DIR
from pdf_intelligence.inspector import DocumentInspectionResult


def save_json_report(
    result: DocumentInspectionResult,
    output_dir: Optional[str] = None,
) -> str:
    """Saves complete DocumentInspectionResult as structured JSON."""
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(result.file_name)[0]
    out_path = os.path.join(out_dir, f"{base_name}_inspection.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    return out_path


def save_markdown_report(
    result: DocumentInspectionResult,
    output_dir: Optional[str] = None,
) -> str:
    """Saves human-readable markdown inspection summary."""
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(result.file_name)[0]
    out_path = os.path.join(out_dir, f"{base_name}_summary.md")

    md_lines = [
        f"# PDF Inspection Report: {result.file_name}",
        "",
        "## Document Overview",
        f"- **File Path**: `{result.file_path}`",
        f"- **File Size**: `{result.file_size_bytes:,}` bytes ({result.file_size_bytes / (1024*1024):.2f} MB)",
        f"- **SHA-256**: `{result.sha256}`",
        f"- **Page Count**: {result.page_count}",
        f"- **Encrypted**: {result.is_encrypted}",
        f"- **Total Inspection Time**: {result.total_inspection_duration_ms:.2f} ms",
        f"- **OCR Engine Status**: {'Available' if result.ocr_available else 'Unavailable'} ({result.ocr_engine_info.get('status')})",
        "",
        "## Page Classification Summary",
        "| Disposition | Count |",
        "|---|---|",
    ]

    for disposition, count in result.page_classifications_summary.items():
        md_lines.append(f"| `{disposition}` | {count} |")

    md_lines.extend([
        "",
        "## Page-by-Page Inspection Details",
        "",
        "| Page | Dimensions | Rotation | Native Chars | Images | Image Coverage | Classification | Provenance | Timing (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for p in result.pages:
        md_lines.append(
            f"| {p.display_page_number} | {p.width:.1f} x {p.height:.1f} | {p.rotation}° | "
            f"{p.native_char_count} | {p.image_count} | {p.image_coverage_ratio:.1%} | "
            f"`{p.classification}` | `{p.provenance}` | {p.total_duration_ms:.1f} |"
        )

    md_lines.append("")
    md_lines.append("## Detailed Page Breakdown")

    for p in result.pages:
        md_lines.extend([
            f"### Page {p.display_page_number}",
            f"- **Classification**: `{p.classification}`",
            f"- **Provenance**: `{p.provenance}`",
            f"- **Reasons**:",
        ])
        for r in p.classification_reasons:
            md_lines.append(f"  - {r}")

        if p.ocr_executed:
            md_lines.append(f"- **OCR Time**: {p.ocr_duration_ms:.2f} ms")
        if p.ocr_error:
            md_lines.append(f"- **OCR Error/Warning**: {p.ocr_error}")

        md_lines.append("- **Extracted Text Preview**:")
        preview_lines = p.extracted_text.splitlines()[:10]
        if preview_lines:
            md_lines.append("```text")
            for line in preview_lines:
                md_lines.append(line[:120])
            if len(p.extracted_text.splitlines()) > 10:
                md_lines.append(f"... [{len(p.extracted_text.splitlines()) - 10} more lines]")
            md_lines.append("```")
        else:
            md_lines.append("_No text extracted_")
        md_lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return out_path


def save_text_dumps(
    result: DocumentInspectionResult,
    output_dir: Optional[str] = None,
) -> Dict[int, str]:
    """Saves per-page extracted text dumps."""
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    pages_dir = os.path.join(out_dir, f"{os.path.splitext(result.file_name)[0]}_pages")
    os.makedirs(pages_dir, exist_ok=True)

    saved_files: Dict[int, str] = {}
    for p in result.pages:
        filename = f"page_{p.display_page_number:03d}_{p.provenance.replace('+', '_')}.txt"
        filepath = os.path.join(pages_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            header = [
                f"# Page {p.display_page_number} ({result.file_name})",
                f"# Classification: {p.classification}",
                f"# Provenance: {p.provenance}",
                f"# Native Chars: {p.native_char_count} | Extracted Chars: {len(p.extracted_text)}",
                "=" * 60,
                "",
            ]
            f.write("\n".join(header) + p.extracted_text + "\n")
        saved_files[p.page_number] = filepath

    return saved_files
