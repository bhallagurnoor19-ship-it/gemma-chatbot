"""
Deterministic 5-State Page Triage Classifier
Classifies PDF pages into:
- DIGITAL_TEXT: High native character count, low raster image coverage.
- SCANNED_IMAGE: Low native character count, high raster image coverage.
- MIXED_CONTENT: High native character count AND high raster image coverage.
- INTENTIONALLY_BLANK: Zero or minimal text matching blank statement, negligible image coverage.
- UNCERTAIN: Ambiguous cases (e.g. low text, low image, not meeting blank criteria).
"""

import enum
import re
from typing import List, Tuple
import fitz

from pdf_intelligence.config import (
    CHAR_COUNT_MIN,
    IMAGE_COVERAGE_THRESHOLD,
    BLANK_IMAGE_COVERAGE_MAX,
    INTENTIONALLY_BLANK_MAX_CHARS,
    INTENTIONALLY_BLANK_REGEX,
)


class PageDisposition(str, enum.Enum):
    DIGITAL_TEXT = "DIGITAL_TEXT"
    SCANNED_IMAGE = "SCANNED_IMAGE"
    MIXED_CONTENT = "MIXED_CONTENT"
    INTENTIONALLY_BLANK = "INTENTIONALLY_BLANK"
    UNCERTAIN = "UNCERTAIN"


def calculate_rectangles_union_area(rects: List[fitz.Rect], clip_rect: fitz.Rect) -> float:
    """
    Computes the exact 2D union area of multiple rectangles clipped to clip_rect.
    Uses coordinate-compression sweep-line algorithm to avoid double-counting
    overlapping images and exclude out-of-page image bleed.
    """
    valid_rects = []
    for r in rects:
        clipped = r & clip_rect  # intersection with page bounds
        if not clipped.is_empty and clipped.width > 0 and clipped.height > 0:
            valid_rects.append(clipped)

    if not valid_rects:
        return 0.0

    # Collect unique sorted x-coordinates
    x_coords = sorted(list(set(
        [r.x0 for r in valid_rects] + [r.x1 for r in valid_rects]
    )))

    total_union_area = 0.0

    for i in range(len(x_coords) - 1):
        x_left = x_coords[i]
        x_right = x_coords[i + 1]
        dx = x_right - x_left
        if dx <= 0:
            continue

        # Find all y-intervals active in this x-strip
        y_intervals: List[Tuple[float, float]] = []
        for r in valid_rects:
            if r.x0 <= x_left and r.x1 >= x_right:
                y_intervals.append((r.y0, r.y1))

        if not y_intervals:
            continue

        # Merge overlapping y-intervals
        y_intervals.sort(key=lambda item: item[0])
        merged_y: List[Tuple[float, float]] = []
        cur_start, cur_end = y_intervals[0]

        for s, e in y_intervals[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                merged_y.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged_y.append((cur_start, cur_end))

        # Sum height of merged intervals multiplied by strip width
        strip_height = sum(e - s for s, e in merged_y)
        total_union_area += dx * strip_height

    return float(total_union_area)


def classify_page(
    native_text: str,
    image_rects: List[fitz.Rect],
    page_rect: fitz.Rect,
) -> Tuple[PageDisposition, float, List[str]]:
    """
    Exhaustively classifies a PDF page into exactly one of 5 dispositions.
    Returns: (disposition, image_coverage_ratio, reasons)
    """
    reasons: List[str] = []
    page_area = float(page_rect.width * page_rect.height)
    if page_area <= 0:
        return PageDisposition.UNCERTAIN, 0.0, ["Invalid or zero page area"]

    # 1. Image area & coverage calculation
    union_image_area = calculate_rectangles_union_area(image_rects, page_rect)
    image_coverage_ratio = min(1.0, max(0.0, union_image_area / page_area))

    clean_text = native_text.strip()
    char_count = len(clean_text)

    reasons.append(
        f"Native text chars: {char_count} (threshold: {CHAR_COUNT_MIN})"
    )
    reasons.append(
        f"Image count: {len(image_rects)}, Union coverage: {image_coverage_ratio:.2%} "
        f"(threshold: {IMAGE_COVERAGE_THRESHOLD:.0%})"
    )

    # 2. Check for INTENTIONALLY_BLANK
    has_blank_regex = bool(re.search(INTENTIONALLY_BLANK_REGEX, clean_text))
    is_empty_text = (char_count == 0)
    low_image_for_blank = (image_coverage_ratio < BLANK_IMAGE_COVERAGE_MAX)

    if low_image_for_blank:
        if is_empty_text:
            reasons.append("Empty page: zero native text characters and negligible image coverage")
            return PageDisposition.INTENTIONALLY_BLANK, image_coverage_ratio, reasons
        elif char_count <= INTENTIONALLY_BLANK_MAX_CHARS and has_blank_regex:
            reasons.append("Explicit intentionally blank statement detected with minimal text")
            return PageDisposition.INTENTIONALLY_BLANK, image_coverage_ratio, reasons

    # 3. Main triage matrix
    has_sufficient_text = (char_count >= CHAR_COUNT_MIN)
    has_high_image = (image_coverage_ratio >= IMAGE_COVERAGE_THRESHOLD)

    if has_sufficient_text and not has_high_image:
        reasons.append("Sufficient native digital text and image coverage below threshold")
        return PageDisposition.DIGITAL_TEXT, image_coverage_ratio, reasons

    if not has_sufficient_text and has_high_image:
        reasons.append("Insufficient native text and high raster image coverage (scanned document)")
        return PageDisposition.SCANNED_IMAGE, image_coverage_ratio, reasons

    if has_sufficient_text and has_high_image:
        reasons.append("Both substantial native digital text and high raster image coverage present")
        return PageDisposition.MIXED_CONTENT, image_coverage_ratio, reasons

    # 4. Fallback: UNCERTAIN
    reasons.append(
        "Low native text density (<50 chars) and low image coverage (<40%), "
        "not qualifying as intentionally blank"
    )
    return PageDisposition.UNCERTAIN, image_coverage_ratio, reasons
