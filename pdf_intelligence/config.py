"""
Centralized Configuration for PDF Intelligence Subsystem
"""

import os
from pathlib import Path

# Project root resolution
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pdf_intelligence")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "output")
DEFAULT_TESSDATA_DIR = os.path.join(DATA_DIR, "tessdata")

# Ingestion & Triage Thresholds
CHAR_COUNT_MIN = 50                 # Minimum native characters to consider page text-bearing
IMAGE_COVERAGE_THRESHOLD = 0.40     # >= 40% area covered by images indicates visual/scanned content
BLANK_IMAGE_COVERAGE_MAX = 0.05     # < 5% image area allowed for a blank page candidate
INTENTIONALLY_BLANK_MAX_CHARS = 100  # Max chars allowed if blank regex matches

# Regex for intentionally blank page declarations
INTENTIONALLY_BLANK_REGEX = (
    r"(?i)\b(?:this\s+page\s+(?:is\s+)?(?:intentionally\s+left\s+blank|intentionally\s+blank)|"
    r"intentionally\s+left\s+blank|page\s+intentionally\s+left\s+blank)\b"
)

# OCR Configuration
DEFAULT_OCR_DPI = 150               # Balanced DPI for fast and accurate OCR extraction
DEFAULT_OCR_LANGUAGE = "eng"
OCR_TIMEOUT_SECONDS = 30.0

# Supported Document Extensions
SUPPORTED_EXTENSIONS = {".pdf"}
