"""
Centralized Configuration & Security Rules for Unified RAG Subsystem
- Manages paths, collection identifiers, embedding configurations, and security exclusions.
"""

import fnmatch
import os
from pathlib import Path
from typing import List, Set

# ── Base Directory Paths ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
MANIFESTS_DIR = os.path.join(DATA_DIR, "manifests")
INDEX_MANIFEST_PATH = os.path.join(MANIFESTS_DIR, "index_manifest.json")
RAGIGNORE_PATH = os.path.join(PROJECT_ROOT, ".ragignore")

# ── Collection Identifiers (Logically and Physically Isolated) ────────────────
DOCUMENT_COLLECTION_NAME = "documents_collection"
CODEBASE_COLLECTION_NAME = "codebase_collection"
CONVERSATION_COLLECTION_NAME = "conversation_memory_collection"

# ── Embedding Model Configuration ─────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
PRIMARY_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "embeddinggemma")
FALLBACK_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_EMBEDDING_POLICY = os.environ.get("OLLAMA_EMBEDDING_POLICY", "adaptive")
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_TIMEOUT_SECONDS = 60

# ── Document & Code Chunking Parameters ───────────────────────────────────────
DOC_CHUNK_SIZE = 750
DOC_CHUNK_OVERLAP = 120

CODE_MIN_CHUNK_LINES = 3
CODE_MAX_CHUNK_LINES = 120

# ── Retrieval Parameters ──────────────────────────────────────────────────────
DEFAULT_TOP_K = 4
MAX_CONTEXT_TOKENS = 2000
SIMILARITY_DISTANCE_THRESHOLD = 0.70  # Cosine distance cutoff
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per file limit

# ── Conversational Memory Parameters ──────────────────────────────────────────
MEMORY_TOP_K = int(os.environ.get("RAG_MEMORY_TOP_K", "4"))
MEMORY_DISTANCE_THRESHOLD = float(os.environ.get("RAG_MEMORY_DISTANCE_THRESHOLD", "0.65"))
RECENT_CONTEXT_TURNS = int(os.environ.get("RAG_RECENT_CONTEXT_TURNS", "2"))
RECENT_CONTEXT_MAX_CHARS = int(os.environ.get("RAG_RECENT_CONTEXT_MAX_CHARS", "3500"))

# ── Supported File Extensions ─────────────────────────────────────────────────
SUPPORTED_DOC_EXTENSIONS: Set[str] = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".html",
    ".htm",
}

SUPPORTED_CODE_EXTENSIONS: Set[str] = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".bat",
    ".vbs",
    ".sh",
    ".ps1",
    ".sql",
    ".js",
    ".ts",
    ".css",
    ".html",
    ".htm",
}

# ── Source-Aware Ranking Configuration ─────────────────────────────────────────
# Configurable weights for source-aware ranking in Codebase RAG
DEFAULT_RANKING_CONFIG = {
    "enabled": True,
    "impl_query": {
        "impl_boost": 1.08,
        "test_multiplier": 0.88,
        "config_boost": 1.00,
        "frontend_boost": 1.00,
    },
    "test_query": {
        "test_boost": 1.15,
        "impl_multiplier": 0.95,
        "config_boost": 1.00,
        "frontend_boost": 1.00,
    },
    "config_query": {
        "config_boost": 1.15,
        "test_multiplier": 0.90,
        "impl_boost": 1.00,
        "frontend_boost": 1.00,
    },
    "frontend_query": {
        "frontend_boost": 1.15,
        "test_multiplier": 0.90,
        "impl_boost": 1.00,
        "config_boost": 1.00,
    },
}

# ── Hardcoded Security Blacklist (Never bypassed, even if .ragignore is altered)
HARDCODED_SECURITY_BLACKLIST: Set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".git",
    "server.log",
    "launcher.log",
}

HARDCODED_EXT_BLACKLIST: Set[str] = {
    ".key",
    ".pem",
    ".crt",
    ".pfx",
    ".secret",
    ".token",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".exe",
    ".dll",
    ".so",
    ".bin",
    ".pyc",
    ".pyo",
}


def load_ragignore_patterns(ragignore_file: str = RAGIGNORE_PATH) -> List[str]:
    """Load exclusion patterns from .ragignore file."""
    patterns = []
    if os.path.exists(ragignore_file):
        try:
            with open(ragignore_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        patterns.append(stripped)
        except Exception:
            pass
    return patterns


def is_path_ignored(target_path: str, base_dir: str = PROJECT_ROOT) -> bool:
    """
    Evaluate whether a file or directory path is excluded from RAG indexing.
    Applies both hardcoded security boundaries and .ragignore wildcard rules.
    """
    abs_target = os.path.abspath(target_path)
    abs_base = os.path.abspath(base_dir)

    try:
        rel_path = os.path.relpath(abs_target, abs_base)
    except ValueError:
        # Cross-drive on Windows
        rel_path = abs_target

    norm_rel = rel_path.replace("\\", "/")
    basename = os.path.basename(abs_target)
    _, ext = os.path.splitext(basename)

    # 1. Hardcoded security checks (Infallible safety net)
    if basename.lower() in HARDCODED_SECURITY_BLACKLIST or basename.startswith(".env"):
        return True

    if ext.lower() in HARDCODED_EXT_BLACKLIST:
        return True

    # Deny files inside internal directories
    parts = norm_rel.split("/")
    if any(p in {".git", "__pycache__", ".app_profile", ".test_profile", "chroma_db", "manifests"} for p in parts):
        return True

    # 2. Dynamic .ragignore pattern matching
    patterns = load_ragignore_patterns()
    for pattern in patterns:
        clean_pat = pattern.rstrip("/").replace("\\", "/")
        is_dir_pattern = pattern.endswith("/")

        # Match against relative path or basename
        if fnmatch.fnmatch(norm_rel, clean_pat) or fnmatch.fnmatch(basename, clean_pat):
            return True

        # Match against prefix directories
        if is_dir_pattern:
            for part in parts:
                if fnmatch.fnmatch(part, clean_pat):
                    return True
        else:
            if fnmatch.fnmatch(norm_rel, f"*{clean_pat}*") or fnmatch.fnmatch(basename, clean_pat):
                return True

    return False


def is_safe_path(target_path: str, base_dir: str = PROJECT_ROOT) -> bool:
    """
    Ensure target path resolves strictly within base_dir and does not escape via symlink or directory traversal.
    """
    try:
        real_target = os.path.realpath(os.path.abspath(target_path))
        real_base = os.path.realpath(os.path.abspath(base_dir))
        common = os.path.commonpath([real_target, real_base])
        return common == real_base
    except Exception:
        return False

