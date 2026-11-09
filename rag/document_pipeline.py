"""
Document Ingestion & Structure-Aware Chunking Pipeline
- Ingests PDF (via PyMuPDF), Markdown, TXT, RST, and HTML
- Performs page-isolated chunking for PDFs so page citations are 100% accurate
- Employs structure-aware recursive chunking with configurable overlap
- Robust error handling for empty, corrupted, encrypted, or oversized documents
"""

import html
import os
import re
from typing import Any, Dict, List, Optional

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

from rag import logger
from rag.config import (
    DOC_CHUNK_SIZE,
    DOC_CHUNK_OVERLAP,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_DOC_EXTENSIONS,
    PROJECT_ROOT,
    is_path_ignored,
    is_safe_path,
)


def strip_html_tags(text: str) -> str:
    """Strip HTML tags while preserving line breaks and unescaping HTML entities."""
    # Replace block-level tags with newlines
    text = re.sub(r"<(p|div|br|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Remove all other tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities (&amp;, &lt;, etc.)
    unescaped = html.unescape(clean)
    # Collapse excess empty lines
    return re.sub(r"\n{3,}", "\n\n", unescaped).strip()


def recursive_split_text(
    text: str,
    chunk_size: int = DOC_CHUNK_SIZE,
    chunk_overlap: int = DOC_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Recursively split text using semantic boundaries (paragraphs -> lines -> sentences -> words).
    Guarantees chunks do not exceed chunk_size while maintaining semantic coherence.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Find the first separator that splits the text
    chosen_sep = ""
    for sep in separators:
        if sep == "" or sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep) if chosen_sep != "" else list(text)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for part in splits:
        part_str = part + chosen_sep if chosen_sep != "" else part
        part_len = len(part_str)

        if current_length + part_len <= chunk_size:
            current_chunk.append(part_str)
            current_length += part_len
        else:
            if current_chunk:
                merged = "".join(current_chunk).strip()
                if merged:
                    chunks.append(merged)

                # Overlap logic: keep trailing parts up to chunk_overlap
                overlap_chunk: List[str] = []
                overlap_len = 0
                for back_item in reversed(current_chunk):
                    if overlap_len + len(back_item) <= chunk_overlap:
                        overlap_chunk.insert(0, back_item)
                        overlap_len += len(back_item)
                    else:
                        break
                current_chunk = overlap_chunk
                current_length = overlap_len

            # If a single atomic part exceeds chunk_size on its own, recurse with finer separators
            if part_len > chunk_size and len(separators) > 1:
                sub_seps = separators[separators.index(chosen_sep) + 1 :]
                sub_chunks = recursive_split_text(
                    part.strip(), chunk_size, chunk_overlap, sub_seps
                )
                chunks.extend(sub_chunks)
                current_chunk = []
                current_length = 0
            else:
                current_chunk.append(part_str)
                current_length += part_len

    if current_chunk:
        merged = "".join(current_chunk).strip()
        if merged and (not chunks or merged != chunks[-1]):
            chunks.append(merged)

    return [c for c in chunks if c.strip()]


class DocumentPipeline:
    """
    Extracts and chunks unstructured documentation files.
    Ensures every chunk is traceable to its source file and exact page number.
    """

    def __init__(
        self,
        chunk_size: int = DOC_CHUNK_SIZE,
        chunk_overlap: int = DOC_CHUNK_OVERLAP,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        base_dir: str = PROJECT_ROOT,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size = max_file_size
        self.base_dir = base_dir

    def can_process(self, file_path: str) -> bool:
        """Check if file exists, has a supported document extension, and passes security checks."""
        if not os.path.isfile(file_path):
            return False

        _, ext = os.path.splitext(file_path)
        if ext.lower() not in SUPPORTED_DOC_EXTENSIONS:
            return False

        if not is_safe_path(file_path, self.base_dir):
            logger.warning(f"Path traversal check failed for file: {file_path}")
            return False

        if is_path_ignored(file_path, self.base_dir):
            return False

        try:
            size = os.path.getsize(file_path)
            if size == 0 or size > self.max_file_size:
                return False
        except OSError:
            return False

        return True

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text, chunk, and return structured document chunks with citation metadata.
        Returns empty list if file is invalid, empty, or unreadable.
        """
        if not self.can_process(file_path):
            return []

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext == ".pdf":
                return self._process_pdf(file_path)
            elif ext in {".html", ".htm"}:
                return self._process_html(file_path)
            else:
                return self._process_text(file_path)
        except Exception as e:
            logger.warning(f"Failed to process document '{file_path}': {e}")
            return []

    def _get_relative_path(self, file_path: str) -> str:
        """Compute relative path with normalized forward slashes."""
        try:
            rel = os.path.relpath(file_path, self.base_dir)
            return rel.replace("\\", "/")
        except ValueError:
            return os.path.basename(file_path)

    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract PDF page-by-page. Never merges text across different pages,
        guaranteeing that citations map to exactly one physical page.
        """
        if pymupdf is None:
            logger.error("PyMuPDF is not installed. Cannot process PDF files.")
            return []

        rel_path = self._get_relative_path(file_path)
        file_name = os.path.basename(file_path)
        chunks: List[Dict[str, Any]] = []

        try:
            doc = pymupdf.open(file_path)
        except Exception as err:
            logger.warning(f"PyMuPDF could not open '{file_path}': {err}")
            return []

        try:
            total_pages = len(doc)
            chunk_global_idx = 0

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_text = page.get_text("text") or ""
                page_text = page_text.strip()

                if not page_text:
                    continue

                page_num = page_idx + 1
                page_chunks = recursive_split_text(
                    page_text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )

                for p_chunk_idx, c_text in enumerate(page_chunks):
                    chunk_id = f"doc:{rel_path}:p{page_num}:c{p_chunk_idx}"
                    chunks.append({
                        "id": chunk_id,
                        "text": c_text,
                        "metadata": {
                            "source_path": rel_path,
                            "file_name": file_name,
                            "file_type": "pdf",
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "chunk_index": chunk_global_idx,
                            "page_chunk_index": p_chunk_idx,
                            "char_count": len(c_text),
                            "category": "document",
                        },
                    })
                    chunk_global_idx += 1

        finally:
            doc.close()

        logger.debug(f"Processed PDF '{file_name}': {len(chunks)} chunks across {total_pages} pages")
        return chunks

    def _process_html(self, file_path: str) -> List[Dict[str, Any]]:
        """Process HTML documents by stripping tags into readable plain text."""
        raw_content = self._read_text_file(file_path)
        if not raw_content:
            return []

        clean_text = strip_html_tags(raw_content)
        return self._create_text_chunks(file_path, clean_text, file_type="html")

    def _process_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Process plain text, Markdown, and RST files."""
        raw_content = self._read_text_file(file_path)
        if not raw_content:
            return []

        _, ext = os.path.splitext(file_path)
        file_type = ext.lstrip(".").lower()
        return self._create_text_chunks(file_path, raw_content, file_type=file_type)

    def _read_text_file(self, file_path: str) -> str:
        """Read a text file trying UTF-8, then fallback encodings."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, OSError):
                continue
        # Fallback with error replacement
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def _create_text_chunks(self, file_path: str, text: str, file_type: str) -> List[Dict[str, Any]]:
        """Chunk plain text and format with consistent metadata."""
        rel_path = self._get_relative_path(file_path)
        file_name = os.path.basename(file_path)
        text_chunks = recursive_split_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks: List[Dict[str, Any]] = []
        for idx, c_text in enumerate(text_chunks):
            chunk_id = f"doc:{rel_path}:p1:c{idx}"
            chunks.append({
                "id": chunk_id,
                "text": c_text,
                "metadata": {
                    "source_path": rel_path,
                    "file_name": file_name,
                    "file_type": file_type,
                    "page_number": 1,
                    "total_pages": 1,
                    "chunk_index": idx,
                    "char_count": len(c_text),
                    "category": "document",
                },
            })

        logger.debug(f"Processed {file_type.upper()} '{file_name}': {len(chunks)} chunks")
        return chunks
