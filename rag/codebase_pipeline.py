"""
Codebase Ingestion & Structure-Aware AST Parsing Pipeline
- Extracts Python classes, methods, functions, docstrings, imports, decorators, and line ranges
- Employs AST-based parsing with exact start/end line numbers
- Supports configuration files (JSON, YAML, TOML) and script files with safe line-based chunking
- Robust fallback to line-based chunking on Python SyntaxError or unparseable code
"""

import ast
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from rag import logger
from rag.config import (
    CODE_MAX_CHUNK_LINES,
    CODE_MIN_CHUNK_LINES,
    MAX_FILE_SIZE_BYTES,
    PROJECT_ROOT,
    SUPPORTED_CODE_EXTENSIONS,
    is_path_ignored,
    is_safe_path,
)

EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".bat": "batch",
    ".vbs": "vbs",
    ".sh": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
    ".js": "javascript",
    ".ts": "typescript",
    ".css": "css",
    ".html": "html",
    ".htm": "html",
}

CONFIG_EXTENSIONS: Set[str] = {".json", ".yaml", ".yml", ".toml"}


class CodebasePipeline:
    """
    Parses source code and configuration files into semantic code chunks.
    Preserves exact source lines, symbol names, and line number metadata.
    """

    def __init__(
        self,
        max_chunk_lines: int = CODE_MAX_CHUNK_LINES,
        min_chunk_lines: int = CODE_MIN_CHUNK_LINES,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        base_dir: str = PROJECT_ROOT,
    ):
        self.max_chunk_lines = max_chunk_lines
        self.min_chunk_lines = min_chunk_lines
        self.max_file_size = max_file_size
        self.base_dir = base_dir

    def can_process(self, file_path: str) -> bool:
        """Check if file exists, has a supported code/config extension, and passes security rules."""
        if not os.path.isfile(file_path):
            return False

        _, ext = os.path.splitext(file_path)
        if ext.lower() not in SUPPORTED_CODE_EXTENSIONS:
            return False

        if not is_safe_path(file_path, self.base_dir):
            logger.warning(f"Path traversal check failed for code file: {file_path}")
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
        Parse a source code or config file into semantic chunks with line metadata.
        Returns empty list if file is invalid, ignored, or empty.
        """
        if not self.can_process(file_path):
            return []

        raw_content = self._read_text_file(file_path)
        if not raw_content or not raw_content.strip():
            return []

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".py":
            try:
                return self._process_python_ast(file_path, raw_content)
            except SyntaxError as e:
                logger.warning(
                    f"SyntaxError in Python file '{file_path}' (line {e.lineno}): falling back to line chunking"
                )
                return self._process_line_based(file_path, raw_content, language="python")
            except Exception as e:
                logger.warning(
                    f"AST parsing failed for '{file_path}': {e}; falling back to line chunking"
                )
                return self._process_line_based(file_path, raw_content, language="python")
        else:
            lang = EXT_TO_LANG.get(ext, "text")
            return self._process_line_based(file_path, raw_content, language=lang)

    def _get_relative_path(self, file_path: str) -> str:
        """Compute relative path with normalized forward slashes."""
        try:
            rel = os.path.relpath(file_path, self.base_dir)
            return rel.replace("\\", "/")
        except ValueError:
            return os.path.basename(file_path)

    def _read_text_file(self, file_path: str) -> str:
        """Read a text file trying UTF-8, then fallback encodings."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, OSError):
                continue
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def _process_python_ast(self, file_path: str, source: str) -> List[Dict[str, Any]]:
        """
        Parse Python source code via AST.
        Extracts module headers, classes, methods, functions, and standalone code blocks.
        """
        tree = ast.parse(source, filename=file_path)
        lines = source.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return []

        rel_path = self._get_relative_path(file_path)
        file_name = os.path.basename(file_path)

        raw_chunks: List[Dict[str, Any]] = []
        covered_lines: Set[int] = set()

        # 1. Module Header & Imports
        import_nodes: List[ast.AST] = []
        docstring_end_line = 0

        # Check for module docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            docstring_end_line = getattr(tree.body[0], "end_lineno", tree.body[0].lineno)

        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_nodes.append(stmt)

        if import_nodes or docstring_end_line > 0:
            start_l = 1
            last_import_line = (
                max(getattr(n, "end_lineno", n.lineno) for n in import_nodes)
                if import_nodes
                else docstring_end_line
            )
            end_l = max(docstring_end_line, last_import_line)
            header_lines = lines[start_l - 1 : end_l]
            header_text = "\n".join(header_lines).strip()
            if header_text:
                raw_chunks.append({
                    "symbol_type": "module_header",
                    "symbol_name": "module",
                    "start_line": start_l,
                    "end_line": end_l,
                    "text": header_text,
                })
                for ln in range(start_l, end_l + 1):
                    covered_lines.add(ln)

        # 2. Top-level Classes & Functions
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                class_chunks = self._extract_class_chunks(stmt, lines)
                for c in class_chunks:
                    raw_chunks.append(c)
                    for ln in range(c["start_line"], c["end_line"] + 1):
                        covered_lines.add(ln)

            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_chunks = self._extract_function_chunks(stmt, lines, parent_class=None)
                for c in func_chunks:
                    raw_chunks.append(c)
                    for ln in range(c["start_line"], c["end_line"] + 1):
                        covered_lines.add(ln)

        # 3. Uncovered top-level code blocks (e.g. if __name__ == '__main__': or global logic)
        uncovered_start = None
        for i in range(1, total_lines + 1):
            line_str = lines[i - 1].strip()
            if i not in covered_lines and line_str and not line_str.startswith("#"):
                if uncovered_start is None:
                    uncovered_start = i
            else:
                if uncovered_start is not None:
                    uncovered_end = i - 1
                    block_lines = lines[uncovered_start - 1 : uncovered_end]
                    block_text = "\n".join(block_lines).strip()
                    if block_text and len(block_lines) >= self.min_chunk_lines:
                        # Split if exceeds max lines
                        sub_blocks = self._split_line_range(
                            lines, uncovered_start, uncovered_end, "code_block", "global"
                        )
                        raw_chunks.extend(sub_blocks)
                    uncovered_start = None

        if uncovered_start is not None:
            uncovered_end = total_lines
            block_lines = lines[uncovered_start - 1 : uncovered_end]
            block_text = "\n".join(block_lines).strip()
            if block_text and len(block_lines) >= self.min_chunk_lines:
                sub_blocks = self._split_line_range(
                    lines, uncovered_start, uncovered_end, "code_block", "global"
                )
                raw_chunks.extend(sub_blocks)

        # Sort chunks by start_line
        raw_chunks.sort(key=lambda x: x["start_line"])

        # Format into final chunk structures
        final_chunks: List[Dict[str, Any]] = []
        for idx, chunk_info in enumerate(raw_chunks):
            st = chunk_info["symbol_type"]
            sn = chunk_info["symbol_name"]
            s_line = chunk_info["start_line"]
            e_line = chunk_info["end_line"]
            c_text = chunk_info["text"]

            chunk_id = f"code:{rel_path}:{st}:{sn}:{s_line}_{e_line}:c{idx}"
            final_chunks.append({
                "id": chunk_id,
                "text": c_text,
                "metadata": {
                    "source_path": rel_path,
                    "file_name": file_name,
                    "language": "python",
                    "symbol_name": sn,
                    "symbol_type": st,
                    "start_line": s_line,
                    "end_line": e_line,
                    "chunk_index": idx,
                    "char_count": len(c_text),
                    "category": "codebase",
                },
            })

        logger.debug(
            f"Processed Python AST '{file_name}': {len(final_chunks)} chunks across {total_lines} lines"
        )
        return final_chunks

    def _extract_class_chunks(
        self, class_node: ast.ClassDef, lines: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract class definition header and its individual member methods."""
        chunks: List[Dict[str, Any]] = []
        class_name = class_node.name
        start_line = class_node.lineno
        end_line = getattr(class_node, "end_lineno", start_line)

        # Collect methods inside the class
        methods = [
            m for m in class_node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if not methods:
            # Simple class with no methods (dataclass, constants, etc.)
            sub_chunks = self._split_line_range(
                lines, start_line, end_line, symbol_type="class", symbol_name=class_name
            )
            chunks.extend(sub_chunks)
            return chunks

        # Class header: lines from class start up to the start of the first method
        first_method_start = methods[0].lineno
        header_end = first_method_start - 1
        if header_end >= start_line:
            header_lines = lines[start_line - 1 : header_end]
            header_text = "\n".join(header_lines).strip()
            if header_text:
                chunks.append({
                    "symbol_type": "class",
                    "symbol_name": class_name,
                    "start_line": start_line,
                    "end_line": header_end,
                    "text": header_text,
                })

        # Process each method inside the class
        for method in methods:
            m_chunks = self._extract_function_chunks(method, lines, parent_class=class_name)
            chunks.extend(m_chunks)

        return chunks

    def _extract_function_chunks(
        self,
        func_node: Any,
        lines: List[str],
        parent_class: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract a function or method, sub-chunking if exceeding max_chunk_lines."""
        func_name = func_node.name
        full_symbol_name = f"{parent_class}.{func_name}" if parent_class else func_name
        symbol_type = "method" if parent_class else "function"

        start_line = func_node.lineno
        end_line = getattr(func_node, "end_lineno", start_line)

        return self._split_line_range(
            lines, start_line, end_line, symbol_type=symbol_type, symbol_name=full_symbol_name
        )

    def _split_line_range(
        self,
        lines: List[str],
        start_line: int,
        end_line: int,
        symbol_type: str,
        symbol_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Split a range of lines into chunks if it exceeds max_chunk_lines.
        Preserves exact start and end line numbers for every sub-chunk.
        """
        total_range_lines = end_line - start_line + 1
        if total_range_lines <= self.max_chunk_lines:
            chunk_text = "\n".join(lines[start_line - 1 : end_line]).strip()
            if not chunk_text:
                return []
            return [{
                "symbol_type": symbol_type,
                "symbol_name": symbol_name,
                "start_line": start_line,
                "end_line": end_line,
                "text": chunk_text,
            }]

        # Line-based sliding window with overlap
        chunks: List[Dict[str, Any]] = []
        step = max(1, self.max_chunk_lines - 15)  # 15 line overlap
        cur_start = start_line

        while cur_start <= end_line:
            cur_end = min(end_line, cur_start + self.max_chunk_lines - 1)
            chunk_text = "\n".join(lines[cur_start - 1 : cur_end]).strip()
            if chunk_text:
                chunks.append({
                    "symbol_type": symbol_type,
                    "symbol_name": symbol_name,
                    "start_line": cur_start,
                    "end_line": cur_end,
                    "text": chunk_text,
                })
            if cur_end >= end_line:
                break
            cur_start += step

        return chunks

    def _process_line_based(
        self, file_path: str, source: str, language: str
    ) -> List[Dict[str, Any]]:
        """
        Safe line-based chunking for non-Python or unparseable source/config files.
        Maintains start and end line numbers and chunk indices.
        """
        lines = source.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return []

        rel_path = self._get_relative_path(file_path)
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)

        symbol_type = (
            "config"
            if ext.lower() in CONFIG_EXTENSIONS
            else ("template" if ext.lower() in {".html", ".htm"} else "code_block")
        )
        symbol_name = file_name

        chunks: List[Dict[str, Any]] = []
        step = max(1, self.max_chunk_lines - 15)
        cur_start = 1
        chunk_idx = 0

        while cur_start <= total_lines:
            cur_end = min(total_lines, cur_start + self.max_chunk_lines - 1)
            chunk_lines = lines[cur_start - 1 : cur_end]
            chunk_text = "\n".join(chunk_lines).strip()

            if chunk_text:
                chunk_id = (
                    f"code:{rel_path}:{symbol_type}:{symbol_name}:{cur_start}_{cur_end}:c{chunk_idx}"
                )
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source_path": rel_path,
                        "file_name": file_name,
                        "language": language,
                        "symbol_name": symbol_name,
                        "symbol_type": symbol_type,
                        "start_line": cur_start,
                        "end_line": cur_end,
                        "chunk_index": chunk_idx,
                        "char_count": len(chunk_text),
                        "category": "codebase",
                    },
                })
                chunk_idx += 1

            if cur_end >= total_lines:
                break
            cur_start += step

        logger.debug(
            f"Processed {language.upper()} '{file_name}': {len(chunks)} chunks across {total_lines} lines"
        )
        return chunks
