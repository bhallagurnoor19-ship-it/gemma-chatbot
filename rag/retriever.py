"""
Unified Hybrid Semantic Retriever for Document and Codebase Collections
- Queries isolated ChromaDB collections (documents_collection and codebase_collection)
- Supports 'documents', 'codebase', and 'both' retrieval modes
- Distinguishes structured retrieval statuses (success, no_results, embedding_unavailable, database_error)
- Returns sanitized user-facing error messages for integration without exposing internal traces
- Supports full backwards compatibility via container-emulated RetrievalResponse
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rag import logger
from rag.config import (
    CODEBASE_COLLECTION_NAME,
    DEFAULT_RANKING_CONFIG,
    DEFAULT_TOP_K,
    DOCUMENT_COLLECTION_NAME,
    INDEX_MANIFEST_PATH,
    SIMILARITY_DISTANCE_THRESHOLD,
)
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager


class RetrievalStatus(str, Enum):
    """Categorized status for retrieval operations."""
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    EMBEDDING_UNAVAILABLE = "embedding_unavailable"
    DATABASE_ERROR = "database_error"


@dataclass
class RetrievedChunk:
    """Structured search result carrying chunk text, similarity scores, and citation metadata."""

    content: str
    source_path: str
    file_name: str
    category: str  # "document" or "codebase"
    distance: float
    similarity: float
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    role: Optional[str] = None
    model_id: Optional[str] = None
    agent_role: Optional[str] = None
    pipeline_stage: Optional[str] = None
    timestamp: Optional[str] = None
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary representation."""
        return asdict(self)


@dataclass
class RetrievalResponse:
    """
    Structured outcome of a retrieval operation.
    Provides explicit status, sanitized error messaging, and emulates list semantics for compatibility.
    """

    status: RetrievalStatus
    chunks: List[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    mode: str = "both"
    error_message: Optional[str] = None
    total_candidates: int = 0

    # ── Container Emulation for Seamless Backwards Compatibility ──────────────
    def __iter__(self):
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> RetrievedChunk:
        return self.chunks[idx]

    def __bool__(self) -> bool:
        return bool(self.chunks)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, list):
            return self.chunks == other
        if isinstance(other, RetrievalResponse):
            return self.status == other.status and self.chunks == other.chunks
        return super().__eq__(other)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for API serialization."""
        return {
            "status": self.status.value if isinstance(self.status, RetrievalStatus) else str(self.status),
            "chunks": [c.to_dict() for c in self.chunks],
            "query": self.query,
            "mode": self.mode,
            "error_message": self.error_message,
            "total_candidates": self.total_candidates,
        }


class UnifiedRetriever:
    """
    Semantic search engine querying Document RAG and Codebase RAG collections.
    Provides configurable top-k limits, cosine relevance thresholds, and deduplication.
    Enforces active generation filtering so uncommitted, stale, or orphaned generations
    are never returned to the user or prompt context.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: int = DEFAULT_TOP_K,
        distance_threshold: float = SIMILARITY_DISTANCE_THRESHOLD,
        manifest_path: Optional[str] = None,
        oversample_factor: int = 5,
        min_candidates: int = 25,
        ranking_config: Optional[Dict[str, Any]] = None,
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.embedding_service = embedding_service or EmbeddingService()
        self.top_k = max(1, top_k)
        self.distance_threshold = distance_threshold
        self.manifest_path = (
            os.path.abspath(manifest_path)
            if manifest_path
            else os.path.abspath(INDEX_MANIFEST_PATH)
        )
        self.oversample_factor = max(1, oversample_factor)
        self.min_candidates = max(1, min_candidates)
        self.ranking_config = (
            ranking_config if ranking_config is not None else DEFAULT_RANKING_CONFIG
        )

    def _load_manifest(
        self, custom_path: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Load committed manifest from disk to map active files and generation IDs.
        Returns (manifest_dict, None) on success.
        Returns (None, error_message) if manifest file is missing, malformed, or unreadable.
        """
        target = os.path.abspath(custom_path) if custom_path else self.manifest_path
        if not target:
            return None, "Index manifest path is not configured."
        if not os.path.exists(target):
            return None, f"Index manifest does not exist at '{target}'."
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data, None
                return None, f"Index manifest root is not a dictionary (got {type(data).__name__})."
        except Exception as e:
            return None, f"Index manifest could not be read or parsed: {e}"

    def retrieve(
        self,
        query: str,
        mode: str = "both",
        target: Optional[str] = None,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        manifest_path: Optional[str] = None,
    ) -> RetrievalResponse:
        """
        Retrieve relevant context chunks with structured status information.
        Modes: 'documents', 'codebase', or 'both'.
        Filters out any chunks belonging to obsolete or uncommitted generations.
        Returns RetrievalResponse distinguishing success, no_results, embedding_unavailable, and database_error.
        """
        effective_mode = (target or mode).strip().lower()
        effective_k = top_k if (top_k is not None and top_k > 0) else self.top_k
        effective_dist_cutoff = (
            distance_threshold
            if distance_threshold is not None
            else self.distance_threshold
        )

        if not query or not query.strip():
            return RetrievalResponse(
                status=RetrievalStatus.NO_RESULTS,
                chunks=[],
                query=query or "",
                mode=effective_mode,
            )

        # Ensure vector store initialized
        if not self.vector_store.is_initialized():
            try:
                self.vector_store.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize vector store for retrieval: {e}")
                return RetrievalResponse(
                    status=RetrievalStatus.DATABASE_ERROR,
                    chunks=[],
                    query=query,
                    mode=effective_mode,
                    error_message="Vector database initialization failed.",
                )

        # 1. Generate query embedding
        try:
            if hasattr(self.embedding_service, "embed_query"):
                query_vector = self.embedding_service.embed_query(query.strip())
            else:
                query_vector = self.embedding_service.embed_text(query.strip())
        except Exception as e:
            logger.warning(
                f"Retriever: Failed to embed query '{query[:60]}...': {e}"
            )
            return RetrievalResponse(
                status=RetrievalStatus.EMBEDDING_UNAVAILABLE,
                chunks=[],
                query=query,
                mode=effective_mode,
                error_message="Embedding service is currently unavailable. Please verify Ollama is running.",
            )

        # 2. Select target collections
        collections_to_query: List[str] = []
        if effective_mode in ("documents", "doc", "docs"):
            collections_to_query = [DOCUMENT_COLLECTION_NAME]
        elif effective_mode in ("codebase", "code"):
            collections_to_query = [CODEBASE_COLLECTION_NAME]
        elif effective_mode in ("both", "all", "auto"):
            collections_to_query = [DOCUMENT_COLLECTION_NAME, CODEBASE_COLLECTION_NAME]
        else:
            raise ValueError(
                f"Invalid retrieval mode '{effective_mode}'. Expected 'documents', 'codebase', or 'both'."
            )

        # Load active manifest state for active-generation filtering
        manifest, manifest_err = self._load_manifest(custom_path=manifest_path)
        if manifest_err is not None:
            logger.error(f"Retriever aborted: {manifest_err}")
            return RetrievalResponse(
                status=RetrievalStatus.DATABASE_ERROR,
                chunks=[],
                query=query,
                mode=effective_mode,
                error_message=f"Index manifest error: {manifest_err}",
            )

        # 3. Query collections
        raw_candidates: List[RetrievedChunk] = []

        for col_name in collections_to_query:
            manifest_cat_key = "documents" if col_name == DOCUMENT_COLLECTION_NAME else "codebase"
            raw_cat = manifest.get(manifest_cat_key, {})
            cat_manifest: Dict[str, Any] = {}
            if isinstance(raw_cat, dict):
                cat_manifest = {
                    k.replace("\\", "/"): v for k, v in raw_cat.items()
                }

            # Over-sample candidate chunks to ensure active generation chunks are not starved
            # by higher-ranking stale/orphaned chunks in ChromaDB
            query_n = max(effective_k * self.oversample_factor, self.min_candidates)
            try:
                col_obj = self.vector_store.get_collection(col_name)
                col_count = col_obj.count()
                query_n = min(query_n, max(1, col_count))
            except Exception:
                pass

            try:
                res = self.vector_store.query(
                    collection_name=col_name,
                    query_embeddings=query_vector,
                    n_results=query_n,
                )
            except Exception as e:
                logger.warning(f"Retriever: Query on '{col_name}' failed or collection uninitialized: {e}")
                return RetrievalResponse(
                    status=RetrievalStatus.DATABASE_ERROR,
                    chunks=[],
                    query=query,
                    mode=effective_mode,
                    error_message="Vector storage error during query execution.",
                )

            ids_list = res.get("ids", [[]])
            docs_list = res.get("documents", [[]])
            metas_list = res.get("metadatas", [[]])
            dists_list = res.get("distances", [[]])

            if not ids_list or not ids_list[0]:
                continue

            ids = ids_list[0]
            docs = docs_list[0]
            metas = metas_list[0] if metas_list else [{}] * len(ids)
            dists = dists_list[0] if dists_list else [1.0] * len(ids)

            default_cat = "document" if col_name == DOCUMENT_COLLECTION_NAME else "codebase"

            for c_id, text, meta, dist in zip(ids, docs, metas, dists):
                if dist is None:
                    continue

                dist_float = float(dist)
                # Relevance threshold filter (cosine distance: lower is closer)
                if dist_float > effective_dist_cutoff:
                    continue

                meta = meta or {}
                source_path_raw = str(meta.get("source_path", ""))
                norm_source = source_path_raw.replace("\\", "/")
                chunk_gen = meta.get("generation_id")

                # ── Active Generation Filtering ───────────────────────────
                # 1. File must have an active entry in the committed manifest
                if norm_source not in cat_manifest:
                    logger.debug(
                        f"Retriever: Excluding chunk '{c_id}' because '{norm_source}' has no active manifest entry."
                    )
                    continue

                active_entry = cat_manifest[norm_source]
                if isinstance(active_entry, dict):
                    active_gen = active_entry.get("generation_id")
                    active_ids = set(active_entry.get("chunk_ids", []))

                    # 2. If chunk has a generation_id, it must match active_gen
                    if active_gen and chunk_gen and chunk_gen != active_gen:
                        logger.debug(
                            f"Retriever: Excluding chunk '{c_id}' with generation '{chunk_gen}' "
                            f"(active generation is '{active_gen}')."
                        )
                        continue

                    # 3. If active_ids is defined, chunk ID must belong to active_ids
                    if active_ids and c_id not in active_ids:
                        logger.debug(
                            f"Retriever: Excluding chunk '{c_id}' because it is not in active chunk IDs."
                        )
                        continue

                # Cosine similarity calculation: 1.0 - distance
                sim = max(0.0, min(1.0, 1.0 - dist_float))

                page_num = self._safe_int(meta.get("page_number"))
                total_pages = self._safe_int(meta.get("total_pages"))
                start_l = self._safe_int(meta.get("start_line"))
                end_l = self._safe_int(meta.get("end_line"))

                chunk = RetrievedChunk(
                    content=text,
                    source_path=source_path_raw,
                    file_name=str(meta.get("file_name", "")),
                    category=str(meta.get("category", default_cat)),
                    distance=round(dist_float, 4),
                    similarity=round(sim, 4),
                    page_number=page_num,
                    total_pages=total_pages,
                    symbol_name=meta.get("symbol_name") or None,
                    symbol_type=meta.get("symbol_type") or None,
                    start_line=start_l,
                    end_line=end_l,
                    chunk_id=c_id,
                )
                raw_candidates.append(chunk)

        # 4. Deduplication
        deduped: List[RetrievedChunk] = self._deduplicate_chunks(raw_candidates)

        # 5. Ranking: Apply source-aware ranking if enabled, otherwise sort by distance
        ranked_chunks = self._apply_source_aware_ranking(deduped, query)

        # 6. Build structured response
        final_chunks = ranked_chunks[:effective_k]
        if final_chunks:
            return RetrievalResponse(
                status=RetrievalStatus.SUCCESS,
                chunks=final_chunks,
                query=query,
                mode=effective_mode,
                total_candidates=len(raw_candidates),
            )
        else:
            return RetrievalResponse(
                status=RetrievalStatus.NO_RESULTS,
                chunks=[],
                query=query,
                mode=effective_mode,
                total_candidates=0,
            )

    @staticmethod
    def _safe_int(val: Any) -> Optional[int]:
        """Safely convert value to int if possible, else None."""
        if val is None or val == "":
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _deduplicate_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Deduplicate chunks by chunk_id and content hash.
        Retains the occurrence with lowest distance (highest similarity).
        """
        seen_ids: Set[str] = set()
        seen_content_hashes: Set[str] = set()
        unique_chunks: List[RetrievedChunk] = []

        sorted_chunks = sorted(chunks, key=lambda c: c.distance)

        for chunk in sorted_chunks:
            if chunk.chunk_id and chunk.chunk_id in seen_ids:
                continue

            norm_content = " ".join(chunk.content.split())
            c_hash = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()
            if c_hash in seen_content_hashes:
                continue

            if chunk.chunk_id:
                seen_ids.add(chunk.chunk_id)
            seen_content_hashes.add(c_hash)
            unique_chunks.append(chunk)

        return unique_chunks

    def _detect_query_intent(self, query: str) -> str:
        """
        Classify query intent to guide source-aware ranking:
        - 'test_query': queries about tests, test cases, test suites, mocks, assertions
        - 'config_query': queries about configuration, settings, environment, .ragignore, manifests
        - 'frontend_query': queries about UI, HTML, CSS, frontend templates, DOM, client controls
        - 'impl_query': queries about implementation logic, functions, pipelines, backend methods (default)
        """
        q = query.lower()

        # Test intent
        if re.search(
            r"\b(test|tests|testing|unittest|pytest|assert|assertion|assertions|mock|fixtures?|test_case|testsuite)\b",
            q,
        ):
            return "test_query"

        # Frontend intent
        if re.search(
            r"\b(html|css|frontend|ui|interface|templates?|buttons?|badges?|modals?|dropdowns?|sliders?|themes?|client|browser|javascript|dom|eventlistener|index\.html|streambotpipeline|sse|pills?)\b",
            q,
        ):
            return "frontend_query"

        # Config intent
        if re.search(
            r"\b(config|configs|configuration|settings|environment|ragignore|\.ragignore|manifest|parameters|threshold|constants?|paths?|requirements\.txt)\b",
            q,
        ):
            return "config_query"

        return "impl_query"

    def _classify_chunk_source(self, chunk: RetrievedChunk) -> str:
        """
        Determine the architectural role of a chunk:
        'document', 'frontend', 'test', 'config', or 'impl'.
        """
        if chunk.category == "document":
            return "document"

        norm_path = (chunk.source_path or "").replace("\\", "/").lower()
        file_name = (chunk.file_name or "").lower()

        # Frontend templates and assets
        if (
            file_name.endswith((".html", ".htm", ".css", ".js"))
            or "/templates/" in norm_path
            or norm_path.startswith("templates/")
            or "/static/" in norm_path
            or norm_path.startswith("static/")
            or chunk.symbol_type == "template"
        ):
            return "frontend"

        # Tests
        if (
            file_name.startswith("test_")
            or file_name.startswith("tests_")
            or file_name.endswith(("_test.py", "_tests.py"))
            or "/tests/" in norm_path
            or norm_path.startswith("tests/")
            or "tests_" in file_name
        ):
            return "test"

        # Configuration
        if (
            file_name in ("config.py", "requirements.txt", ".ragignore")
            or file_name.endswith((".json", ".yaml", ".yml", ".toml", ".env.example"))
            or "/config" in norm_path
        ):
            return "config"

        return "impl"

    def _compute_ranking_score(self, chunk: RetrievedChunk, query_intent: str) -> float:
        """
        Compute calibrated score for ranking based on cosine similarity and source-aware multipliers.
        """
        base_sim = chunk.similarity
        if not self.ranking_config or not self.ranking_config.get("enabled", True):
            chunk.score = base_sim
            return chunk.score

        source_type = self._classify_chunk_source(chunk)
        if source_type == "document":
            chunk.score = base_sim
            return chunk.score

        intent_rules = self.ranking_config.get(query_intent, {})
        multiplier = 1.0

        if source_type == "impl":
            multiplier = intent_rules.get("impl_boost") or intent_rules.get("impl_multiplier", 1.0)
        elif source_type == "test":
            multiplier = intent_rules.get("test_boost") or intent_rules.get("test_multiplier", 1.0)
        elif source_type == "config":
            multiplier = intent_rules.get("config_boost") or intent_rules.get("config_multiplier", 1.0)
        elif source_type == "frontend":
            multiplier = intent_rules.get("frontend_boost") or intent_rules.get("frontend_multiplier", 1.0)

        chunk.score = round(base_sim * multiplier, 4)
        return chunk.score

    def _apply_source_aware_ranking(
        self, chunks: List[RetrievedChunk], query: str
    ) -> List[RetrievedChunk]:
        """
        Apply source-aware ranking to deduplicated candidate chunks.
        Sorts candidates primarily by adjusted score (descending), then by raw distance (ascending).
        """
        if not chunks:
            return []

        intent = self._detect_query_intent(query)
        for chunk in chunks:
            self._compute_ranking_score(chunk, intent)

        # Sort descending by score, tie-breaking on lowest raw distance
        return sorted(
            chunks,
            key=lambda c: (-(c.score if c.score is not None else c.similarity), c.distance),
        )

