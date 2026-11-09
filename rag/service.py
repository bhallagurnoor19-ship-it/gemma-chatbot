"""
RAG Service & Chatbot Integration Layer
- Coordinates between user query routing, UnifiedRetriever, and PromptBuilder
- Supports retrieval modes: 'off', 'auto', 'documents', 'codebase', 'both'
- Provides conservative automated routing to avoid unnecessary retrieval on greetings or general chat
- Emits structured context and metadata for downstream LLMs (Gemma, DeepSeek, Qwen, Gemini)
- Guarantees fail-safe resilience: retrieval errors never crash the chatbot
"""

from dataclasses import dataclass, field
import os
import re
from typing import Any, Dict, List, Optional, Union

from rag import logger
from rag.prompt_builder import PromptBuilder
from rag.retriever import RetrievedChunk, RetrievalResponse, RetrievalStatus, UnifiedRetriever

# ── Intent Classification Regex Patterns ─────────────────────────────────────
# Normal greetings and pleasantries -> Off
GREETINGS_PATTERN = re.compile(
    r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|yo|sup|hiya)\b[!?.]*$",
    re.IGNORECASE,
)

# Small talk and conversational phrases -> Off
SMALL_TALK_PATTERN = re.compile(
    r"\b(who\s+are\s+you|what\s+can\s+you\s+do|tell\s+me\s+a\s+joke|how\s+are\s+you|"
    r"thank\s+you|thanks|bye|goodbye|see\s+ya|nice\s+to\s+meet\s+you)\b",
    re.IGNORECASE,
)

# Combined documentation and repository queries -> Both
COMBINED_PATTERN = re.compile(
    r"(?=.*\b(docs?|documentation|guide|manual|spec|pdf|handbook|policy|readme)\b)"
    r"(?=.*\b(code|implementation|source|repo|repository|script|function|class|method)\b)",
    re.IGNORECASE,
)

# Document queries -> Documents
DOCUMENTS_PATTERN = re.compile(
    r"\b(pdf|documents?|docs?|handbook|guide|manual|policy|specs?|guidelines?|uploaded\s+file|"
    r"whitepaper|release\s+notes|documentation|readme)\b|"
    r"(in\s+the\s+doc|what\s+does\s+the\s+guide\s+say|according\s+to\s+the\s+(pdf|doc)|"
    r"find\s+in\s+documents|search\s+documents)",
    re.IGNORECASE,
)

# Codebase / Repository queries -> Codebase
CODEBASE_PATTERN = re.compile(
    r"\b(codebase|repo|repository|source\s+code|project\s+structure|in\s+app\.py|in\s+chatbot\.py|"
    r"rag\s+package|rag/|indexer\.py|retriever\.py|prompt_builder\.py|"
    r"directory\s+structure|classes?|functions?|methods?|endpoints?|routes?)\b|"
    r"(search\s+project|find\s+in\s+repo|look\s+up\s+in\s+codebase|search\s+code|"
    r"how\s+is\s+it\s+implemented\s+in\s+this\s+project|show\s+me\s+the\s+code\s+for)",
    re.IGNORECASE,
)

# Historical Conversational Memory queries -> Memory
MEMORY_PATTERN = re.compile(
    r"\b(earlier|previously|before|prior\s+conversation|past\s+discussion|we\s+discussed|we\s+decided|"
    r"you\s+mentioned|you\s+said|i\s+said|i\s+asked|last\s+time|previous\s+(message|turn|conversation|chat|discussion|question|answer)|"
    r"what\s+did\s+(we|you|i)\s+(decide|discuss|say|mention|agree|talk\s+about|plan)|"
    r"what\s+was\s+(the|our)\s+(decision|plan|discussion|agreement)|"
    r"what\s+did\s+(qwen|deepseek|gemma|architect|coder)\s+(say|do|implement|write|propose|suggest|complete)|"
    r"(qwen|deepseek|gemma|architect|coder)('s)?\s+(earlier\s+)?(implementation|code|architecture|plan|design|work|response)|"
    r"continue\s+(the|our)?\s*(implementation|work|task|code|discussion)\s+(we\s+discussed|from\s+earlier|from\s+before)|"
    r"continue\s+from\s+(earlier|before|where\s+we\s+left\s+off)|"
    r"as\s+(we|you|i)\s+(discussed|mentioned|said|decided)\s+(earlier|before)|"
    r"prepare\s+a\s+report\s+on\s+(the|our)?\s*(codebase\s+work|discussions?|conversation|project\s+history)|"
    r"summarize\s+(our|the)\s+(conversation|discussion|chat|history)|"
    r"what\s+have\s+we\s+(done|built|decided|completed)\s+so\s+far)\b",
    re.IGNORECASE,
)


@dataclass
class RAGContextResult:
    """Structured outcome of RAG service context retrieval."""

    mode: str  # "off", "documents", "codebase", "both"
    status: str  # "off", "success", "no_results", "error"
    context_text: str = ""
    chunks: List[RetrievedChunk] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    token_count: int = 0

    @property
    def has_context(self) -> bool:
        """True if non-empty context text is available for prompt injection."""
        return bool(self.context_text and self.context_text.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation for API serialization."""
        return {
            "mode": self.mode,
            "status": self.status,
            "has_context": self.has_context,
            "context_text": self.context_text,
            "chunks_count": len(self.chunks),
            "sources": self.sources,
            "error_message": self.error_message,
            "token_count": self.token_count,
        }


class RAGService:
    """
    High-level integration controller for the RAG subsystem.
    Orchestrates query classification, retrieval execution, and prompt context synthesis
    across isolated Document, Codebase, and Conversational Memory collections.
    """

    def __init__(
        self,
        retriever: Optional[UnifiedRetriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        memory_store: Optional[Any] = None,
        default_mode: str = "auto",
    ):
        self.retriever = retriever or UnifiedRetriever()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self._memory_store = memory_store
        self.default_mode = default_mode.lower()

    @property
    def memory_store(self):
        """Lazy-loaded conversation memory store instance."""
        if self._memory_store is None:
            from rag.memory import get_conversation_store
            self._memory_store = get_conversation_store()
        return self._memory_store

    def detect_retrieval_mode(self, query: str, explicit_mode: Optional[str] = "auto") -> str:
        """
        Determine the effective retrieval mode:
        - If explicit_mode is set to 'off', 'documents', 'codebase', 'both', 'memory', or 'all', honors it directly.
        - If 'auto', applies conservative heuristics to select mode without over-retrieving.
        """
        req_mode = (explicit_mode or self.default_mode).strip().lower()

        # Direct explicit mode matches
        if req_mode in ("off", "disabled", "none", "false"):
            return "off"
        if req_mode in ("documents", "doc", "docs"):
            return "documents"
        if req_mode in ("codebase", "code", "repo"):
            return "codebase"
        if req_mode in ("both",):
            return "both"
        if req_mode in ("memory", "history", "chat_history"):
            return "memory"
        if req_mode in ("all",):
            return "all"

        if req_mode != "auto":
            return "off"

        # ── Conservative Auto-Routing ─────────────────────────────────────────
        if not query or not query.strip():
            return "off"

        q_clean = query.strip()

        # 1. Greetings & Pleasantries -> Off
        if GREETINGS_PATTERN.match(q_clean):
            return "off"

        # 2. General Small Talk (short queries) -> Off
        if len(q_clean) <= 60 and SMALL_TALK_PATTERN.search(q_clean):
            return "off"

        is_memory = bool(MEMORY_PATTERN.search(q_clean))
        is_codebase = bool(CODEBASE_PATTERN.search(q_clean))
        is_document = bool(DOCUMENTS_PATTERN.search(q_clean))
        is_combined_doc_code = bool(COMBINED_PATTERN.search(q_clean))

        # 3. Multi-source combinations
        if is_combined_doc_code or (is_codebase and is_document):
            return "all" if is_memory else "both"

        if is_memory:
            if is_codebase:
                return "memory+codebase"
            if is_document:
                return "memory+documents"
            return "memory"

        # 4. Single source
        if is_codebase:
            return "codebase"

        if is_document:
            return "documents"

        # 5. Conservative Default -> Off
        return "off"

    def retrieve_context(
        self,
        query: str,
        mode: str = "auto",
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_role: Optional[str] = None,
        model_id: Optional[str] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        distance_threshold: Optional[float] = None,
    ) -> RAGContextResult:
        """
        Execute grounded context retrieval across documents, codebase, and/or conversational memory.
        Returns a structured RAGContextResult with status, context text, and sources.
        Guarantees safe failure fallback without raising exceptions.
        """
        effective_mode = self.detect_retrieval_mode(query, explicit_mode=mode)

        if effective_mode == "off":
            return RAGContextResult(
                mode="off",
                status="off",
                context_text="",
                chunks=[],
                sources=[],
            )

        try:
            chunks: List[RetrievedChunk] = []
            errors: List[str] = []

            needs_documents = effective_mode in ("documents", "both", "all", "memory+documents")
            needs_codebase = effective_mode in ("codebase", "both", "all", "memory+codebase")
            needs_memory = effective_mode in ("memory", "all", "memory+codebase", "memory+documents")

            # 1. Retrieve from UnifiedRetriever (Documents and/or Codebase)
            if needs_documents or needs_codebase:
                retriever_mode = (
                    "both" if (needs_documents and needs_codebase)
                    else ("documents" if needs_documents else "codebase")
                )
                ret_resp = self.retriever.retrieve(
                    query=query,
                    mode=retriever_mode,
                    top_k=top_k,
                    distance_threshold=distance_threshold,
                )
                if ret_resp.status == RetrievalStatus.SUCCESS and ret_resp.chunks:
                    chunks.extend(ret_resp.chunks)
                elif ret_resp.status in (RetrievalStatus.EMBEDDING_UNAVAILABLE, RetrievalStatus.DATABASE_ERROR):
                    if ret_resp.error_message:
                        errors.append(f"Retriever: {ret_resp.error_message}")

            # 2. Retrieve from Conversational Memory
            if needs_memory:
                # Infer target agent role if mentioned explicitly in query
                target_agent = agent_role
                if not target_agent:
                    q_lower = query.lower()
                    if "qwen" in q_lower or "coder" in q_lower:
                        target_agent = "coder"
                    elif "deepseek" in q_lower or "architect" in q_lower:
                        target_agent = "architect"
                    elif "gemma" in q_lower:
                        target_agent = "general_assistant"

                try:
                    mem_chunks = self.memory_store.retrieve_memory(
                        query=query,
                        conversation_id=conversation_id,
                        project_id=project_id,
                        agent_role=target_agent,
                        model_id=model_id,
                        top_k=top_k or 4,
                        distance_threshold=distance_threshold or 0.65,
                    )
                    if mem_chunks:
                        chunks.extend(mem_chunks)
                except Exception as mem_err:
                    logger.error(f"Error querying conversation memory: {mem_err}")
                    errors.append(f"Memory: {mem_err}")

            if chunks:
                context_text = self.prompt_builder.build_context_block(
                    chunks,
                    max_tokens=max_tokens,
                )
                sources = self._extract_sources(chunks)
                approx_tokens = int(len(context_text) / self.prompt_builder.chars_per_token)

                return RAGContextResult(
                    mode=effective_mode,
                    status="success",
                    context_text=context_text,
                    chunks=chunks,
                    sources=sources,
                    token_count=approx_tokens,
                )

            if errors:
                err_msg = "; ".join(errors)
                logger.warning(f"RAGService retrieval encounter error(s): {err_msg}")
                return RAGContextResult(
                    mode=effective_mode,
                    status="error",
                    context_text="",
                    chunks=[],
                    sources=[],
                    error_message=err_msg,
                )

            return RAGContextResult(
                mode=effective_mode,
                status="no_results",
                context_text="",
                chunks=[],
                sources=[],
            )

        except Exception as e:
            logger.error(f"RAGService: Unexpected error during context retrieval: {e}")
            return RAGContextResult(
                mode=effective_mode,
                status="error",
                context_text="",
                chunks=[],
                sources=[],
                error_message=str(e),
            )

    def _extract_sources(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Extract clean, structured source citations from retrieved chunks."""
        seen_keys = set()
        sources: List[Dict[str, Any]] = []

        for chunk in chunks:
            citation_tag = self.prompt_builder.format_citation(chunk)
            key = (chunk.source_path, chunk.page_number, chunk.start_line, chunk.end_line, chunk.chunk_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            src_dict: Dict[str, Any] = {
                "citation": citation_tag,
                "file_name": chunk.file_name,
                "source_path": chunk.source_path,
                "category": chunk.category,
                "similarity": chunk.similarity,
                "page_number": chunk.page_number,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol_name": chunk.symbol_name,
            }
            if hasattr(chunk, "role"):
                src_dict["role"] = getattr(chunk, "role", "")
            if hasattr(chunk, "model_id"):
                src_dict["model_id"] = getattr(chunk, "model_id", "")
            if hasattr(chunk, "agent_role"):
                src_dict["agent_role"] = getattr(chunk, "agent_role", "")
            if hasattr(chunk, "pipeline_stage"):
                src_dict["pipeline_stage"] = getattr(chunk, "pipeline_stage", "")
            if hasattr(chunk, "timestamp"):
                src_dict["timestamp"] = getattr(chunk, "timestamp", "")

            sources.append(src_dict)

        return sources


# ── Global Singleton Accessor ────────────────────────────────────────────────
_rag_service_instance: Optional[RAGService] = None


def get_rag_service(
    retriever: Optional[UnifiedRetriever] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    memory_store: Optional[Any] = None,
    reset: bool = False,
) -> RAGService:
    """
    Return the singleton RAGService instance.
    Pass reset=True to reinitialize (e.g. for isolated unit testing).
    """
    global _rag_service_instance
    if _rag_service_instance is None or reset:
        _rag_service_instance = RAGService(
            retriever=retriever,
            prompt_builder=prompt_builder,
            memory_store=memory_store,
        )
    return _rag_service_instance


def detect_retrieval_mode(query: str, explicit_mode: Optional[str] = "auto") -> str:
    """Convenience module function for query mode detection."""
    return get_rag_service().detect_retrieval_mode(query=query, explicit_mode=explicit_mode)
