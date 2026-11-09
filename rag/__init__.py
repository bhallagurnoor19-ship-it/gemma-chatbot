"""
Unified RAG (Retrieval-Augmented Generation) Subsystem for Gemma Chatbot
Provides Document RAG (PDF, TXT, MD) and Codebase RAG (Python AST, Project Modules).
"""

import logging
import os

# ── Subsystem Logging Configuration ───────────────────────────────────────────
logger = logging.getLogger("rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [RAG-%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from rag.config import (
    CODEBASE_COLLECTION_NAME,
    DOCUMENT_COLLECTION_NAME,
    DOCUMENTS_DIR,
    CHROMA_DIR,
    MANIFESTS_DIR,
    OLLAMA_BASE_URL,
    PRIMARY_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    is_path_ignored,
)

from rag.embeddings import (
    EmbeddingService,
    OllamaConnectionError,
    EmbeddingModelNotFoundError,
    EmbeddingDimensionError,
    EmbeddingGenerationError,
)
from rag.vector_store import (
    VectorStoreManager,
    sanitize_metadata,
)
from rag.document_pipeline import DocumentPipeline
from rag.codebase_pipeline import CodebasePipeline
from rag.indexer import IncrementalIndexer
from rag.retriever import UnifiedRetriever, RetrievedChunk, RetrievalResponse, RetrievalStatus
from rag.prompt_builder import PromptBuilder
from rag.service import RAGService, RAGContextResult, get_rag_service, detect_retrieval_mode

__version__ = "0.5.0"
__all__ = [
    "logger",
    "DOCUMENT_COLLECTION_NAME",
    "CODEBASE_COLLECTION_NAME",
    "DOCUMENTS_DIR",
    "CHROMA_DIR",
    "MANIFESTS_DIR",
    "OLLAMA_BASE_URL",
    "PRIMARY_EMBEDDING_MODEL",
    "FALLBACK_EMBEDDING_MODEL",
    "is_path_ignored",
    "EmbeddingService",
    "OllamaConnectionError",
    "EmbeddingModelNotFoundError",
    "EmbeddingDimensionError",
    "EmbeddingGenerationError",
    "VectorStoreManager",
    "sanitize_metadata",
    "DocumentPipeline",
    "CodebasePipeline",
    "IncrementalIndexer",
    "UnifiedRetriever",
    "RetrievedChunk",
    "RetrievalResponse",
    "RetrievalStatus",
    "PromptBuilder",
    "RAGService",
    "RAGContextResult",
    "get_rag_service",
    "detect_retrieval_mode",
]



