"""
ChromaDB Persistent Vector Store Implementation for Unified RAG
- Uses chromadb.PersistentClient for in-process, persistent SQLite vector storage
- Manages isolated documents_collection and codebase_collection with cosine similarity
- Sanitizes metadata to ensure strict ChromaDB compatibility
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings

from rag import logger
from rag.config import (
    CHROMA_DIR,
    DOCUMENT_COLLECTION_NAME,
    CODEBASE_COLLECTION_NAME,
    CONVERSATION_COLLECTION_NAME,
)


def sanitize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Union[str, int, float, bool]]:
    """
    Sanitize metadata dictionary to conform strictly to ChromaDB schema requirements:
    Only primitive types (str, int, float, bool) are permitted.
    Lists, dictionaries, and None values are serialized or converted.
    """
    if not meta:
        return {}

    sanitized: Dict[str, Union[str, int, float, bool]] = {}
    for key, val in meta.items():
        if val is None:
            sanitized[key] = ""
        elif isinstance(val, (str, int, float, bool)):
            sanitized[key] = val
        elif isinstance(val, (list, tuple, set)):
            # If items are all strings, join them; otherwise JSON encode
            try:
                sanitized[key] = json.dumps(list(val), ensure_ascii=False)
            except Exception:
                sanitized[key] = str(val)
        elif isinstance(val, dict):
            try:
                sanitized[key] = json.dumps(val, ensure_ascii=False)
            except Exception:
                sanitized[key] = str(val)
        else:
            sanitized[key] = str(val)
    return sanitized


class VectorStoreManager:
    """
    Production-grade manager for persistent vector storage.
    Supports separate, isolated collections for documents and codebase.
    """

    def __init__(self, persist_dir: str = CHROMA_DIR):
        self.persist_dir = persist_dir
        self.client: Optional[chromadb.PersistentClient] = None
        self._collections: Dict[str, Any] = {}

    def initialize(self) -> None:
        """Initialize the persistent ChromaDB client and provision isolated collections."""
        os.makedirs(self.persist_dir, exist_ok=True)

        if self.client is None:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )

        # Provision independent collections with cosine similarity metric
        collection_metadata = {"hnsw:space": "cosine"}

        doc_col = self.client.get_or_create_collection(
            name=DOCUMENT_COLLECTION_NAME,
            metadata=collection_metadata
        )
        code_col = self.client.get_or_create_collection(
            name=CODEBASE_COLLECTION_NAME,
            metadata=collection_metadata
        )
        conv_col = self.client.get_or_create_collection(
            name=CONVERSATION_COLLECTION_NAME,
            metadata=collection_metadata
        )

        self._collections[DOCUMENT_COLLECTION_NAME] = doc_col
        self._collections[CODEBASE_COLLECTION_NAME] = code_col
        self._collections[CONVERSATION_COLLECTION_NAME] = conv_col

        logger.info(
            f"ChromaDB initialized at '{self.persist_dir}' with collections: "
            f"['{DOCUMENT_COLLECTION_NAME}', '{CODEBASE_COLLECTION_NAME}', '{CONVERSATION_COLLECTION_NAME}']"
        )

    def is_initialized(self) -> bool:
        """Check whether the vector store and isolated collections are initialized."""
        return (
            self.client is not None
            and DOCUMENT_COLLECTION_NAME in self._collections
            and CODEBASE_COLLECTION_NAME in self._collections
            and CONVERSATION_COLLECTION_NAME in self._collections
        )

    def get_collection(self, collection_name: str) -> Any:
        """Retrieve an initialized collection by name."""
        if not self.is_initialized():
            self.initialize()

        if collection_name not in self._collections:
            valid_names = list(self._collections.keys())
            raise ValueError(
                f"Unknown collection '{collection_name}'. Expected one of: {valid_names}"
            )
        return self._collections[collection_name]

    def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add or upsert vectorized chunks into the specified collection.
        Automatically sanitizes metadata fields to prevent ChromaDB schema rejection.
        """
        if not ids or not documents or not embeddings:
            return 0

        if len(ids) != len(documents) or len(ids) != len(embeddings):
            raise ValueError(
                f"Array length mismatch: ids={len(ids)}, documents={len(documents)}, "
                f"embeddings={len(embeddings)}"
            )

        col = self.get_collection(collection_name)

        sanitized_metadatas = None
        if metadatas is not None:
            if len(metadatas) != len(ids):
                raise ValueError(
                    f"Metadata length ({len(metadatas)}) does not match ids ({len(ids)})"
                )
            sanitized_metadatas = [sanitize_metadata(m) for m in metadatas]

        # Use upsert to guarantee idempotency
        col.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=sanitized_metadatas,
        )

        logger.debug(f"Upserted {len(ids)} items into '{collection_name}'")
        return len(ids)

    def query(
        self,
        collection_name: str,
        query_embeddings: Union[List[float], List[List[float]]],
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query a collection using vector embeddings and optional metadata filters.
        Returns a dictionary with 'ids', 'documents', 'metadatas', and 'distances'.
        """
        col = self.get_collection(collection_name)

        # Normalize query vector format (must be list of vectors for Chroma query)
        if query_embeddings and isinstance(query_embeddings[0], (int, float)):
            query_embeddings = [query_embeddings]  # type: ignore

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": max(1, n_results),
            "include": ["documents", "metadatas", "distances"]
        }

        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document

        results = col.query(**query_kwargs)
        return results

    def delete_by_ids(self, collection_name: str, ids: List[str]) -> bool:
        """Delete specific vectors from a collection by ID."""
        if not ids:
            return True

        col = self.get_collection(collection_name)
        try:
            col.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} IDs from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting IDs from '{collection_name}': {e}")
            raise

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Retrieve count, name, and metadata for a collection."""
        col = self.get_collection(collection_name)
        count = col.count()
        return {
            "name": collection_name,
            "count": count,
            "metadata": col.metadata,
        }

    def reset_collection(self, collection_name: str) -> None:
        """Reset (empty and recreate) an individual collection."""
        if self.client is None:
            self.initialize()

        logger.warning(f"Resetting collection '{collection_name}'")
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        self._collections[collection_name] = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def reset_all(self, include_memory: bool = False) -> None:
        """Clear collections. By default resets documents and codebase; optionally resets memory."""
        self.reset_collection(DOCUMENT_COLLECTION_NAME)
        self.reset_collection(CODEBASE_COLLECTION_NAME)
        if include_memory:
            self.reset_collection(CONVERSATION_COLLECTION_NAME)
