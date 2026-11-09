"""
Conversational Memory Vector Database & Selective Historical RAG
- Dedicated ChromaDB collection ('conversation_memory_collection') isolated from documents and codebase
- Stores original message content with full agent/model execution metadata
- Background non-blocking queue for zero-latency live inference dispatch
- Strict idempotency and deduplication
- Semantic retrieval with metadata filtering (conversation_id, agent_role, project_id, model_id)
- Context window pruning strategy (separating recent turns from historical memory)
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
import queue
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from rag import logger
from rag.config import (
    CHROMA_DIR,
    CONVERSATION_COLLECTION_NAME,
    MEMORY_DISTANCE_THRESHOLD,
    MEMORY_TOP_K,
    RECENT_CONTEXT_MAX_CHARS,
    RECENT_CONTEXT_TURNS,
)
from rag.embeddings import EmbeddingService
from rag.retriever import RetrievedChunk
from rag.vector_store import VectorStoreManager, sanitize_metadata


@dataclass
class ConversationMessage:
    """Structured representation of a single conversation turn or pipeline stage."""

    conversation_id: str
    message_id: str
    role: str  # "user" or "assistant"
    content: str
    source: str = "text"  # "text" or "voice"
    model_id: str = "unknown"
    agent_role: str = "unknown"  # "architect", "coder", "general_assistant", "vision_assistant", "cloud_assistant", "user"
    pipeline_stage: str = ""  # "architect", "coder", "vision", "chat", "gemini", or ""
    project_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if self.source not in ("text", "voice"):
            self.source = "text"

    def to_metadata(self) -> Dict[str, Any]:
        """Convert fields to a sanitized dictionary for ChromaDB storage."""
        return {
            "conversation_id": str(self.conversation_id),
            "message_id": str(self.message_id),
            "role": str(self.role),
            "source": str(self.source or "text"),
            "model_id": str(self.model_id or "unknown"),
            "agent_role": str(self.agent_role or "unknown"),
            "pipeline_stage": str(self.pipeline_stage or ""),
            "project_id": str(self.project_id or ""),
            "timestamp": str(self.timestamp),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full dictionary representation."""
        return asdict(self)


class ConversationMemoryStore:
    """
    Dedicated manager for historical conversational memory.
    Coordinates vector storage, asynchronous background indexing, and semantic retrieval.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_service: Optional[EmbeddingService] = None,
        message_store: Optional[Any] = None,
        collection_name: str = CONVERSATION_COLLECTION_NAME,
        persist_dir: str = CHROMA_DIR,
    ):
        self.vector_store = vector_store or VectorStoreManager(persist_dir=persist_dir)
        self.embedding_service = embedding_service or EmbeddingService()
        if message_store is None:
            if persist_dir == CHROMA_DIR:
                try:
                    from message_store import get_message_store
                    self.message_store = get_message_store()
                except Exception:
                    self.message_store = None
            else:
                self.message_store = None
        else:
            self.message_store = message_store
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        self._indexed_ids: Set[str] = set()
        self._lock = threading.Lock()

        # Background non-blocking worker queue
        self._queue: queue.Queue[Optional[ConversationMessage]] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._background_worker,
            name="ConversationMemoryWorker",
            daemon=True,
        )
        self._worker_thread.start()

        # Reconcile any unindexed messages from durable store
        try:
            self.reconcile_unindexed_messages()
        except Exception as e:
            logger.debug(f"Initial unindexed reconciliation skipped: {e}")

    def _get_vector_id(self, msg: ConversationMessage) -> str:
        """Deterministic unique ID for vector storage and deduplication."""
        return f"{msg.conversation_id}_{msg.message_id}"

    def store_message_sync(self, msg: ConversationMessage) -> bool:
        """
        Store and embed a single conversation message synchronously.
        Ensures idempotent upsert and metadata sanitization.
        """
        if not msg.content or not msg.content.strip():
            return False

        vector_id = self._get_vector_id(msg)

        with self._lock:
            if vector_id in self._indexed_ids:
                return True

        try:
            # Generate vector embedding for original message content
            embeddings = self.embedding_service.embed_documents([msg.content])
            if not embeddings:
                logger.error(f"Failed to generate embedding for message '{vector_id}'")
                if self.message_store:
                    try:
                        self.message_store.mark_indexed_in_memory(
                            msg.conversation_id, msg.message_id, success=False
                        )
                    except Exception:
                        pass
                return False

            raw_meta = msg.to_metadata()
            sanitized_meta = sanitize_metadata(raw_meta)

            self.vector_store.add_documents(
                collection_name=self.collection_name,
                ids=[vector_id],
                documents=[msg.content],
                embeddings=embeddings,
                metadatas=[sanitized_meta],
            )

            with self._lock:
                self._indexed_ids.add(vector_id)

            if self.message_store:
                try:
                    self.message_store.mark_indexed_in_memory(
                        msg.conversation_id, msg.message_id, success=True
                    )
                except Exception as me:
                    logger.debug(f"Failed to record indexing status in message_store: {me}")

            logger.debug(
                f"ConversationMemoryStore: Stored message '{vector_id}' "
                f"(role={msg.role}, agent={msg.agent_role}, model={msg.model_id})"
            )
            return True

        except Exception as e:
            if self.message_store:
                try:
                    self.message_store.mark_indexed_in_memory(
                        msg.conversation_id, msg.message_id, success=False
                    )
                except Exception:
                    pass
            logger.error(f"ConversationMemoryStore error storing message '{vector_id}': {e}")
            return False

    def store_message_async(self, msg: ConversationMessage) -> None:
        """
        Queue message for background indexing.
        Guarantees zero blocking on the live HTTP/SSE streaming path.
        """
        if not msg.content or not msg.content.strip():
            return

        vector_id = self._get_vector_id(msg)
        with self._lock:
            if vector_id in self._indexed_ids:
                return

        self._queue.put(msg)

    def store_messages_batch(self, msgs: List[ConversationMessage]) -> int:
        """
        Store multiple messages in a single batch operation for performance.
        """
        valid_msgs: List[ConversationMessage] = []
        with self._lock:
            for m in msgs:
                if m.content and m.content.strip():
                    vid = self._get_vector_id(m)
                    if vid not in self._indexed_ids:
                        valid_msgs.append(m)

        if not valid_msgs:
            return 0

        try:
            texts = [m.content for m in valid_msgs]
            embeddings = self.embedding_service.embed_documents(texts)
            if len(embeddings) != len(valid_msgs):
                logger.error("Embedding count mismatch during batch memory storage")
                return 0

            ids = [self._get_vector_id(m) for m in valid_msgs]
            metadatas = [sanitize_metadata(m.to_metadata()) for m in valid_msgs]

            self.vector_store.add_documents(
                collection_name=self.collection_name,
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            with self._lock:
                for vid in ids:
                    self._indexed_ids.add(vid)

            if self.message_store:
                for m in valid_msgs:
                    try:
                        self.message_store.mark_indexed_in_memory(
                            m.conversation_id, m.message_id, success=True
                        )
                    except Exception:
                        pass

            return len(valid_msgs)

        except Exception as e:
            logger.error(f"ConversationMemoryStore error during batch storage: {e}")
            return 0

    def reconcile_unindexed_messages(self, limit: int = 50) -> int:
        """
        Recover and index any messages that were persisted in SQLite but not yet indexed in vector store
        (e.g., following an application restart or crash).
        """
        if not self.message_store:
            return 0
        try:
            unindexed = self.message_store.get_unindexed_messages(limit=limit)
            for rec in unindexed:
                msg = ConversationMessage(
                    conversation_id=rec.conversation_id,
                    message_id=rec.message_id,
                    role=rec.role,
                    content=rec.content,
                    source=rec.source,
                    model_id=rec.model_id or "unknown",
                    agent_role=rec.agent_role or "unknown",
                    pipeline_stage=rec.pipeline_stage or "",
                )
                self.store_message_async(msg)
            return len(unindexed)
        except Exception as e:
            logger.warning(f"Could not reconcile unindexed messages: {e}")
            return 0

    def _background_worker(self) -> None:
        """Worker loop processing queued messages asynchronously."""
        while not self._stop_event.is_set():
            try:
                msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if msg is None:
                self._queue.task_done()
                break

            try:
                self.store_message_sync(msg)
            except Exception as e:
                logger.error(f"Background indexing worker error: {e}")
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for queued messages to finish processing."""
        start = time.time()
        while not self._queue.empty():
            if time.time() - start > timeout:
                return False
            time.sleep(0.05)
        return True

    def retrieve_memory(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_role: Optional[str] = None,
        model_id: Optional[str] = None,
        top_k: int = MEMORY_TOP_K,
        distance_threshold: float = MEMORY_DISTANCE_THRESHOLD,
    ) -> List[RetrievedChunk]:
        """
        Execute semantic similarity search over historical conversational memory.
        Applies metadata filters when provided (e.g. scoped to conversation or agent).
        Returns retrieved chunks formatted as RetrievedChunk objects.
        """
        if not query or not query.strip():
            return []

        try:
            query_embedding = self.embedding_service.embed_query(query.strip())
            if not query_embedding:
                logger.warning("ConversationMemoryStore: Could not generate query embedding")
                return []

            # Construct ChromaDB metadata filter
            conditions = []
            if conversation_id:
                conditions.append({"conversation_id": str(conversation_id)})
            if project_id:
                conditions.append({"project_id": str(project_id)})
            if agent_role:
                conditions.append({"agent_role": str(agent_role)})
            if model_id:
                conditions.append({"model_id": str(model_id)})

            where_clause = None
            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}

            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_embeddings=[query_embedding],
                n_results=max(1, top_k * 2),  # Oversample to allow post-threshold filtering
                where=where_clause,
            )

            ids_list = results.get("ids", [[]])[0]
            docs_list = results.get("documents", [[]])[0]
            metas_list = results.get("metadatas", [[]])[0]
            dists_list = results.get("distances", [[]])[0]

            retrieved: List[RetrievedChunk] = []
            for doc_id, doc_text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
                if dist > distance_threshold:
                    continue  # Filter out weak matches

                sim = max(0.0, min(1.0, 1.0 - dist))
                chunk = RetrievedChunk(
                    content=doc_text,
                    source_path=f"conversation:{meta.get('conversation_id', 'unknown')}",
                    file_name=f"msg_{meta.get('message_id', doc_id)}",
                    category="memory",
                    distance=float(dist),
                    similarity=float(sim),
                    symbol_name=meta.get("agent_role", "unknown"),
                    chunk_id=str(doc_id),
                    role=meta.get("role", "unknown"),
                    model_id=meta.get("model_id", "unknown"),
                    agent_role=meta.get("agent_role", "unknown"),
                    pipeline_stage=meta.get("pipeline_stage", ""),
                    timestamp=meta.get("timestamp", ""),
                    conversation_id=meta.get("conversation_id", ""),
                )
                retrieved.append(chunk)
                if len(retrieved) >= top_k:
                    break

            return retrieved

        except Exception as e:
            logger.error(f"ConversationMemoryStore error during retrieval: {e}")
            return []

    @staticmethod
    def filter_recent_context(
        messages: List[Dict[str, Any]],
        max_turns: int = RECENT_CONTEXT_TURNS,
        max_chars: int = RECENT_CONTEXT_MAX_CHARS,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Partition conversation messages into recent context (for live model input)
        and older history (preserved in memory store only).

        Structure of messages:
        - The last message is typically the current user turn.
        - Preceding messages are prior conversational turns.

        Returns:
            (recent_messages, older_messages)
        """
        if not messages:
            return [], []

        # If only 1 message (the current query), nothing is older
        if len(messages) <= 1:
            return list(messages), []

        prior_messages = [m for m in messages[:-1] if m.get("role") != "system"]
        current_user_msg = messages[-1]

        # max_turns represents conversational pairs (user + assistant) = 2 messages per turn
        max_prior_count = max(1, max_turns * 2)

        if len(prior_messages) <= max_prior_count:
            candidate_recent = list(prior_messages)
            older_messages: List[Dict[str, Any]] = []
        else:
            cutoff = len(prior_messages) - max_prior_count
            older_messages = prior_messages[:cutoff]
            candidate_recent = prior_messages[cutoff:]

        # Apply character cap to candidate_recent (to prevent massive code dumps from blowing context)
        while len(candidate_recent) > 2:
            total_chars = sum(len(str(m.get("content", ""))) for m in candidate_recent)
            if total_chars <= max_chars:
                break
            # Drop the oldest turn in candidate_recent
            older_messages.append(candidate_recent.pop(0))

        recent_messages = candidate_recent + [current_user_msg]
        return recent_messages, older_messages

    def close(self) -> None:
        """Stop background worker cleanly."""
        self._stop_event.set()
        self._queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)


# ── Global Singleton Accessor ────────────────────────────────────────────────
_memory_store_instance: Optional[ConversationMemoryStore] = None


def get_conversation_store(
    vector_store: Optional[VectorStoreManager] = None,
    embedding_service: Optional[EmbeddingService] = None,
    reset: bool = False,
) -> ConversationMemoryStore:
    """Return singleton ConversationMemoryStore instance."""
    global _memory_store_instance
    if _memory_store_instance is None or reset:
        if _memory_store_instance is not None:
            try:
                _memory_store_instance.close()
            except Exception:
                pass
        _memory_store_instance = ConversationMemoryStore(
            vector_store=vector_store,
            embedding_service=embedding_service,
        )
    return _memory_store_instance
