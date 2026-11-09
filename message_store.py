"""
Durable Message Storage, Persistence & Lifecycle Management
- SQLite-backed transactional message store with WAL mode
- Supports states: 'received', 'processing', 'persisted', 'completed', 'failed'
- Strict idempotency and deduplication by (conversation_id, message_id)
- Multi-agent provenance attribution and parent-child linking
- Voice-ready metadata (source: 'text' | 'voice')
- Synchronization tracking with ChromaDB conversational memory
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("rag.message_store")


class MessageStatus(str, Enum):
    RECEIVED = "received"
    PROCESSING = "processing"
    PERSISTED = "persisted"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageSource(str, Enum):
    TEXT = "text"
    VOICE = "voice"


@dataclass
class MessageRecord:
    conversation_id: str
    message_id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    source: str = "text"  # 'text' or 'voice'
    status: str = "received"  # received, processing, persisted, completed, failed
    parent_message_id: Optional[str] = None
    model_id: Optional[str] = None
    agent_role: Optional[str] = None
    pipeline_stage: Optional[str] = None
    error_message: Optional[str] = None
    indexed_in_memory: int = 0  # 0 = pending, 1 = indexed, -1 = failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "status": self.status,
            "parent_message_id": self.parent_message_id,
            "model_id": self.model_id,
            "agent_role": self.agent_role,
            "pipeline_stage": self.pipeline_stage,
            "error_message": self.error_message,
            "indexed_in_memory": self.indexed_in_memory,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    def to_ack(self) -> Dict[str, Any]:
        ack = {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "status": self.status,
            "source": self.source,
        }
        if self.parent_message_id:
            ack["parent_message_id"] = self.parent_message_id
        if self.error_message:
            ack["error_message"] = self.error_message
        return ack


class MessageStore:
    """
    Thread-safe, ACID-compliant persistent message store.
    Backed by SQLite in WAL (Write-Ahead-Logging) mode.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "messages.db")

        self.db_path = db_path
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create thread-local SQLite connection with WAL mode."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # In-memory SQLite databases (used for testing) need shared cache or persistent conn
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode; we manage explicit transactions
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode if on disk
            if self.db_path != ":memory:":
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.execute("PRAGMA busy_timeout=10000;")
                except Exception as e:
                    logger.warning(f"Could not configure WAL pragma: {e}")
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        conversation_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'text',
                        status TEXT NOT NULL,
                        parent_message_id TEXT,
                        model_id TEXT,
                        agent_role TEXT,
                        pipeline_stage TEXT,
                        error_message TEXT,
                        indexed_in_memory INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        metadata_json TEXT,
                        PRIMARY KEY (conversation_id, message_id)
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_message_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_indexing ON messages(indexed_in_memory);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);"
                )
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def register_message(
        self,
        conversation_id: str,
        message_id: str,
        role: str,
        content: str,
        source: str = "text",
        status: str = MessageStatus.RECEIVED.value,
        parent_message_id: Optional[str] = None,
        model_id: Optional[str] = None,
        agent_role: Optional[str] = None,
        pipeline_stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MessageRecord, bool]:
        """
        Register a message. If (conversation_id, message_id) already exists,
        returns the existing record and is_new=False without mutating it.
        """
        now = time.time()
        meta_json = json.dumps(metadata or {})
        source_str = source.value if hasattr(source, "value") else str(source)
        source_val = source_str if source_str in ("text", "voice") else "text"

        with self._lock:
            conn = self._get_connection()
            # Check existing first
            existing = self.get_message(conversation_id, message_id)
            if existing is not None:
                return existing, False

            conn.execute("BEGIN IMMEDIATE;")
            try:
                # Double-check inside transaction
                cur = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? AND message_id = ?",
                    (conversation_id, message_id),
                )
                row = cur.fetchone()
                if row is not None:
                    conn.execute("COMMIT;")
                    return self._row_to_record(row), False

                conn.execute(
                    """
                    INSERT INTO messages (
                        conversation_id, message_id, role, content, source, status,
                        parent_message_id, model_id, agent_role, pipeline_stage,
                        error_message, indexed_in_memory, created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        message_id,
                        role,
                        content,
                        source_val,
                        status,
                        parent_message_id,
                        model_id,
                        agent_role,
                        pipeline_stage,
                        None,
                        0,
                        now,
                        now,
                        meta_json,
                    ),
                )
                conn.execute("COMMIT;")

                record = MessageRecord(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    role=role,
                    content=content,
                    source=source_val,
                    status=status,
                    parent_message_id=parent_message_id,
                    model_id=model_id,
                    agent_role=agent_role,
                    pipeline_stage=pipeline_stage,
                    error_message=None,
                    indexed_in_memory=0,
                    created_at=now,
                    updated_at=now,
                    metadata=metadata or {},
                )
                return record, True
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def persist_user_message(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
        source: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageRecord:
        """
        Durable persistence of a user message with status='persisted'.
        Idempotent: if already exists, updates status to 'persisted' if previously 'received'.
        """
        with self._lock:
            existing = self.get_message(conversation_id, message_id)
            if existing is None:
                record, _ = self.register_message(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    role="user",
                    content=content,
                    source=source,
                    status=MessageStatus.PERSISTED.value,
                    model_id="user",
                    agent_role="user",
                    metadata=metadata,
                )
                return record
            else:
                if existing.status == MessageStatus.RECEIVED.value:
                    self.update_status(conversation_id, message_id, MessageStatus.PERSISTED.value)
                    existing.status = MessageStatus.PERSISTED.value
                return existing

    def persist_assistant_message(
        self,
        conversation_id: str,
        message_id: str,
        parent_message_id: str,
        content: str,
        model_id: str = "assistant",
        agent_role: str = "assistant",
        pipeline_stage: str = "",
        source: str = "text",
        status: str = MessageStatus.COMPLETED.value,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageRecord:
        """
        Durable persistence of an assistant message, linked to parent user message.
        """
        record, _ = self.register_message(
            conversation_id=conversation_id,
            message_id=message_id,
            role="assistant",
            content=content,
            source=source,
            status=status,
            parent_message_id=parent_message_id,
            model_id=model_id,
            agent_role=agent_role,
            pipeline_stage=pipeline_stage,
            metadata=metadata,
        )
        return record

    def update_status(
        self,
        conversation_id: str,
        message_id: str,
        status: str,
        error_message: Optional[str] = None,
        model_id: Optional[str] = None,
        agent_role: Optional[str] = None,
        pipeline_stage: Optional[str] = None,
    ) -> bool:
        """Atomically transition status of a message."""
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                updates = ["status = ?", "updated_at = ?"]
                params: List[Any] = [status, now]

                if error_message is not None:
                    updates.append("error_message = ?")
                    params.append(error_message)
                if model_id is not None:
                    updates.append("model_id = ?")
                    params.append(model_id)
                if agent_role is not None:
                    updates.append("agent_role = ?")
                    params.append(agent_role)
                if pipeline_stage is not None:
                    updates.append("pipeline_stage = ?")
                    params.append(pipeline_stage)

                params.extend([conversation_id, message_id])
                sql = f"UPDATE messages SET {', '.join(updates)} WHERE conversation_id = ? AND message_id = ?"
                cur = conn.execute(sql, params)
                conn.execute("COMMIT;")
                return cur.rowcount > 0
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def mark_completed(
        self,
        conversation_id: str,
        user_message_id: str,
        assistant_message_id: Optional[str] = None,
    ) -> bool:
        """Mark user message (and assistant message if provided) as completed."""
        with self._lock:
            ok = self.update_status(conversation_id, user_message_id, MessageStatus.COMPLETED.value)
            if assistant_message_id:
                self.update_status(conversation_id, assistant_message_id, MessageStatus.COMPLETED.value)
            return ok

    def mark_failed(
        self,
        conversation_id: str,
        message_id: str,
        error_message: str,
    ) -> bool:
        """Mark message as failed with error details."""
        return self.update_status(
            conversation_id,
            message_id,
            status=MessageStatus.FAILED.value,
            error_message=error_message,
        )

    def get_message(self, conversation_id: str, message_id: str) -> Optional[MessageRecord]:
        """Fetch a single message by composite primary key."""
        with self._lock:
            conn = self._get_connection()
            cur = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? AND message_id = ?",
                (conversation_id, message_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def get_assistant_response_for_user_message(
        self, conversation_id: str, user_message_id: str
    ) -> Optional[MessageRecord]:
        """Fetch completed assistant response associated with a user message."""
        with self._lock:
            conn = self._get_connection()
            cur = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ? AND parent_message_id = ? AND role = 'assistant'
                ORDER BY created_at DESC LIMIT 1
                """,
                (conversation_id, user_message_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def get_conversation_messages(
        self, conversation_id: str, limit: int = 200
    ) -> List[MessageRecord]:
        """Fetch all messages in a conversation ordered chronologically."""
        with self._lock:
            conn = self._get_connection()
            cur = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC LIMIT ?
                """,
                (conversation_id, limit),
            )
            return [self._row_to_record(r) for r in cur.fetchall()]

    def get_unindexed_messages(self, limit: int = 50) -> List[MessageRecord]:
        """Fetch messages that have not yet been indexed in vector memory (indexed_in_memory == 0)."""
        with self._lock:
            conn = self._get_connection()
            cur = conn.execute(
                """
                SELECT * FROM messages
                WHERE indexed_in_memory = 0 AND status IN ('persisted', 'completed')
                ORDER BY created_at ASC LIMIT ?
                """,
                (limit,),
            )
            return [self._row_to_record(r) for r in cur.fetchall()]

    def mark_indexed_in_memory(
        self, conversation_id: str, message_id: str, success: bool = True
    ) -> bool:
        """Record whether message vector was successfully indexed into ChromaDB."""
        val = 1 if success else -1
        with self._lock:
            conn = self._get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = conn.execute(
                    "UPDATE messages SET indexed_in_memory = ?, updated_at = ? WHERE conversation_id = ? AND message_id = ?",
                    (val, time.time(), conversation_id, message_id),
                )
                conn.execute("COMMIT;")
                return cur.rowcount > 0
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def count_messages(self, conversation_id: Optional[str] = None) -> int:
        """Count messages globally or in a conversation."""
        with self._lock:
            conn = self._get_connection()
            if conversation_id:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                )
            else:
                cur = conn.execute("SELECT COUNT(*) FROM messages")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def reset_all(self) -> None:
        """Purge all messages (used for test isolation)."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute("DELETE FROM messages;")
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def close(self) -> None:
        """Close connection on current thread."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def close(self) -> None:
        """Close thread-local SQLite connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _row_to_record(self, row: sqlite3.Row) -> MessageRecord:
        meta = {}
        if row["metadata_json"]:
            try:
                meta = json.loads(row["metadata_json"])
            except Exception:
                meta = {}

        return MessageRecord(
            conversation_id=row["conversation_id"],
            message_id=row["message_id"],
            role=row["role"],
            content=row["content"],
            source=row["source"],
            status=row["status"],
            parent_message_id=row["parent_message_id"],
            model_id=row["model_id"],
            agent_role=row["agent_role"],
            pipeline_stage=row["pipeline_stage"],
            error_message=row["error_message"],
            indexed_in_memory=row["indexed_in_memory"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )


_global_message_store: Optional[MessageStore] = None
_store_lock = threading.Lock()


def get_message_store(db_path: Optional[str] = None) -> MessageStore:
    """Singleton factory for MessageStore."""
    global _global_message_store
    with _store_lock:
        if _global_message_store is None or (db_path and _global_message_store.db_path != db_path):
            _global_message_store = MessageStore(db_path=db_path)
        return _global_message_store
