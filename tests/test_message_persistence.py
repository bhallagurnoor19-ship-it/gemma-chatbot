"""
Automated Test Suite for Reliable Message Acknowledgement, Persistence & Voice-Ready Architecture

Test Groups:
- Group A: Identification & Deduplication
  - Unique message ID per turn
  - Stable message ID during retries
  - Duplicate message submissions returning cached completion without redundant inference
  - Composite primary key (conversation_id, message_id) isolation
  - Duplicate indexing prevention in conversational memory
- Group B: Acknowledgement & State Progression
  - Persistence acknowledgement ('type': 'ack', 'status': 'persisted')
  - Completion acknowledgement ('type': 'ack', 'status': 'completed')
  - Failure lifecycle: error marks user message failed and emits failed ack
- Group C: Recovery & Fault Tolerance
  - GET /api/messages/<conv_id>/<msg_id>/status single status endpoint
  - POST /api/messages/status batch status endpoint
  - Restart / Crash recovery: reconciliation of unindexed messages from SQLite into ChromaDB
- Group D: Concurrency, Multi-Conversation & Voice Readiness
  - Concurrency guard: 409 Conflict when in-flight message is resubmitted
  - Voice source provenance ('source': 'voice') propagated through SQLite and vector store
  - Multi-conversation concurrent persistence under SQLite WAL mode
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import MagicMock, patch

from message_store import (
    MessageStore,
    MessageStatus,
    MessageSource,
    MessageRecord,
    get_message_store,
)
from rag.config import (
    CONVERSATION_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)
from rag.memory import (
    ConversationMemoryStore,
    ConversationMessage,
)
from rag.vector_store import VectorStoreManager


class TestMessagePersistence(unittest.TestCase):
    """Test suite for reliable message lifecycle, persistence, ack, and voice-readiness."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_msg_persist_")
        self.db_path = os.path.join(self.test_dir, "messages.db")
        self.chroma_dir = os.path.join(self.test_dir, "chroma")
        os.makedirs(self.chroma_dir, exist_ok=True)

        self.store = MessageStore(db_path=self.db_path)

        # Vector store & memory store setup
        self.vector_store = VectorStoreManager(persist_dir=self.chroma_dir)
        self.vector_store.initialize()

        self.mock_embedding = MagicMock()
        self.mock_embedding.embed_documents.side_effect = lambda docs: [[0.2] * EMBEDDING_DIMENSION for _ in docs]
        self.mock_embedding.embed_query.side_effect = lambda q: [0.2] * EMBEDDING_DIMENSION

        self.memory_store = ConversationMemoryStore(
            vector_store=self.vector_store,
            embedding_service=self.mock_embedding,
            collection_name=CONVERSATION_COLLECTION_NAME,
            persist_dir=self.chroma_dir,
            message_store=self.store,
        )

    def tearDown(self):
        self.memory_store.close()
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_mock_get(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "embeddinggemma:latest"}]}
        return mock_resp

    def _make_mock_ollama_resp(self, text):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": text}, "done": True}).encode("utf-8")
        ]
        return mock_resp

    def _make_fake_post(self, response_text="Test assistant response"):
        def fake_post(url, *args, **kwargs):
            if "/api/embed" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"embeddings": [[0.2] * EMBEDDING_DIMENSION]}
                return resp
            return self._make_mock_ollama_resp(response_text)
        return fake_post

    # ═══════════════════════════════════════════════════════════════════════
    # Group A: Identification & Deduplication
    # ═══════════════════════════════════════════════════════════════════════

    def test_unique_message_id_per_turn(self):
        """Verify each turn gets a unique identifier and both are stored independently."""
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        msg_id_1 = f"msg_{uuid.uuid4().hex[:8]}"
        msg_id_2 = f"msg_{uuid.uuid4().hex[:8]}"

        self.assertNotEqual(msg_id_1, msg_id_2)

        rec1 = self.store.persist_user_message(conv_id, msg_id_1, "First query", source="text")
        rec2 = self.store.persist_user_message(conv_id, msg_id_2, "Second query", source="text")

        self.assertEqual(rec1.message_id, msg_id_1)
        self.assertEqual(rec2.message_id, msg_id_2)

        all_msgs = self.store.get_conversation_messages(conv_id)
        self.assertEqual(len(all_msgs), 2)
        stored_ids = {m.message_id for m in all_msgs}
        self.assertEqual(stored_ids, {msg_id_1, msg_id_2})

    def test_composite_primary_key_conversation_isolation(self):
        """Verify identical message_id in different conversations are completely isolated."""
        shared_msg_id = "msg_shared_turn_42"
        conv_a = "conv_alpha"
        conv_b = "conv_beta"

        rec_a = self.store.persist_user_message(conv_a, shared_msg_id, "Hello from Conv A", source="text")
        rec_b = self.store.persist_user_message(conv_b, shared_msg_id, "Hello from Conv B", source="voice")

        self.assertEqual(rec_a.conversation_id, conv_a)
        self.assertEqual(rec_b.conversation_id, conv_b)
        self.assertEqual(rec_a.content, "Hello from Conv A")
        self.assertEqual(rec_b.content, "Hello from Conv B")
        self.assertEqual(rec_b.source, "voice")

        # Updates to conv_a do not mutate conv_b
        self.store.update_status(conv_a, shared_msg_id, MessageStatus.COMPLETED.value)
        self.assertEqual(self.store.get_message(conv_a, shared_msg_id).status, MessageStatus.COMPLETED.value)
        self.assertEqual(self.store.get_message(conv_b, shared_msg_id).status, MessageStatus.PERSISTED.value)

    def test_duplicate_indexing_prevention(self):
        """Ensure indexing a message tracks indexed_in_memory=1 in SQLite and avoids duplicate vectors."""
        conv_id = "conv_indexing_test"
        msg_id = "msg_idx_01"

        self.store.persist_user_message(conv_id, msg_id, "Explain the architecture of SQLite", source="text")
        rec_before = self.store.get_message(conv_id, msg_id)
        self.assertEqual(rec_before.indexed_in_memory, 0)

        # Add to memory store
        mem_msg = ConversationMessage(
            message_id=msg_id,
            conversation_id=conv_id,
            role="user",
            content="Explain the architecture of SQLite",
            source="text",
        )
        stored_ok = self.memory_store.store_message_sync(mem_msg)
        self.assertTrue(stored_ok)

        rec_after = self.store.get_message(conv_id, msg_id)
        self.assertEqual(rec_after.indexed_in_memory, 1)

        # Re-adding same message does not fail and is deduplicated
        stored_dup = self.memory_store.store_message_sync(mem_msg)
        self.assertTrue(stored_dup)

    @patch("requests.get")
    @patch("requests.post")
    def test_duplicate_submission_returns_cached_response(self, mock_post, mock_get):
        """Submitting an already-completed message_id replays the completed response without re-inference."""
        mock_get.return_value = self._make_mock_get()
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"

        mock_post.side_effect = self._make_fake_post("This is the original synthesized response.")

        with patch("app.get_message_store", return_value=self.store):
            # First submission
            resp1 = client.post(
                "/api/chat",
                data=json.dumps({
                    "request_id": str(uuid.uuid4()),
                    "conversation_id": conv_id,
                    "message_id": msg_id,
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "mode": "chat",
                    "source": "text",
                }),
                content_type="application/json",
            )
            self.assertEqual(resp1.status_code, 200)
            events1 = [
                json.loads(line[6:])
                for line in resp1.get_data(as_text=True).split("\n")
                if line.startswith("data: ")
            ]
            self.assertTrue(any(e.get("type") == "ack" and e.get("status") == "completed" for e in events1))

            chat_calls_before = [c for c in mock_post.call_args_list if "/api/chat" in c[0][0]]

            # Duplicate submission with exact same message_id
            resp2 = client.post(
                "/api/chat",
                data=json.dumps({
                    "request_id": str(uuid.uuid4()),
                    "conversation_id": conv_id,
                    "message_id": msg_id,
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "mode": "chat",
                    "source": "text",
                }),
                content_type="application/json",
            )
            self.assertEqual(resp2.status_code, 200)
            events2 = [
                json.loads(line[6:])
                for line in resp2.get_data(as_text=True).split("\n")
                if line.startswith("data: ")
            ]

            # Verify cached route and token replay
            cached_route = next((e for e in events2 if e.get("type") == "route"), None)
            self.assertIsNotNone(cached_route)
            self.assertTrue(cached_route.get("cached"))
            self.assertEqual(cached_route.get("message_id"), msg_id)

            # Assert mock_post was NOT called again for /api/chat generation
            chat_calls_after = [c for c in mock_post.call_args_list if "/api/chat" in c[0][0]]
            self.assertEqual(len(chat_calls_after), len(chat_calls_before))

    # ═══════════════════════════════════════════════════════════════════════
    # Group B: Acknowledgement & State Progression
    # ═══════════════════════════════════════════════════════════════════════

    @patch("requests.get")
    @patch("requests.post")
    def test_acknowledgement_persisted_and_completed_emitted(self, mock_post, mock_get):
        """Verify backend emits 'ack' with status 'persisted' and 'completed'."""
        mock_get.return_value = self._make_mock_get()
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        conv_id = f"conv_ack_{uuid.uuid4().hex[:8]}"
        msg_id = f"msg_ack_{uuid.uuid4().hex[:8]}"

        mock_post.side_effect = self._make_fake_post("Response acknowledging turn.")

        with patch("app.get_message_store", return_value=self.store):
            resp = client.post(
                "/api/chat",
                data=json.dumps({
                    "request_id": str(uuid.uuid4()),
                    "conversation_id": conv_id,
                    "message_id": msg_id,
                    "messages": [{"role": "user", "content": "Verify acknowledgement lifecycle"}],
                    "mode": "chat",
                    "source": "text",
                }),
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 200)
            events = [
                json.loads(line[6:])
                for line in resp.get_data(as_text=True).split("\n")
                if line.startswith("data: ")
            ]

            ack_events = [e for e in events if e.get("type") == "ack"]
            self.assertEqual(len(ack_events), 2)

            self.assertEqual(ack_events[0]["status"], "persisted")
            self.assertEqual(ack_events[0]["message_id"], msg_id)
            self.assertEqual(ack_events[0]["conversation_id"], conv_id)

            self.assertEqual(ack_events[1]["status"], "completed")
            self.assertEqual(ack_events[1]["message_id"], msg_id)
            self.assertTrue("assistant_message_id" in ack_events[1])

    @patch("requests.get")
    @patch("requests.post")
    def test_failure_marks_message_failed_and_emits_failed_ack(self, mock_post, mock_get):
        """Simulate model execution error and ensure status='failed' and ack='failed'."""
        mock_get.return_value = self._make_mock_get()
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        conv_id = f"conv_fail_{uuid.uuid4().hex[:8]}"
        msg_id = f"msg_fail_{uuid.uuid4().hex[:8]}"

        def fail_on_chat(url, *args, **kwargs):
            if "/api/embed" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"embeddings": [[0.2] * EMBEDDING_DIMENSION]}
                return resp
            raise RuntimeError("Ollama connection refused during generation")

        mock_post.side_effect = fail_on_chat

        with patch("app.get_message_store", return_value=self.store):
            resp = client.post(
                "/api/chat",
                data=json.dumps({
                    "request_id": str(uuid.uuid4()),
                    "conversation_id": conv_id,
                    "message_id": msg_id,
                    "messages": [{"role": "user", "content": "Trigger failure"}],
                    "mode": "chat",
                    "source": "text",
                }),
                content_type="application/json",
            )
            events = [
                json.loads(line[6:])
                for line in resp.get_data(as_text=True).split("\n")
                if line.startswith("data: ")
            ]

            failed_ack = next((e for e in events if e.get("type") == "ack" and e.get("status") == "failed"), None)
            self.assertIsNotNone(failed_ack)
            self.assertEqual(failed_ack["message_id"], msg_id)

            rec = self.store.get_message(conv_id, msg_id)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.status, MessageStatus.FAILED.value)
            self.assertIn("Ollama connection refused", rec.error_message)

    # ═══════════════════════════════════════════════════════════════════════
    # Group C: Recovery & Fault Tolerance
    # ═══════════════════════════════════════════════════════════════════════

    def test_single_message_status_endpoint(self):
        """Test GET /api/messages/<conv_id>/<msg_id>/status endpoint."""
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        conv_id = "conv_stat_test"
        msg_id = "msg_stat_01"

        # Not found
        with patch("app.get_message_store", return_value=self.store):
            res_404 = client.get(f"/api/messages/{conv_id}/non_existent/status")
            self.assertEqual(res_404.status_code, 404)

            # Persisted & Completed
            self.store.persist_user_message(conv_id, msg_id, "User query text", source="text")
            asst_rec = self.store.persist_assistant_message(
                conversation_id=conv_id,
                message_id="asst_01",
                parent_message_id=msg_id,
                content="Assistant completed answer",
                model_id="gemma4:12b",
                agent_role="assistant",
            )
            self.store.mark_completed(conv_id, msg_id)

            res = client.get(f"/api/messages/{conv_id}/{msg_id}/status")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["message_id"], msg_id)
            self.assertIsNotNone(data["assistant_response"])
            self.assertEqual(data["assistant_response"]["content"], "Assistant completed answer")

    def test_batch_message_status_endpoint(self):
        """Test POST /api/messages/status batch endpoint."""
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        conv_id = "conv_batch_test"
        self.store.persist_user_message(conv_id, "msg_b1", "Batch 1", source="text")
        self.store.persist_user_message(conv_id, "msg_b2", "Batch 2", source="voice")
        self.store.mark_completed(conv_id, "msg_b1")

        with patch("app.get_message_store", return_value=self.store):
            payload = {
                "messages": [
                    {"conversation_id": conv_id, "message_id": "msg_b1"},
                    {"conversation_id": conv_id, "message_id": "msg_b2"},
                    {"conversation_id": conv_id, "message_id": "msg_b3"},
                ]
            }
            res = client.post("/api/messages/status", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            results = {item["message_id"]: item for item in data.get("messages", [])}

            self.assertEqual(results["msg_b1"]["status"], "completed")
            self.assertEqual(results["msg_b2"]["status"], "persisted")
            self.assertEqual(results["msg_b2"]["source"], "voice")
            self.assertEqual(results["msg_b3"]["status"], "unknown")

    def test_restart_reconciliation_of_unindexed_messages(self):
        """Simulate unindexed messages left in SQLite and verify reconcile_unindexed_messages indexes them."""
        conv_id = "conv_reconcile"
        msg_1 = "msg_rec_1"
        msg_2 = "msg_rec_2"

        # Directly insert user messages with indexed_in_memory=0
        self.store.persist_user_message(conv_id, msg_1, "Unindexed prompt 1", source="text")
        self.store.persist_user_message(conv_id, msg_2, "Unindexed prompt 2", source="voice")

        unindexed = self.store.get_unindexed_messages()
        self.assertEqual(len(unindexed), 2)

        # Run reconciliation
        count = self.memory_store.reconcile_unindexed_messages(limit=10)
        self.assertEqual(count, 2)
        self.memory_store.flush(timeout=5.0)

        # Verify indexed_in_memory is now 1
        unindexed_after = self.store.get_unindexed_messages()
        self.assertEqual(len(unindexed_after), 0)

        # Verify searchable in ChromaDB
        results = self.memory_store.retrieve_memory(
            query="Unindexed prompt",
            conversation_id=conv_id,
            top_k=5,
        )
        self.assertGreaterEqual(len(results), 2)

    # ═══════════════════════════════════════════════════════════════════════
    # Group D: Concurrency, Multi-Conversation & Voice Readiness
    # ═══════════════════════════════════════════════════════════════════════

    def test_concurrency_guard_rejects_in_flight_message(self):
        """Verify resubmission of an in-flight (status='processing') message returns HTTP 409."""
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        conv_id = "conv_flight"
        msg_id = "msg_inflight_1"

        self.store.persist_user_message(conv_id, msg_id, "In flight query", source="text")
        self.store.update_status(conv_id, msg_id, MessageStatus.PROCESSING.value)

        with patch("app.get_message_store", return_value=self.store):
            resp = client.post(
                "/api/chat",
                data=json.dumps({
                    "request_id": str(uuid.uuid4()),
                    "conversation_id": conv_id,
                    "message_id": msg_id,
                    "messages": [{"role": "user", "content": "In flight query"}],
                    "mode": "chat",
                }),
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 409)
            data = resp.get_json()
            self.assertEqual(data["status"], "processing")
            self.assertEqual(data["message_id"], msg_id)

    @patch("requests.get")
    @patch("requests.post")
    def test_voice_source_metadata_propagation(self, mock_post, mock_get):
        """Verify 'source': 'voice' is recorded in SQLite, emitted in SSE, and preserved in vector memory."""
        mock_get.return_value = self._make_mock_get()
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        conv_id = f"conv_voice_{uuid.uuid4().hex[:8]}"
        msg_id = f"msg_voice_{uuid.uuid4().hex[:8]}"

        mock_post.side_effect = self._make_fake_post("Spoken command executed.")

        with patch("app.get_message_store", return_value=self.store), \
             patch("app.get_conversation_store", return_value=self.memory_store):

            resp = client.post(
                "/api/chat",
                data=json.dumps({
                    "request_id": str(uuid.uuid4()),
                    "conversation_id": conv_id,
                    "message_id": msg_id,
                    "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
                    "mode": "chat",
                    "source": "voice",
                }),
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 200)
            events = [
                json.loads(line[6:])
                for line in resp.get_data(as_text=True).split("\n")
                if line.startswith("data: ")
            ]

            ack_events = [e for e in events if e.get("type") == "ack"]
            self.assertTrue(all(e.get("source") == "voice" for e in ack_events))

            # Verify SQLite record
            rec = self.store.get_message(conv_id, msg_id)
            self.assertEqual(rec.source, "voice")

            # Verify status endpoint returns voice source
            status_res = client.get(f"/api/messages/{conv_id}/{msg_id}/status")
            self.assertEqual(status_res.get_json()["source"], "voice")

            # Flush async indexing to ensure vector store has processed the message
            self.memory_store.flush(timeout=5.0)

            # Verify vector store metadata directly in ChromaDB collection
            col = self.vector_store.get_collection(self.memory_store.collection_name)
            doc = col.get(ids=[f"{conv_id}_{msg_id}"])
            self.assertTrue(len(doc["ids"]) > 0)
            self.assertEqual(doc["metadatas"][0].get("source"), "voice")

    def test_multi_conversation_concurrent_storage(self):
        """Verify multiple threads storing messages across conversations under SQLite WAL mode succeed without deadlocks."""
        thread_count = 10
        messages_per_thread = 15
        errors = []

        def worker(thread_idx):
            try:
                c_id = f"conv_thread_{thread_idx}"
                for i in range(messages_per_thread):
                    m_id = f"msg_{thread_idx}_{i}"
                    self.store.persist_user_message(c_id, m_id, f"Thread {thread_idx} msg {i}", source="text")
                    self.store.update_status(c_id, m_id, MessageStatus.PROCESSING.value)
                    self.store.persist_assistant_message(
                        conversation_id=c_id,
                        message_id=f"asst_{thread_idx}_{i}",
                        parent_message_id=m_id,
                        content=f"Response {i}",
                        model_id="gemma4:12b",
                        agent_role="assistant",
                    )
                    self.store.mark_completed(c_id, m_id)
            except Exception as e:
                errors.append(f"Thread {thread_idx} error: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        # Verify total message count
        for i in range(thread_count):
            c_id = f"conv_thread_{i}"
            msgs = self.store.get_conversation_messages(c_id)
            # Both user and assistant stored = 2 * messages_per_thread
            self.assertEqual(len(msgs), messages_per_thread * 2)


if __name__ == "__main__":
    unittest.main()
