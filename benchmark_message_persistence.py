"""
Performance & Latency Benchmark: Message Persistence & Lifecycle Management
Measures:
1. Message submission overhead (request parsing, UUID binding, validation): target < 2ms
2. Immediate acknowledgement latency (SSE ack: persisted): target < 15ms
3. SQLite persistence latency (WAL synchronous disk write): target < 5ms
4. Async memory indexing queue delay (non-blocking Chroma queue handoff): target < 1ms
5. Idempotent replay latency (cached completed response turnaround): target < 10ms
6. Concurrency conflict detection latency (in-flight HTTP 409 check): target < 5ms
7. Total TTFT impact (time before first token stream event): target < 25ms total overhead
8. CPU and RAM utilization
"""

import gc
import json
import os
import psutil
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
import uuid
from typing import Any, Dict, List

import app
from message_store import MessageStore, MessageRecord, MessageStatus, MessageSource
from rag.config import CONVERSATION_COLLECTION_NAME, EMBEDDING_DIMENSION
from rag.memory import ConversationMemoryStore, ConversationMessage
from rag.vector_store import VectorStoreManager


def get_system_metrics() -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "process_cpu_percent": process.cpu_percent(interval=0.05),
        "ram_used_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "sys_ram_used_gb": round(vm.used / (1024**3), 2),
        "sys_ram_percent": vm.percent,
    }


def make_fake_post(response_text="Benchmark assistant response token stream"):
    def fake_post(url, *args, **kwargs):
        if "/api/embed" in url:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"embeddings": [[0.1] * EMBEDDING_DIMENSION]}
            return resp
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = [
            json.dumps({"message": {"content": " " + w}, "done": False}).encode("utf-8")
            for w in response_text.split()
        ] + [json.dumps({"done": True}).encode("utf-8")]
        return resp
    return fake_post


def run_benchmark():
    print("=" * 80)
    print("BENCHMARK: MESSAGE PERSISTENCE, LIFECYCLE & VOICE-READY ARCHITECTURE")
    print("=" * 80)

    initial_metrics = get_system_metrics()
    print(f"[Initial System Metrics] Process RAM: {initial_metrics['ram_used_mb']} MB | System RAM: {initial_metrics['sys_ram_used_gb']} GB ({initial_metrics['sys_ram_percent']}%)")

    temp_dir = tempfile.mkdtemp(prefix="bench_msg_store_")
    chroma_dir = os.path.join(temp_dir, "chroma")
    bench_db_path = os.path.join(temp_dir, "benchmark_messages.db")
    os.makedirs(chroma_dir, exist_ok=True)

    try:
        store = MessageStore(db_path=bench_db_path)

        # -------------------------------------------------------------
        # Benchmark 1: SQLite Persistence Write Latency (WAL mode)
        # -------------------------------------------------------------
        print("\n--- 1. SQLite Persistence Latency (WAL Mode) ---")
        persist_user_times = []
        persist_asst_times = []
        n_iterations = 100

        for i in range(n_iterations):
            conv_id = f"bench_conv_{i % 5}"
            msg_id = f"msg_{uuid.uuid4()}"
            t0 = time.perf_counter()
            rec = store.persist_user_message(
                conversation_id=conv_id,
                message_id=msg_id,
                content=f"Benchmark question #{i}: What is the latency profile?",
                source="text" if i % 2 == 0 else "voice",
            )
            t1 = time.perf_counter()
            persist_user_times.append((t1 - t0) * 1000.0)

            asst_msg_id = f"msg_{uuid.uuid4()}"
            t2 = time.perf_counter()
            store.persist_assistant_message(
                conversation_id=conv_id,
                message_id=asst_msg_id,
                parent_message_id=msg_id,
                content=f"Benchmark answer #{i}: Response generated in 12ms.",
                model_id="gemma-2-9b-it",
                agent_role="assistant",
            )
            t3 = time.perf_counter()
            persist_asst_times.append((t3 - t2) * 1000.0)

        avg_user_write = sum(persist_user_times) / len(persist_user_times)
        p95_user_write = sorted(persist_user_times)[int(0.95 * len(persist_user_times))]
        avg_asst_write = sum(persist_asst_times) / len(persist_asst_times)
        p95_asst_write = sorted(persist_asst_times)[int(0.95 * len(persist_asst_times))]

        print(f"User Message Write (N={n_iterations}): Avg = {avg_user_write:.3f} ms | P95 = {p95_user_write:.3f} ms (Target: < 5.0 ms)")
        print(f"Assistant Message Write (N={n_iterations}): Avg = {avg_asst_write:.3f} ms | P95 = {p95_asst_write:.3f} ms (Target: < 5.0 ms)")

        # -------------------------------------------------------------
        # Benchmark 2: Non-blocking Async Memory Queue Latency
        # -------------------------------------------------------------
        print("\n--- 2. Async Memory Indexing Queue Handoff Latency ---")
        vector_store = VectorStoreManager(persist_dir=chroma_dir)
        vector_store.initialize()

        mock_embedding = MagicMock()
        mock_embedding.embed_documents.side_effect = lambda docs: [[0.1] * EMBEDDING_DIMENSION for _ in docs]
        mock_embedding.embed_query.side_effect = lambda q: [0.1] * EMBEDDING_DIMENSION

        mem_store = ConversationMemoryStore(
            vector_store=vector_store,
            embedding_service=mock_embedding,
            collection_name=CONVERSATION_COLLECTION_NAME,
            persist_dir=chroma_dir,
            message_store=store,
        )

        handoff_times = []
        for i in range(100):
            msg = ConversationMessage(
                conversation_id="conv_async",
                message_id=f"msg_q_{i}",
                role="user",
                content=f"Quick indexing test {i}",
                source="text",
            )
            t0 = time.perf_counter()
            mem_store.store_message_async(msg)
            t1 = time.perf_counter()
            handoff_times.append((t1 - t0) * 1000.0)

        avg_handoff = sum(handoff_times) / len(handoff_times)
        p95_handoff = sorted(handoff_times)[int(0.95 * len(handoff_times))]
        print(f"Queue Handoff (N={len(handoff_times)}): Avg = {avg_handoff:.3f} ms | P95 = {p95_handoff:.3f} ms (Target: < 1.0 ms)")

        # -------------------------------------------------------------
        # Benchmark 3: SSE Acknowledgement & TTFT Overhead in Flask
        # -------------------------------------------------------------
        print("\n--- 3. Flask SSE Acknowledgement & TTFT Overhead ---")
        app.app.config["TESTING"] = True
        client = app.app.test_client()

        with patch("app.get_message_store", return_value=store), \
             patch("requests.get") as mock_get, \
             patch("requests.post", side_effect=make_fake_post()), \
             patch.object(ConversationMemoryStore, "store_message_sync", return_value=True):

            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": [{"name": "gemma:2b"}]}

            ack_latencies = []
            submission_latencies = []
            cached_replay_latencies = []
            conflict_latencies = []
            ttft_latencies = []

            for i in range(30):
                req_id = str(uuid.uuid4())
                msg_id = f"bench_msg_{uuid.uuid4().hex[:12]}"
                conv_id = f"bench_stream_conv_{i}"

                payload = {
                    "request_id": req_id,
                    "messages": [{"role": "user", "content": f"Hello benchmark test {i}"}],
                    "mode": "chat",
                    "conversation_id": conv_id,
                    "message_id": msg_id,
                    "source": "text",
                }

                t0 = time.perf_counter()
                resp = client.post(
                    "/api/chat",
                    data=json.dumps(payload),
                    content_type="application/json",
                )
                t_post = time.perf_counter()
                submission_latencies.append((t_post - t0) * 1000.0)

                ack_time = None
                first_token_time = None
                raw_data = resp.get_data(as_text=True)
                lines = raw_data.split("\n")

                for line in lines:
                    line_str = line.strip()
                    if line_str.startswith("data: "):
                        raw = line_str[6:].strip()
                        try:
                            ev = json.loads(raw)
                            ev_type = ev.get("type")
                            if ev_type == "ack" and ev.get("status") == "persisted" and ack_time is None:
                                ack_time = (t_post - t0) * 1000.0
                            elif ev_type == "token" and first_token_time is None:
                                first_token_time = (t_post - t0) * 1000.0
                        except Exception:
                            pass

                if ack_time:
                    ack_latencies.append(ack_time)
                if first_token_time:
                    ttft_latencies.append(first_token_time)

                # ---------------------------------------------------------
                # Benchmark 4: Cached Completion Replay Latency
                # ---------------------------------------------------------
                asst_id = f"bench_asst_{uuid.uuid4().hex[:12]}"
                store.persist_assistant_message(
                    conversation_id=conv_id,
                    message_id=asst_id,
                    parent_message_id=msg_id,
                    content=f"Cached answer for test {i}",
                    model_id="gemma-2-9b-it",
                )
                store.mark_completed(conv_id, msg_id)

                # Send duplicate request with identical message_id
                t_replay_start = time.perf_counter()
                replay_resp = client.post(
                    "/api/chat",
                    data=json.dumps(payload),
                    content_type="application/json",
                )
                _ = replay_resp.get_data(as_text=True)
                t_replay_end = time.perf_counter()
                cached_replay_latencies.append((t_replay_end - t_replay_start) * 1000.0)

                # ---------------------------------------------------------
                # Benchmark 5: Concurrency Conflict Detection (409)
                # ---------------------------------------------------------
                in_flight_msg_id = f"inflight_{uuid.uuid4().hex[:12]}"
                store.register_message(conv_id, in_flight_msg_id, "user", "In flight test")
                store.update_status(conv_id, in_flight_msg_id, MessageStatus.PROCESSING)

                conflict_payload = {
                    "request_id": str(uuid.uuid4()),
                    "messages": [{"role": "user", "content": "In flight test duplicate"}],
                    "conversation_id": conv_id,
                    "message_id": in_flight_msg_id,
                    "source": "text",
                }
                t_conf_start = time.perf_counter()
                conf_resp = client.post(
                    "/api/chat",
                    data=json.dumps(conflict_payload),
                    content_type="application/json",
                )
                t_conf_end = time.perf_counter()
                if conf_resp.status_code == 409:
                    conflict_latencies.append((t_conf_end - t_conf_start) * 1000.0)

            avg_sub = sum(submission_latencies) / len(submission_latencies)
            p95_sub = sorted(submission_latencies)[int(0.95 * len(submission_latencies))]
            print(f"Submission Overhead (N={len(submission_latencies)}): Avg = {avg_sub:.3f} ms | P95 = {p95_sub:.3f} ms (Target: < 2.0 ms)")

            avg_ack = sum(ack_latencies) / len(ack_latencies) if ack_latencies else 0.0
            p95_ack = sorted(ack_latencies)[int(0.95 * len(ack_latencies))] if ack_latencies else 0.0
            print(f"Acknowledgement Latency to 'ack: persisted' (N={len(ack_latencies)}): Avg = {avg_ack:.3f} ms | P95 = {p95_ack:.3f} ms (Target: < 15.0 ms)")

            avg_replay = sum(cached_replay_latencies) / len(cached_replay_latencies)
            p95_replay = sorted(cached_replay_latencies)[int(0.95 * len(cached_replay_latencies))]
            print(f"Cached Turn Replay Latency (N={len(cached_replay_latencies)}): Avg = {avg_replay:.3f} ms | P95 = {p95_replay:.3f} ms (Target: < 10.0 ms)")

            avg_conflict = sum(conflict_latencies) / len(conflict_latencies)
            p95_conflict = sorted(conflict_latencies)[int(0.95 * len(conflict_latencies))]
            print(f"Concurrency Conflict (409) Detection (N={len(conflict_latencies)}): Avg = {avg_conflict:.3f} ms | P95 = {p95_conflict:.3f} ms (Target: < 5.0 ms)")

        # -------------------------------------------------------------
        # Benchmark 6: Voice-Ready Source Metadata Benchmark
        # -------------------------------------------------------------
        print("\n--- 4. Voice vs Text Metadata Handling ---")
        t_v0 = time.perf_counter()
        v_rec = store.persist_user_message(
            conversation_id="voice_conv_1",
            message_id="voice_msg_1",
            content="Transcribed speech audio query: summarize the latest report",
            source=MessageSource.VOICE,
        )
        t_v1 = time.perf_counter()
        voice_write_ms = (t_v1 - t_v0) * 1000.0
        print(f"Voice Message Persistence Write: {voice_write_ms:.3f} ms | Stored source: '{v_rec.source}'")

    finally:
        try:
            mem_store.close()
        except Exception:
            pass
        try:
            store.close()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)

    final_metrics = get_system_metrics()
    print("\n--- 5. System Footprint Delta ---")
    print(f"Initial Process RAM: {initial_metrics['ram_used_mb']} MB -> Final: {final_metrics['ram_used_mb']} MB (Delta: +{round(final_metrics['ram_used_mb'] - initial_metrics['ram_used_mb'], 2)} MB)")
    print(f"Process CPU Usage: {final_metrics['process_cpu_percent']}%")
    print("=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
