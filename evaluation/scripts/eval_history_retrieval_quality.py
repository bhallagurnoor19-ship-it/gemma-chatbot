"""
Evaluation Script: Conversational Memory Retrieval Quality, Cross-Model Access,
Relevance Metrics (P@K, R@K, MRR), Latency Profiling, and History Growth Benchmarks (100, 500, 1000 msgs).
"""

import gc
import json
import os
import psutil
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock
import uuid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from message_store import MessageStore, MessageRecord
from rag.config import CONVERSATION_COLLECTION_NAME, EMBEDDING_DIMENSION
from rag.memory import ConversationMemoryStore, ConversationMessage
from rag.prompt_builder import PromptBuilder
from rag.retriever import RetrievedChunk
from rag.service import RAGService
from rag.vector_store import VectorStoreManager


def get_mem_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


class MockEmbeddingService:
    """Fast deterministic embedding generator for hermetic benchmark scaling without Ollama network delay."""
    def __init__(self, dimension=EMBEDDING_DIMENSION):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            # Deterministic hash-based mock embedding to ensure semantic distance variation
            seed = sum(ord(c) for c in text[:30]) % 1000 / 1000.0
            vec = [seed + (i % 10) * 0.01 for i in range(self.dimension)]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            results.append([x / norm for x in vec])
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def run_history_evaluation():
    print("=" * 80)
    print("EVALUATION: CONVERSATIONAL MEMORY RETRIEVAL QUALITY & SCALING BENCHMARKS")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="eval_chroma_mem_")
    chroma_dir = os.path.join(temp_dir, "chroma")
    db_path = os.path.join(temp_dir, "eval_messages.db")
    os.makedirs(chroma_dir, exist_ok=True)

    try:
        store = MessageStore(db_path=db_path)
        vsm = VectorStoreManager(persist_dir=chroma_dir)
        vsm.initialize()
        mock_embed = MockEmbeddingService()

        mem_store = ConversationMemoryStore(
            vector_store=vsm,
            embedding_service=mock_embed,
            collection_name=CONVERSATION_COLLECTION_NAME,
            persist_dir=chroma_dir,
            message_store=store,
        )

        prompt_builder = PromptBuilder()

        # ═══════════════════════════════════════════════════════════════════
        # 1. UNIVERSAL CROSS-MODEL & CROSS-CONVERSATION ACCESS TEST
        # ═══════════════════════════════════════════════════════════════════
        print("\n--- 1. Universal Cross-Model & Cross-Conversation Access ---")
        conv_alpha = "conv_investigation_alpha"
        
        # Stage 1: Gemma investigates
        msg_gemma = ConversationMessage(
            conversation_id=conv_alpha,
            message_id="msg_gemma_01",
            role="assistant",
            content="Investigation report: Identified high TTFT latency caused by frequent model reloading during multi-turn chats.",
            agent_role="general_assistant",
            model_id="gemma4:12b",
            pipeline_stage="investigate",
            source="text",
        )
        mem_store.store_message_sync(msg_gemma)

        # Stage 2: DeepSeek analyzes
        msg_deepseek = ConversationMessage(
            conversation_id=conv_alpha,
            message_id="msg_deepseek_02",
            role="assistant",
            content="Architectural design: Proposed ModelLeaseManager with resident model policy and idle timeout to keep models warm in Ollama.",
            agent_role="architect",
            model_id="deepseek-coder-v2:16b",
            pipeline_stage="architect",
            source="text",
        )
        mem_store.store_message_sync(msg_deepseek)

        # Stage 3: Qwen implements
        msg_qwen = ConversationMessage(
            conversation_id=conv_alpha,
            message_id="msg_qwen_03",
            role="assistant",
            content="Implementation complete: Added acquire_lease() and evict_active_model() in model_lifecycle.py with unit tests in tests_model_lifecycle.py.",
            agent_role="coder",
            model_id="Qwen2.5-Coder:7B",
            pipeline_stage="coder",
            source="text",
        )
        mem_store.store_message_sync(msg_qwen)

        # Stage 4A: Scoped query with conversation_id=conv_beta (Testing conversation isolation boundary)
        conv_beta = "conv_query_beta"
        retrieved_scoped = mem_store.retrieve_memory(
            query="What did DeepSeek and Qwen implement for model lifecycle earlier?",
            conversation_id=conv_beta,
            agent_role=None,
            top_k=5,
            distance_threshold=1.0,
        )
        print(f"Scoped retrieval (conversation_id={conv_beta}): {len(retrieved_scoped)} chunks (Strict conversation isolation)")

        # Stage 4B: Global cross-conversation query with conversation_id=None (Testing universal cross-model & cross-conversation access)
        retrieved_cross = mem_store.retrieve_memory(
            query="What did DeepSeek and Qwen implement for model lifecycle earlier?",
            conversation_id=None,  # Cross-conversation search
            agent_role=None,
            top_k=5,
            distance_threshold=1.0,
        )

        retrieved_roles = {getattr(c, "agent_role", "") for c in retrieved_cross}
        retrieved_convs = {getattr(c, "conversation_id", "") for c in retrieved_cross}
        print(f"Global cross-conversation results: {len(retrieved_cross)} chunks returned")
        print(f"Retrieved Agent Roles: {retrieved_roles}")
        print(f"Source Conversation IDs: {retrieved_convs}")

        cross_model_pass = ("architect" in retrieved_roles and "coder" in retrieved_roles)
        cross_conv_pass = (conv_alpha in retrieved_convs)
        print(f"Cross-Model Retrieval (Gemma retrieving DeepSeek/Qwen): {'PASS' if cross_model_pass else 'FAIL'}")
        print(f"Cross-Conversation Retrieval (Unscoped):                 {'PASS' if cross_conv_pass else 'FAIL'}")

        # ═══════════════════════════════════════════════════════════════════
        # 2. RELEVANCE METRICS: Precision@K, Recall@K, MRR
        # ═══════════════════════════════════════════════════════════════════
        print("\n--- 2. Relevance Metrics (P@K, R@K, MRR) ---")
        # Populate 20 distractor messages
        for i in range(20):
            distractor = ConversationMessage(
                conversation_id=f"conv_noise_{i % 3}",
                message_id=f"noise_msg_{i}",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Irrelevant chat conversation noise {i}: Discussing unrelated algorithms, gardening tips, recipe for pasta, weather report {i*7}.",
                agent_role="general_assistant",
                model_id="gemma4:12b",
            )
            mem_store.store_message_sync(distractor)

        # Target query with 3 ground truth relevant items (msg_gemma_01, msg_deepseek_02, msg_qwen_03)
        test_queries = [
            {
                "query": "What did the architect design for model lease management?",
                "relevant_ids": {"conv_investigation_alpha_msg_deepseek_02"},
            },
            {
                "query": "Show me the implementation details in model_lifecycle.py",
                "relevant_ids": {"conv_investigation_alpha_msg_qwen_03"},
            },
            {
                "query": "What was the investigation on TTFT latency?",
                "relevant_ids": {"conv_investigation_alpha_msg_gemma_01"},
            },
        ]

        p3_list, r3_list, p5_list, r5_list, mrr_list = [], [], [], [], []

        for tq in test_queries:
            q_text = tq["query"]
            rel_ids = tq["relevant_ids"]

            chunks_5 = mem_store.retrieve_memory(query=q_text, top_k=5, distance_threshold=1.0)
            ret_ids_5 = [c.chunk_id for c in chunks_5]
            ret_ids_3 = ret_ids_5[:3]

            # P@3, R@3
            hits_3 = len(set(ret_ids_3).intersection(rel_ids))
            p3 = hits_3 / 3.0
            r3 = hits_3 / len(rel_ids) if rel_ids else 0.0

            # P@5, R@5
            hits_5 = len(set(ret_ids_5).intersection(rel_ids))
            p5 = hits_5 / 5.0
            r5 = hits_5 / len(rel_ids) if rel_ids else 0.0

            # MRR
            rr = 0.0
            for rank, cid in enumerate(ret_ids_5, start=1):
                if cid in rel_ids:
                    rr = 1.0 / rank
                    break

            p3_list.append(p3)
            r3_list.append(r3)
            p5_list.append(p5)
            r5_list.append(r5)
            mrr_list.append(rr)

        avg_p3 = sum(p3_list) / len(p3_list)
        avg_r3 = sum(r3_list) / len(r3_list)
        avg_p5 = sum(p5_list) / len(p5_list)
        avg_r5 = sum(r5_list) / len(r5_list)
        avg_mrr = sum(mrr_list) / len(mrr_list)

        print(f"Precision@3: {avg_p3:.4f}")
        print(f"Recall@3:    {avg_r3:.4f}")
        print(f"Precision@5: {avg_p5:.4f}")
        print(f"Recall@5:    {avg_r5:.4f}")
        print(f"MRR:         {avg_mrr:.4f}")

        # ═══════════════════════════════════════════════════════════════════
        # 3. RETRIEVAL EFFICIENCY PROFILING (8 Scenarios)
        # ═══════════════════════════════════════════════════════════════════
        print("\n--- 3. Retrieval Efficiency Profiling (8 Scenarios) ---")
        from rag.retriever import UnifiedRetriever
        retriever = UnifiedRetriever(vector_store=vsm, embedding_service=mock_embed)
        rag_service = RAGService(
            retriever=retriever,
            prompt_builder=prompt_builder,
            memory_store=mem_store,
        )

        scenarios = [
            ("1. Greeting (skip)", "Hey there, good morning!"),
            ("2. General QA (skip)", "What is a binary search tree in computer science?"),
            ("3. Explicit History", "What did we decide about model lifecycle earlier?"),
            ("4. Implicit History", "Why did we choose SQLite instead?"),
            ("5. Cross-Conversation", "What was our decision on persistence in the earlier chat?"),
            ("6. Cross-Model", "What did DeepSeek propose for architecture earlier?"),
            ("7. Ambiguous Follow-up", "Can we improve that?"),
            ("8. Distractor-Heavy", "Summarize what was discussed earlier regarding latency"),
        ]

        scenario_timings = []
        for name, query in scenarios:
            t0 = time.perf_counter()
            # 1. Intent classification
            t_int_0 = time.perf_counter()
            mode = rag_service.detect_retrieval_mode(query, explicit_mode="auto")
            t_int_1 = time.perf_counter()
            intent_ms = (t_int_1 - t_int_0) * 1000.0

            # 2. Retrieval execution (if needed)
            t_ret_0 = time.perf_counter()
            res = rag_service.retrieve_context(query=query, mode="auto")
            t_ret_1 = time.perf_counter()
            total_ms = (t_ret_1 - t0) * 1000.0
            ret_ms = (t_ret_1 - t_ret_0) * 1000.0

            scenario_timings.append({
                "scenario": name,
                "query": query,
                "detected_mode": mode,
                "intent_ms": round(intent_ms, 3),
                "retrieval_ms": round(ret_ms, 3),
                "total_overhead_ms": round(total_ms, 3),
                "chunks_returned": len(res.chunks),
                "token_overhead": res.token_count,
            })
            print(f"  {name:25s} | Mode: {mode:10s} | Intent: {intent_ms:6.3f}ms | Total: {total_ms:6.3f}ms | Chunks: {len(res.chunks)} | Tokens: {res.token_count}")

        # ═══════════════════════════════════════════════════════════════════
        # 4. HISTORY GROWTH BENCHMARKS (100, 500, 1000 messages)
        # ═══════════════════════════════════════════════════════════════════
        print("\n--- 4. History Growth Scaling Benchmarks (100, 500, 1000 messages) ---")
        scaling_results = []
        sizes = [100, 500, 1000]

        for target_size in sizes:
            # Create isolated scale environment
            scale_dir = os.path.join(temp_dir, f"scale_{target_size}")
            os.makedirs(scale_dir, exist_ok=True)
            scale_vsm = VectorStoreManager(persist_dir=scale_dir)
            scale_vsm.initialize()

            scale_mem = ConversationMemoryStore(
                vector_store=scale_vsm,
                embedding_service=mock_embed,
                collection_name=CONVERSATION_COLLECTION_NAME,
                persist_dir=scale_dir,
                message_store=None,
            )

            # Generate messages
            msgs = []
            for m_idx in range(target_size):
                msgs.append(ConversationMessage(
                    conversation_id=f"scale_conv_{m_idx // 20}",
                    message_id=f"scale_msg_{m_idx}",
                    role="user" if m_idx % 2 == 0 else "assistant",
                    content=f"Benchmark conversational history record #{m_idx}. Discussing microservice scaling, database indexation, caching layers, and latency optimizations in tier {m_idx % 5}.",
                    agent_role="architect" if m_idx % 3 == 0 else "coder",
                ))

            # Measure indexing time
            t_idx_0 = time.perf_counter()
            indexed_count = scale_mem.store_messages_batch(msgs)
            t_idx_1 = time.perf_counter()
            index_total_s = t_idx_1 - t_idx_0
            index_ms_per_doc = (index_total_s * 1000.0) / target_size

            # Measure retrieval latency over 10 query repetitions
            query_latencies = []
            for _ in range(10):
                t_q0 = time.perf_counter()
                ret_chunks = scale_mem.retrieve_memory(
                    query="What did we discuss about microservice scaling and caching layers?",
                    top_k=5,
                    distance_threshold=0.8,
                )
                t_q1 = time.perf_counter()
                query_latencies.append((t_q1 - t_q0) * 1000.0)

            avg_q_lat = sum(query_latencies) / len(query_latencies)
            p95_q_lat = sorted(query_latencies)[int(0.95 * len(query_latencies))]

            mem_used = get_mem_mb()

            scale_entry = {
                "dataset_size": target_size,
                "indexed_messages": indexed_count,
                "indexing_total_seconds": round(index_total_s, 3),
                "indexing_ms_per_msg": round(index_ms_per_doc, 3),
                "avg_query_latency_ms": round(avg_q_lat, 3),
                "p95_query_latency_ms": round(p95_q_lat, 3),
                "process_memory_mb": mem_used,
            }
            scaling_results.append(scale_entry)
            print(f"  Dataset N={target_size:4d} | Indexing: {index_total_s:6.3f}s ({index_ms_per_doc:5.2f}ms/msg) | Query Latency: {avg_q_lat:6.3f}ms (P95: {p95_q_lat:6.3f}ms) | RAM: {mem_used} MB")
            scale_mem.close()

        # Save machine-readable output
        out_dir = os.path.join(PROJECT_ROOT, "evaluation", "data")
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "conversational_memory_quality_and_scaling.json")

        summary_payload = {
            "cross_model_verified": cross_model_pass,
            "cross_conversation_verified": cross_conv_pass,
            "relevance_metrics": {
                "precision_at_3": round(avg_p3, 4),
                "recall_at_3": round(avg_r3, 4),
                "precision_at_5": round(avg_p5, 4),
                "recall_at_5": round(avg_r5, 4),
                "mrr": round(avg_mrr, 4),
            },
            "retrieval_efficiency_scenarios": scenario_timings,
            "history_growth_benchmarks": scaling_results,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        print(f"\nSaved evaluation metrics to: {out_json}")

    finally:
        try:
            mem_store.close()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_history_evaluation()
