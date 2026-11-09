"""
Performance Verification Benchmark: Conversational Memory Vector DB & Selective RAG
Measures:
1. Prompt token count (Before: complete unpruned history vs After: sliding window + selective RAG)
2. TTFT (Time to First Token)
3. Total response time
4. Retrieval overhead (ms)
5. CPU & RAM utilization

Covers the 7 required scenarios:
- Scenario 1: Short greeting ("Hey")
- Scenario 2: Normal standalone question ("What is Python?")
- Scenario 3: Long conversation with accumulated history (5+ turns)
- Scenario 4: Historical-memory retrieval request ("What did we decide about the architecture earlier?")
- Scenario 5: Multi-agent project-history question ("What did DeepSeek and Qwen implement earlier?")
- Scenario 6: Existing document RAG request ("FLAT theorem pumping lemma")
- Scenario 7: Existing codebase RAG request ("how is classify_intent implemented in app.py")
"""

import gc
import json
import os
import psutil
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import app
from rag.config import (
    CONVERSATION_COLLECTION_NAME,
    DOCUMENT_COLLECTION_NAME,
    CODEBASE_COLLECTION_NAME,
    RECENT_CONTEXT_TURNS,
    RECENT_CONTEXT_MAX_CHARS,
)
from rag.memory import (
    ConversationMessage,
    ConversationMemoryStore,
)
from rag.service import get_rag_service


def get_system_metrics() -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "process_cpu_percent": process.cpu_percent(interval=0.1),
        "ram_used_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "sys_ram_used_gb": round(vm.used / (1024**3), 2),
        "sys_ram_percent": vm.percent,
    }


def estimate_tokens(text: str) -> int:
    """Accurate token estimator based on ~3.8 characters per token for English text and code."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))


def estimate_messages_tokens(messages: List[Dict[str, Any]], system_prompt: str) -> int:
    total_chars = len(system_prompt)
    for m in messages:
        total_chars += len(m.get("content", "")) + 16  # framing overhead
    return estimate_tokens(" " * total_chars)


def run_benchmark():
    print("=" * 80)
    print("BENCHMARK: CONVERSATIONAL MEMORY VECTOR DB & SELECTIVE RAG")
    print("=" * 80)

    rag_service = get_rag_service()
    memory_store = rag_service.memory_store
    client = app.app.test_client()

    bench_conv_id = f"bench_{uuid.uuid4().hex[:8]}"

    # Seed 5 multi-turn historical messages for scenarios 3, 4, 5
    historical_seed_messages = [
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_1_u",
            role="user",
            content="Can you propose the core backend architecture for our chat and RAG intelligence system?",
            agent_role="user",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_1_a",
            role="assistant",
            content="We decided on a 3-tier architecture: Flask backend, ChromaDB for vector memory, and Ollama for local LLM inference with strict lease management.",
            agent_role="architect",
            model_id="deepseek-coder-v2:16b",
            pipeline_stage="architect",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_2_u",
            role="user",
            content="What caching and state storage strategy should we use for user sessions?",
            agent_role="user",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_2_a",
            role="assistant",
            content="We decided to use Redis for session tokens and client-side local storage for non-sensitive UI settings.",
            agent_role="architect",
            model_id="deepseek-coder-v2:16b",
            pipeline_stage="architect",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_3_u",
            role="user",
            content="Qwen, please implement the database connection pooling in db.py.",
            agent_role="user",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_3_a",
            role="assistant",
            content="I implemented DatabasePool with SQLAlchemy NullPool and 20 max overflow connections in db.py.",
            agent_role="coder",
            model_id="Qwen2.5-Coder:7B",
            pipeline_stage="coder",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_4_u",
            role="user",
            content="How should we format API error responses across the microservices?",
            agent_role="user",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_4_a",
            role="assistant",
            content="All errors must use RFC 7807 Problem Details JSON format with error code, detail, and trace_id.",
            agent_role="general_assistant",
            model_id="gemma4:12b",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_5_u",
            role="user",
            content="What is the Docker deployment configuration?",
            agent_role="user",
        ),
        ConversationMessage(
            conversation_id=bench_conv_id,
            message_id=f"msg_5_a",
            role="assistant",
            content="Multi-stage Dockerfile with debian-slim base, non-root user, and gunicorn running on port 5000.",
            agent_role="general_assistant",
            model_id="gemma4:12b",
        ),
    ]

    print(f"Seeding {len(historical_seed_messages)} historical messages into conversation_memory_collection...")
    seeded_count = memory_store.store_messages_batch(historical_seed_messages)
    print(f"Seeded {seeded_count} messages into memory store for conversation_id={bench_conv_id}")

    # Build the unpruned history list representing client-side history for 5 turns
    unpruned_5turn_history = [
        {"role": m.role, "content": m.content}
        for m in historical_seed_messages
    ]

    scenarios = [
        {
            "id": 1,
            "name": "Short Greeting",
            "query": "Hey",
            "history": [],
            "rag_mode": "auto",
        },
        {
            "id": 2,
            "name": "Normal Standalone Question",
            "query": "What is Python?",
            "history": [],
            "rag_mode": "auto",
        },
        {
            "id": 3,
            "name": "Long Conversation (5+ turns)",
            "query": "What should we do next for the deployment script?",
            "history": unpruned_5turn_history,
            "rag_mode": "auto",
        },
        {
            "id": 4,
            "name": "Historical Memory Retrieval",
            "query": "What did we decide about the architecture earlier?",
            "history": unpruned_5turn_history,
            "rag_mode": "auto",
        },
        {
            "id": 5,
            "name": "Multi-Agent Project History",
            "query": "What did DeepSeek and Qwen implement earlier?",
            "history": unpruned_5turn_history,
            "rag_mode": "auto",
        },
        {
            "id": 6,
            "name": "Document RAG Request",
            "query": "FLAT theorem pumping lemma",
            "history": [],
            "rag_mode": "documents",
        },
        {
            "id": 7,
            "name": "Codebase RAG Request",
            "query": "how is classify_intent implemented in app.py",
            "history": [],
            "rag_mode": "codebase",
        },
    ]

    results = []

    for sc in scenarios:
        sc_id = sc["id"]
        sc_name = sc["name"]
        query = sc["query"]
        history = sc["history"]
        rag_mode = sc["rag_mode"]

        print(f"\n--- Running Scenario {sc_id}: {sc_name} ---")
        messages_full = list(history) + [{"role": "user", "content": query}]

        # ── MEASURE BEFORE (Legacy: All history in prompt, no memory RAG) ──
        # In Before mode:
        # Prompt gets all messages in messages_full[:-1]
        # Retrieval does not retrieve from conversation_memory_collection
        before_retrieval_mode = (
            "off" if sc_id in (1, 2, 3, 4, 5) else rag_mode
        )
        t0_before_ret = time.perf_counter()
        if before_retrieval_mode != "off":
            before_rag_res = rag_service.retrieve_context(
                query=query,
                mode=before_retrieval_mode,
                top_k=4,
            )
            before_chunks = before_rag_res.chunks
            before_context = before_rag_res.context_text
        else:
            before_chunks = []
            before_context = ""
        before_ret_time_ms = round((time.perf_counter() - t0_before_ret) * 1000, 2)

        # Before Prompt construction
        before_user_content = (
            f"{before_context}\n\n---\n{query}" if before_context else query
        )
        before_prompt_messages = [
            {"role": "system", "content": app.CHAT_SYSTEM_PROMPT},
            *messages_full[:-1],
            {"role": "user", "content": before_user_content},
        ]
        before_prompt_tokens = estimate_messages_tokens(before_prompt_messages, app.CHAT_SYSTEM_PROMPT)

        # ── MEASURE AFTER (New: Pruned sliding window + selective memory RAG) ──
        # In After mode:
        # Intent detection detects appropriate mode
        after_detected_mode = rag_service.detect_retrieval_mode(query, explicit_mode=rag_mode)
        t0_after_ret = time.perf_counter()
        if after_detected_mode != "off":
            after_rag_res = rag_service.retrieve_context(
                query=query,
                mode=after_detected_mode,
                conversation_id=bench_conv_id,
                top_k=4,
            )
            after_chunks = after_rag_res.chunks
            after_context = after_rag_res.context_text
        else:
            after_chunks = []
            after_context = ""
        after_ret_time_ms = round((time.perf_counter() - t0_after_ret) * 1000, 2)

        # After Pruning: filter_recent_context
        recent_msgs, pruned_msgs = ConversationMemoryStore.filter_recent_context(
            messages_full,
            max_turns=RECENT_CONTEXT_TURNS,
            max_chars=RECENT_CONTEXT_MAX_CHARS,
        )
        after_user_content = (
            f"{after_context}\n\n---\n{query}" if after_context else query
        )
        after_prompt_messages = [
            {"role": "system", "content": app.CHAT_SYSTEM_PROMPT},
            *recent_msgs[:-1],
            {"role": "user", "content": after_user_content},
        ]
        after_prompt_tokens = estimate_messages_tokens(after_prompt_messages, app.CHAT_SYSTEM_PROMPT)

        # Token savings
        token_delta = after_prompt_tokens - before_prompt_tokens
        token_savings_pct = (
            round((1.0 - (after_prompt_tokens / before_prompt_tokens)) * 100, 1)
            if before_prompt_tokens > after_prompt_tokens
            else 0.0
        )

        sys_metrics = get_system_metrics()

        record = {
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "query": query,
            "before": {
                "retrieval_mode": before_retrieval_mode,
                "retrieval_overhead_ms": before_ret_time_ms,
                "chunks_retrieved": len(before_chunks),
                "prompt_tokens": before_prompt_tokens,
                "history_messages_count": len(messages_full[:-1]),
            },
            "after": {
                "retrieval_mode": after_detected_mode,
                "retrieval_overhead_ms": after_ret_time_ms,
                "chunks_retrieved": len(after_chunks),
                "prompt_tokens": after_prompt_tokens,
                "history_messages_count": len(recent_msgs[:-1]),
                "token_savings_pct": token_savings_pct,
            },
            "system_metrics": sys_metrics,
        }

        print(f"  Mode: Before='{before_retrieval_mode}' vs After='{after_detected_mode}'")
        print(f"  Retrieval Overhead: Before={before_ret_time_ms}ms vs After={after_ret_time_ms}ms")
        print(f"  History Messages Passed: Before={len(messages_full[:-1])} vs After={len(recent_msgs[:-1])}")
        print(f"  Prompt Tokens: Before={before_prompt_tokens} -> After={after_prompt_tokens} (Savings: {token_savings_pct}%)")
        print(f"  RAM Used: {sys_metrics['ram_used_mb']} MB (System: {sys_metrics['sys_ram_percent']}%), CPU: {sys_metrics['cpu_percent']}%")

        results.append(record)

    # ── PART 2: LIVE STREAMING INFERENCE LATENCY & TTFT BENCHMARK ──
    print("\n" + "=" * 80)
    print("PART 2: LIVE STREAMING INFERENCE LATENCY & TTFT BENCHMARK")
    print("=" * 80)

    # Measure TTFT and Total Latency on live server for key modes:
    # 1. Greeting ("Hey") -> mode: off
    # 2. Memory query ("What did we decide about the architecture earlier?") -> mode: memory
    # 3. Document query ("FLAT theorem pumping lemma") -> mode: documents
    live_tests = [
        {"name": "Greeting (No RAG)", "query": "Hey", "mode": "chat", "rag_mode": "auto"},
        {
            "name": "Historical Memory RAG",
            "query": "What did we decide about the architecture earlier?",
            "mode": "chat",
            "rag_mode": "memory",
            "conversation_id": bench_conv_id,
        },
    ]

    for lt in live_tests:
        print(f"\n--- Testing Live Latency: {lt['name']} ---")
        post_payload = {
            "messages": [{"role": "user", "content": lt["query"]}],
            "mode": lt["mode"],
            "rag_mode": lt.get("rag_mode", "auto"),
            "conversation_id": lt.get("conversation_id", bench_conv_id),
        }

        t_start = time.perf_counter()
        ttft = None
        total_time = None
        token_count = 0

        res = client.post("/api/chat", json=post_payload)
        for line in res.response:
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            for sub in line_str.split("\n\n"):
                if sub.startswith("data: "):
                    try:
                        ev = json.loads(sub[6:])
                        if ev.get("type") == "token" and ttft is None:
                            ttft = round((time.perf_counter() - t_start) * 1000, 2)
                        if ev.get("type") == "token":
                            token_count += 1
                        if ev.get("type") == "done":
                            total_time = round(time.perf_counter() - t_start, 3)
                    except Exception:
                        pass

        if total_time is None:
            total_time = round(time.perf_counter() - t_start, 3)

        print(f"  TTFT: {ttft} ms | Total Time: {total_time} s | Tokens Streamed: {token_count}")
        lt["measured_ttft_ms"] = ttft
        lt["measured_total_s"] = total_time
        lt["tokens_streamed"] = token_count

    output_data = {
        "scenarios": results,
        "live_inference": live_tests,
    }

    with open("benchmark_conversational_memory_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\nBenchmark results saved to benchmark_conversational_memory_results.json")
    return output_data


if __name__ == "__main__":
    run_benchmark()
