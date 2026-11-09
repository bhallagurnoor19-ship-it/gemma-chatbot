"""
Phase 7.2 Stage 1 - Answer Quality & Grounding Evaluation Runner
- Loads rag_evaluation/answer_quality_dataset.json (40 verified queries)
- Executes each query through the live Flask chatbot pipeline (/api/chat)
- Captures:
  1. Chosen model route (Gemma 4, DeepSeek Architect, Qwen Coder, Gemini API)
  2. RAG retrieval status and source chunks
  3. Time-To-First-Token (TTFT) and total response latency
  4. Complete generated answer tokens
  5. Citations emitted by the pipeline
- Saves raw generation results to rag_evaluation/raw_generation_results.json
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional
import requests

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.service import get_rag_service

DATASET_PATH = os.path.join(os.path.dirname(__file__), "answer_quality_dataset.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "raw_generation_results.json")
DEFAULT_ENDPOINT = "http://127.0.0.1:5000/api/chat"


def execute_chat_query(
    question: str,
    rag_mode: str = "auto",
    model_mode: str = "auto",
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Execute a chat query against the production SSE streaming endpoint.
    Extracts tokens, routing decisions, RAG sources, and latency measurements.
    """
    payload = {
        "messages": [{"role": "user", "content": question}],
        "mode": model_mode,
        "rag_mode": rag_mode,
    }

    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    accumulated_tokens: List[str] = []
    route_info: Dict[str, Any] = {}
    done_info: Dict[str, Any] = {}
    stage_events: List[Dict[str, Any]] = []

    try:
        with requests.post(endpoint, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                raw_json = line[6:].strip()
                if not raw_json:
                    continue

                try:
                    event = json.loads(raw_json)
                except Exception:
                    continue

                event_type = event.get("type")

                if event_type == "route":
                    route_info = event

                elif event_type == "stage_start":
                    stage_events.append(event)

                elif event_type == "token":
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    token_str = event.get("token", "")
                    accumulated_tokens.append(token_str)

                elif event_type == "stage_end":
                    stage_events.append(event)

                elif event_type == "done":
                    done_info = event
                    break

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "generated_answer": "".join(accumulated_tokens),
            "total_latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            "ttft_ms": None,
            "route_info": route_info,
        }

    t_end = time.perf_counter()
    total_latency_ms = round((t_end - t_start) * 1000, 2)
    ttft_ms = (
        round((t_first_token - t_start) * 1000, 2)
        if t_first_token is not None
        else None
    )

    full_answer = "".join(accumulated_tokens).strip()

    return {
        "success": True,
        "error": None,
        "generated_answer": full_answer,
        "total_latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "route_info": route_info,
        "stage_events": stage_events,
        "done_info": done_info,
    }


def run_evaluation(
    dataset_path: str = DATASET_PATH,
    output_path: str = OUTPUT_PATH,
    endpoint: str = DEFAULT_ENDPOINT,
    limit: Optional[int] = None,
    category_filter: Optional[str] = None,
    mode_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load dataset, execute each query, capture context and output, and save raw results."""
    print("=" * 70)
    print("PHASE 7.2 STAGE 1 — ANSWER QUALITY EVALUATION RUNNER")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at '{dataset_path}'")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if category_filter:
        dataset = [d for d in dataset if d.get("category") == category_filter]

    if limit and limit > 0:
        dataset = dataset[:limit]

    print(f"Loaded {len(dataset)} evaluation questions.")
    print(f"Target endpoint: {endpoint}")

    # Inspect RAG service independently to capture precise retrieved chunks
    rag_service = get_rag_service()

    raw_results = []

    for idx, item in enumerate(dataset, 1):
        qid = item["id"]
        category = item["category"]
        question = item["question"]
        rag_mode = item.get("rag_mode", "auto")
        model_mode = mode_override or item.get("model_mode", "auto")

        print(f"\n[{idx}/{len(dataset)}] Running {qid} ({category}) | RAG: {rag_mode} | Model: {model_mode}")
        print(f"  Q: {question[:80]}...")

        # 1. Capture exact RAG retrieval context and chunks independently
        rag_res = rag_service.retrieve_context(query=question, mode=rag_mode)
        retrieved_chunks_meta = []
        for c in rag_res.chunks:
            retrieved_chunks_meta.append({
                "file_name": c.file_name,
                "source_path": c.source_path,
                "category": c.category,
                "similarity": c.similarity,
                "score": c.score,
                "page_number": c.page_number,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "symbol_name": c.symbol_name,
                "content_snippet": c.content[:150] + ("..." if len(c.content) > 150 else ""),
            })

        # 2. Execute via production chat endpoint
        exec_res = execute_chat_query(
            question=question,
            rag_mode=rag_mode,
            model_mode=model_mode,
            endpoint=endpoint,
        )

        effective_route = exec_res.get("route_info", {}).get("route", "chat")
        ans = exec_res.get("generated_answer", "")
        success = exec_res.get("success", False)
        tot_ms = exec_res.get("total_latency_ms", 0.0)
        ttft_ms = exec_res.get("ttft_ms")

        print(f"  -> Route: {effective_route} | TTFT: {ttft_ms}ms | Latency: {tot_ms}ms | Success: {success}")
        print(f"  -> Retrieved Chunks: {len(rag_res.chunks)} | Answer length: {len(ans)} chars")

        result_record = {
            "id": qid,
            "category": category,
            "question": question,
            "expected_answer": item.get("expected_answer", ""),
            "expected_facts": item.get("expected_facts", []),
            "expected_sources": item.get("expected_sources", []),
            "expected_locations": item.get("expected_locations", []),
            "evidence_requirements": item.get("evidence_requirements", ""),
            "abstention_expected": item.get("abstention_expected", False),
            "rag_mode": rag_mode,
            "model_mode_requested": model_mode,
            "effective_route": effective_route,
            "retrieval_status": rag_res.status,
            "retrieved_chunks_count": len(rag_res.chunks),
            "retrieved_chunks": retrieved_chunks_meta,
            "context_tokens_approx": rag_res.token_count,
            "generated_answer": ans,
            "success": success,
            "error": exec_res.get("error"),
            "total_latency_ms": tot_ms,
            "ttft_ms": ttft_ms,
            "stage_events": exec_res.get("stage_events", []),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        raw_results.append(result_record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Execution complete. Saved {len(raw_results)} results to '{output_path}'.")
    print("=" * 70)
    return raw_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7.2 Answer Quality Evaluation Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to run")
    parser.add_argument("--category", type=str, default=None, help="Filter queries by category")
    parser.add_argument("--mode", type=str, default=None, help="Override model mode (e.g. chat, pipeline, gemini)")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="Chat SSE API endpoint")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Path to save output JSON")
    args = parser.parse_args()

    run_evaluation(
        limit=args.limit,
        category_filter=args.category,
        mode_override=args.mode,
        endpoint=args.endpoint,
        output_path=args.output,
    )
