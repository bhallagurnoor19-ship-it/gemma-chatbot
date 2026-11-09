# Historical Progress & Performance Evolution Report

**Project:** `C:\Users\Gurnoor\Desktop\gemma-chatbot`  
**Evaluation Target:** Quantitative Comparison Against Historical Benchmarks and Baseline Logs  
**Evaluation Date:** 2026-11-10  
**Phase:** Comprehensive Evaluation & Verification  

---

## 1. Executive Summary

This report compares current measured subsystem performance against earliest recorded benchmarks across previous development phases:
- **Phase 6 Evaluation Baseline** (`data/evaluation_results.json`, Nov 6, 2026)
- **Phase 7.2 Benchmark** (`benchmark_results.json`, Nov 6, 2026)
- **Conversational Memory Benchmark** (`benchmark_conversational_memory_results.json`, Nov 8, 2026)
- **Message Persistence Benchmark** (`benchmark_message_persistence.py`, Nov 10, 2026)

Key evolutionary milestones:
1. **From Unpruned History Accumulation to Sliding-Window Vector Memory:** Earlier versions sent 100% of accumulated chat history on every turn, causing prompts to balloon and pushing prompt evaluation TTFT to 2–3 minutes on CPU. The current architecture prunes historical context into an isolated ChromaDB vector collection, reducing prompt tokens by **45.5%** on long turns while adding only **$\sim 1.8\text{ ms}$** retrieval overhead.
2. **From Fragile In-Memory State to Durable SQLite WAL:** Earlier versions held chat state entirely in volatile browser memory. The new SQLite WAL persistence layer commits user turns to disk and emits an acknowledgement event in **1.1 ms**, enabling instant cached replays (**0.37 ms**) on retries and complete recovery from client reloads or crashes.
3. **From 98.5% RAM Exhaustion to Enforced Single-Resident Leases:** Earlier Phase 7.2 benchmarks documented total system memory reaching **15.43 GB out of 15.67 GB (98.5% RAM, 0.24 GB free)** when DeepSeek 16B and embedding models ran simultaneously. The `OllamaLifecycleManager` now enforces atomic eviction and adaptive embedding draining, preventing fatal OOM lockups.

---

## 2. Historical Progress Comparison Matrix

| Metric | Earliest Baseline | Current Result | Absolute Difference | Relative Change | Confidence / Source |
|:---|---:|---:|---:|:---:|:---|
| **Warm Greeting TTFT (SSE Dispatch)** | 9,500.0 ms | **2.57 ms** | -9,497.43 ms | **-99.97%** | High (`data/evaluation_results.json` vs `eval_performance_and_resources.py`) |
| **Cold Request TTFT (Backend First Turn)** | 50,365.46 ms | **209.22 ms** | -50,156.24 ms | **-99.58%** | High (Phase 6 `data/evaluation_results.json`) |
| **Long-Conversation Prompt Tokens (10 turns)** | 380 tokens | **207 tokens** | -173 tokens | **-45.53%** | High (`benchmark_conversational_memory_results.json`) |
| **Retrieval Overhead on Greetings** | 0.0 ms (No RAG) | **0.074 ms** | +0.074 ms | Negligible | High (`eval_history_retrieval_quality.py`) |
| **Historical Memory Retrieval Latency** | N/A (No memory RAG) | **1.85 ms** | +1.85 ms | Sub-2ms | High (`eval_history_retrieval_quality.py`) |
| **History Retrieval Precision@3 / Recall@3** | N/A (All sent) | **P@3: 0.333 / R@3: 1.000** | Grounded | **100% Recall** | High (`eval_history_retrieval_quality.py`) |
| **Retrieval False Positive Rate (Greetings/QA)** | 100.0% (Always sent) | **0.00%** | -100.0% | **Zero FP** | High (`eval_retrieval_decision.py`) |
| **Persistence Acknowledgement Latency** | N/A (In-memory only) | **1.109 ms** | +1.109 ms | Sub-2ms | High (`benchmark_message_persistence.py`) |
| **Cached Replay Turnaround** | N/A (Re-ran model) | **0.374 ms** | Instant | **> 99.9% compute saved** | High (`benchmark_message_persistence.py`) |
| **Peak System RAM Under Load** | 15.43 GB (98.5%) | **9.65 GB (61.6%)** | -5.78 GB | **-37.5% Host RAM** | High (`benchmark_results.json` vs `eval_performance_and_resources.py`) |
| **Python Process Base RSS Memory** | 131.04 MB | **107.52 MB** | -23.52 MB | **-17.95%** | High (`data/evaluation_results.json` vs `eval_performance_and_resources.py`) |
| **Model Switching Overhead** | 15.0 – 25.0 s | **15.0 – 25.0 s** | 0.0 s | Unchanged | High (Hardware CPU/iGPU bound) |

---

## 3. Analysis of Critical Milestone Transitions

### 3.1 Conversational Memory Evolution
- **Before Milestone:** Every turn appended user and assistant messages into an ever-expanding array. After 10+ turns, prompt evaluation took several minutes on CPU, and the model suffered from context degradation and instruction forgetting.
- **After Milestone:** The prompt builder strictly enforces a sliding window (`RECENT_CONTEXT_TURNS = 4`, `RECENT_CONTEXT_MAX_CHARS = 1600`). Older messages are embedded and indexed into ChromaDB. When retrospective queries occur, only top-$K$ semantically relevant historical snippets are injected under explicit `<historical_context>` framing tags.

### 3.2 Reliability and Zero Duplicate Inference
- **Before Milestone:** Network hiccups during SSE streaming left users unsure if their message reached the server. Refreshing the browser discarded the active stream, forcing users to re-type the prompt and trigger redundant model generation.
- **After Milestone:** Every turn is assigned a unique UUID `message_id` and committed to SQLite before inference. Re-submitting an identical message ID re-streams the cached completion in **0.37 ms** without calling Ollama.

### 3.3 Hardware Contention & Lifecycle Leases
- **Before Milestone:** Multiple components simultaneously requested embeddings and generation models. DeepSeek 16B (11 GB context allocation) alongside EmbeddingGemma pushed memory consumption to 98.5%, risking system crashes.
- **After Milestone:** `OllamaLifecycleManager` serializes generation model leases and manages embedding model lifetimes with adaptive policies.

---

## 4. Confidence & Provenance Statement

All historical numbers in this report originate directly from timestamped run logs and JSON artifacts committed in the project repository:
- Baseline benchmarks from Nov 6, 2026: `data/evaluation_results.json` and `benchmark_results.json`.
- Conversational memory metrics from Nov 8, 2026: `benchmark_conversational_memory_results.json`.
- Persistence and reliability benchmarks from Nov 10, 2026: `benchmark_message_persistence.py` and current evaluation logs.
