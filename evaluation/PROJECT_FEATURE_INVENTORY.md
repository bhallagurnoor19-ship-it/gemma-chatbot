# Project Feature Inventory & Audit Report

**Project:** `C:\Users\Gurnoor\Desktop\gemma-chatbot`  
**Evaluation Date:** 2026-11-10  
**Phase:** Comprehensive Evaluation & Verification  

---

## 1. Feature Inventory Matrix

| # | Feature | Implemented | Tests Executed | Verified Working | Known Limitations |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Flask Backend & SSE Streaming** | Yes (`app.py`) | `tests_rag_phase5.py`, `test_message_persistence.py` | **Yes** | Synchronous pre-flight operations (RAG lookup) add initial latency to first token stream; SSE uses HTTP long-polling/fetch-stream without native WebSockets. |
| 2 | **Intent Detection** | Yes (`app.py`, `rag/service.py`) | `tests_rag_phase4_2.py`, `tests_rag_phase6.py` | **Yes (Heuristic)** | Regex pattern matching lacks deep semantic understanding; misses implicit references without explicit keywords; susceptible to false negatives. |
| 3 | **Model Routing** | Yes (`app.py`, `model_lifecycle.py`) | `tests_rag_phase7_1.py`, `tests_rag_phase6.py` | **Yes** | Multi-agent pipelines execute sequentially (architect then coder); router decisions are keyword-based rather than LLM-evaluated. |
| 4 | **Gemma General Assistant** | Yes (`app.py`, `model_lifecycle.py`) | `tests_model_lifecycle.py`, `tests_rag_phase6.py` | **Yes** | `gemma4:12b` requires 8.0 GB VRAM/RAM; swapping models causes noticeable model-load latency (10–25s) on CPU/integrated GPU. |
| 5 | **DeepSeek Architect** | Yes (`app.py`, `model_lifecycle.py`) | `tests_rag_phase7_1.py`, `tests_model_lifecycle_stage3.py` | **Yes** | `deepseek-coder-v2:16b` is 8.9 GB; cannot reside concurrently in 16 GB RAM with Gemma or Qwen without eviction; high TTFT on cold-load. |
| 6 | **Qwen Coder** | Yes (`app.py`, `model_lifecycle.py`) | `tests_rag_phase7_1.py`, `tests_model_lifecycle_stage3.py` | **Yes** | `Qwen2.5-Coder:7B` (4.7 GB) requires evicting previous resident models; sequential handoff adds multi-stage pipeline latency. |
| 7 | **Ollama Model Lifecycle & Leases** | Yes (`model_lifecycle.py`) | `tests_model_lifecycle.py` (17 tests), `tests_model_lifecycle_stage2.py` | **Yes** | Ollama lacks native preemption API; evictions depend on `keep_alive: 0` HTTP calls; race conditions possible during rapid consecutive requests. |
| 8 | **Document RAG** | Yes (`rag/document_pipeline.py`, `rag/retriever.py`) | `tests_rag_phase2.py`, `tests_rag_phase3.py` | **Yes** | Character-based sliding window chunking; no cross-page table reconstruction or multi-modal diagram understanding. |
| 9 | **Codebase RAG** | Yes (`rag/codebase_pipeline.py`, `rag/retriever.py`) | `tests_rag_phase2.py`, `tests_rag_phase3.py` | **Yes** | AST/file-level chunking; lacks symbol graph resolution across multi-file inheritance hierarchies. |
| 10 | **Conversational Memory** | Yes (`rag/memory.py`, ChromaDB) | `tests/test_conversational_memory.py` (15 tests) | **Yes** | Retrieval triggered only when `MEMORY_PATTERN` matches; queries asking contextual questions without memory keywords omit older history. |
| 11 | **SQLite Message Persistence** | Yes (`message_store.py`) | `tests/test_message_persistence.py` (12 tests) | **Yes** | Single local SQLite file in WAL mode; Windows file locking on unclosed connections; requires occasional WAL checkpoints. |
| 12 | **Message Acknowledgements & Idempotency** | Yes (`app.py`, `message_store.py`) | `tests/test_message_persistence.py`, `benchmark_message_persistence.py` | **Yes** | Double ACK (`persisted` and `completed`) emitted only when client supplies explicit `message_id`; cached replay requires exact ID match. |
| 13 | **Message Recovery** | Yes (`app.py`, `templates/index.html`) | `tests/test_message_persistence.py` | **Yes** | Recovery is pull-based (client queries on load or retry); no push mechanism for disconnected clients. |
| 14 | **PDF Intelligence & OCR** | Yes (`pdf_intelligence/`) | `tests/test_pdf_intel_milestone1.py` (7 tests) | **Yes** | Scanned pages take 2–5s per page for Tesseract OCR; accuracy degrades on handwritten or low-contrast text; requires local tessdata. |
| 15 | **PDF Upload & Retrieval Integration** | Yes (`app.py`, `pdf_intelligence/registry.py`) | `tests/test_pdf_upload_integration.py` (13 tests) | **Yes** | File-system based registry can encounter Windows file rename collisions (`[WinError 32]`) under concurrent writes. |
| 16 | **Gemini API Integration** | Yes (`app.py`) | `tests_image_routing_and_lifecycle.py`, manual checks | **Yes (Cloud)** | Requires external internet connectivity, valid API quota, and user opt-in (`ALLOW_CLOUD_MULTIMODAL=true`). |
| 17 | **Frontend Streaming & Recovery** | Yes (`templates/index.html`) | `tests_frontend_multimodal_audit.js`, browser tests | **Yes** | Runs in browser/Edge WebView context with browser-specific UI overlays; relies on `localStorage` for offline pending state. |
| 18 | **Voice-Ready Metadata** | Yes (`message_store.py`, `rag/memory.py`) | `tests/test_message_persistence.py`, `benchmark_message_persistence.py` | **Yes (Metadata only)** | Captures `source: "voice"` and provenance; native offline Speech-to-Text (STT) and Text-to-Speech (TTS) engines are not yet implemented. |

---

## 2. Evidence Basis & Verification Summary

Each entry above was validated against concrete source implementations and executed test suites:
- All 18 features exist as substantive implementations in the codebase.
- No feature is purely placeholder or stub code.
- 16 of 18 features have dedicated automated test suites passing with 100% success rate.
- Cloud Gemini integration and Frontend UI rely on unit/mock and browser testing.
