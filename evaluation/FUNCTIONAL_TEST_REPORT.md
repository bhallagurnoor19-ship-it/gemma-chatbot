# Subsystem Functional Test & Verification Report

**Project:** `C:\Users\Gurnoor\Desktop\gemma-chatbot`  
**Evaluation Target:** Document RAG, Codebase RAG, PDF Intelligence & OCR, Model Lifecycle Leases, Retrieval Modes, and Fault Tolerance  
**Evaluation Date:** 2026-11-10  
**Phase:** Comprehensive Evaluation & Verification  

---

## 1. Executive Summary

This report provides the functional evaluation of the chatbot's core underlying subsystems:
1. **RAG Vector Collections:** Collection isolation, chunk volume, and retrieval mode routing (`documents`, `codebase`, `both`, `off`).
2. **Manifest Integrity & Resilience:** Verified handling of missing, corrupt, or out-of-sync index manifests.
3. **PDF Intelligence & OCR Extraction:** Validated on actual FLAT examination PDFs located on this computer.
4. **Model Lifecycle & Lease Management:** Verified generation lease acquisition, resident model protection, and release mechanics.

---

## 2. RAG Collection Isolation & Retrieval Modes

### 2.1 Production Index Inventory
The ChromaDB vector database (`data/chroma_db`) was inspected:
- **`documents_collection`:** 10 active chunks (`cloud_deployment_guide.pdf`, `api_specifications.md`, `incident_runbook.txt`).
- **`codebase_collection`:** 713 active chunks indexing the complete repository source tree.
- **`conversation_memory_collection`:** 216 historical conversational turns.

### 2.2 Mode Selection Verification

| Requested Mode | Effective Mode | Retrieval Status | Chunks Returned | Content Categories | Verification Outcome |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`documents`** | `documents` | `success` | 4 | `{'document'}` | **PASS** (Strict document filtering) |
| **`codebase`** | `codebase` | `success` | 4 | `{'codebase'}` | **PASS** (Strict codebase filtering) |
| **`both`** | `both` | `success` | 4 | `{'document', 'codebase'}` | **PASS** (Balanced hybrid retrieval) |
| **`off`** | `off` | `off` | 0 | `set()` | **PASS** (Zero chunk retrieval) |

### 2.3 Resiliency & Fault Handling
1. **Empty Query:** Passed `query=""` $\rightarrow$ returned structured `RAGContextResult(mode="off", status="off", chunks=[])` in **0.015 ms** with zero unhandled exceptions.
2. **Missing Manifest File:** Configured non-existent path `C:\non_existent_manifest.json` $\rightarrow$ intercepted cleanly by `_load_manifest()`, returning `RetrievalStatus.DATABASE_ERROR` with explicit error diagnostic *"Index manifest error: Index manifest does not exist..."*. Prevents querying against unverified or stale vector chunks.

---

## 3. PDF Intelligence & OCR on Local Exam PDFs

### 3.1 Dataset Verification
- **Upload Directory:** `data/pdf_intelligence/uploads`
- **Total Registered PDFs:** 8 PDF files
- **Sample Inspected:** `doc_107fa78da87a.pdf` (Original: `FLAT-_5th_-Dec2025-2.pdf`)

### 3.2 Inspection & OCR Execution Results

| Attribute | Measured Value | Standard / Requirement | Status |
|:---|:---:|:---:|:---:|
| **Page Count** | 2 | Matching physical PDF | PASS |
| **Classification: SCANNED_IMAGE** | 2 pages (100%) | Correctly detected lack of digital font streams | PASS |
| **Classification: DIGITAL_TEXT** | 0 pages | No false digital classifications | PASS |
| **OCR Engine Available** | `True` | Tessdata located at `data/pdf_intelligence/tessdata` | PASS |
| **Total Inspection & OCR Duration** | **1,018.32 ms** | $< 2,500\text{ ms}$ per page | PASS |

---

## 4. Model Lifecycle & Lease Management

Tested via `OllamaLifecycleManager` (`model_lifecycle.py`):
1. **Atomic Generation Lease Acquisition:** Successfully acquired lease for `gemma4:12b` (`lease.is_active = True`).
2. **Active Resident Tracking:** `self._resident_gen_model` correctly tracks `gemma4:12b`.
3. **Automatic Context Exit Release:** Context manager exit releases generation lease and notifies waiting threads via condition variable `self._cv`.
4. **Adaptive Embedding Policy:** Prevents simultaneous execution of heavy generation models (`deepseek-coder-v2:16b`, 8.9 GB) alongside embedding models in 16 GB RAM.

---

## 5. Artifact Data Links

- **Full Machine-Readable JSON:** [`evaluation/data/functional_test_results.json`](file:///C:/Users/Gurnoor/Desktop/gemma-chatbot/evaluation/data/functional_test_results.json)
- **Evaluation Runner:** [`evaluation/scripts/eval_functional_subsystems.py`](file:///C:/Users/Gurnoor/Desktop/gemma-chatbot/evaluation/scripts/eval_functional_subsystems.py)
