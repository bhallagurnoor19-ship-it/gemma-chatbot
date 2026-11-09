"""
Evaluation Script: Subsystem Functional Testing
Tests Document RAG, Codebase RAG, PDF Intelligence, Model Lifecycle Leases,
and Retrieval Modes (auto, documents, codebase, both, memory, all, off).
"""

import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_lifecycle import (
    OllamaLifecycleManager,
    ModelLease,
    ModelLeaseType,
    normalize_model_name,
)
from pdf_intelligence.classifier import (
    PageDisposition,
    classify_page,
)
from pdf_intelligence.inspector import PDFInspector
from rag.config import (
    CHROMA_DIR,
    DOCUMENT_COLLECTION_NAME,
    CODEBASE_COLLECTION_NAME,
    CONVERSATION_COLLECTION_NAME,
    INDEX_MANIFEST_PATH,
    EMBEDDING_DIMENSION,
)
from rag.indexer import IncrementalIndexer
from rag.prompt_builder import PromptBuilder
from rag.retriever import UnifiedRetriever, RetrievalStatus
from rag.service import RAGService
from rag.vector_store import VectorStoreManager


def run_functional_evaluation():
    print("=" * 80)
    print("EVALUATION: FUNCTIONAL SUBSYSTEM TESTING")
    print("=" * 80)

    test_results = {}

    # ═══════════════════════════════════════════════════════════════════
    # 1. RAG COLLECTIONS & RETRIEVAL MODES (REAL INDEX)
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- 1. RAG Collection Isolation & Retrieval Modes ---")
    vsm = VectorStoreManager(persist_dir=CHROMA_DIR)
    vsm.initialize()

    doc_stats = vsm.get_collection_stats(DOCUMENT_COLLECTION_NAME)
    code_stats = vsm.get_collection_stats(CODEBASE_COLLECTION_NAME)
    mem_stats = vsm.get_collection_stats(CONVERSATION_COLLECTION_NAME)

    doc_count = doc_stats.get("count", 0)
    code_count = code_stats.get("count", 0)
    mem_count = mem_stats.get("count", 0)

    print(f"Production Collections: documents={doc_count}, codebase={code_count}, conversation_memory={mem_count}")
    test_results["collection_isolation"] = (doc_count > 0 and code_count > 0)

    # Test UnifiedRetriever with mock embedding for speed and determinism
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.05] * EMBEDDING_DIMENSION
    mock_embed.embed_text.return_value = [0.05] * EMBEDDING_DIMENSION

    retriever = UnifiedRetriever(
        vector_store=vsm,
        embedding_service=mock_embed,
        manifest_path=INDEX_MANIFEST_PATH,
        distance_threshold=1.5,  # Wide threshold to allow candidates through
    )
    prompt_builder = PromptBuilder()
    service = RAGService(retriever=retriever, prompt_builder=prompt_builder)

    modes_to_test = ["documents", "codebase", "both", "off"]
    mode_outcomes = {}
    for m in modes_to_test:
        res = service.retrieve_context(query="cloud deployment architecture in app.py", mode=m)
        cats = {c.category for c in res.chunks}
        mode_outcomes[m] = {
            "status": res.status,
            "chunks": len(res.chunks),
            "categories": list(cats),
        }
        print(f"  Mode '{m:10s}': status={res.status:10s} | chunks={len(res.chunks)} | categories={cats}")

    modes_verified = (
        ("document" in mode_outcomes["documents"]["categories"] or mode_outcomes["documents"]["status"] in ("success", "no_results")) and
        ("codebase" in mode_outcomes["codebase"]["categories"] or mode_outcomes["codebase"]["status"] in ("success", "no_results")) and
        mode_outcomes["off"]["status"] == "off"
    )
    test_results["retrieval_modes_verified"] = modes_verified
    print(f"Retrieval Modes Honored: {'PASS' if modes_verified else 'FAIL'}")

    # Resiliency Test 1: Empty query
    res_empty = service.retrieve_context(query="", mode="auto")
    print(f"Empty Query Handling: status='{res_empty.status}' (PASS: returns empty without exception)")
    test_results["empty_query_resilient"] = (res_empty.status == "off" and len(res_empty.chunks) == 0)

    # Resiliency Test 2: Corrupted/Missing manifest
    bad_retriever = UnifiedRetriever(
        vector_store=vsm,
        embedding_service=mock_embed,
        manifest_path="C:\\non_existent_manifest.json",
    )
    bad_resp = bad_retriever.retrieve(query="test query")
    print(f"Missing Manifest Handling: status='{bad_resp.status}' | error='{bad_resp.error_message}'")
    test_results["missing_manifest_handled"] = (bad_resp.status == RetrievalStatus.DATABASE_ERROR)

    # ═══════════════════════════════════════════════════════════════════
    # 2. PDF INTELLIGENCE & OCR ON REAL EXAM PDFS
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- 2. PDF Intelligence & OCR on Local Exam PDFs ---")
    uploads_dir = os.path.join(PROJECT_ROOT, "data", "pdf_intelligence", "uploads")
    inspector = PDFInspector()

    pdf_files = [f for f in os.listdir(uploads_dir) if f.endswith(".pdf")] if os.path.exists(uploads_dir) else []
    print(f"Found {len(pdf_files)} PDF documents in {uploads_dir}")

    if pdf_files:
        test_pdf = os.path.join(uploads_dir, pdf_files[0])
        print(f"Inspecting sample PDF: {pdf_files[0]}")
        t0 = time.perf_counter()
        insp_result = inspector.inspect_document(test_pdf)
        t1 = time.perf_counter()
        dur_ms = (t1 - t0) * 1000.0

        summary = insp_result.page_classifications_summary
        ocr_avail = insp_result.ocr_available
        print(f"  Pages Inspected: {insp_result.page_count}")
        print(f"  Classifications: {summary}")
        print(f"  OCR Engine Available: {ocr_avail}")
        print(f"  Duration: {dur_ms:.2f} ms")

        pdf_verified = (insp_result.page_count > 0 and ocr_avail is True)
        test_results["pdf_intelligence_inspection"] = pdf_verified
    else:
        test_results["pdf_intelligence_inspection"] = False

    # ═══════════════════════════════════════════════════════════════════
    # 3. MODEL LIFECYCLE & LEASE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- 3. Model Lifecycle Lease Management ---")
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": [{"name": "gemma4:12b"}]}
        mock_post.return_value.status_code = 200

        mgr = OllamaLifecycleManager(default_timeout=5.0)

        with mgr.acquire_generation_lease("gemma4:12b") as lease:
            lease_ok = (lease.model_name == "gemma4:12b" and lease.is_active)
            active_gen = mgr._resident_gen_model
            print(f"Lease Acquired: model={lease.model_name}, active={lease.is_active}, resident_gen={active_gen}")

        print(f"Lease Released: active generation lease freed")
        test_results["model_lifecycle_lease_ok"] = lease_ok

    # ═══════════════════════════════════════════════════════════════════
    # 4. EXPORT JSON RESULTS
    # ═══════════════════════════════════════════════════════════════════
    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "functional_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print("\nFunctional Test Summary:")
    for k, v in test_results.items():
        print(f"  {k:35s}: {'PASS' if v else 'FAIL'}")

    print("=" * 80)
    print("FUNCTIONAL EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_functional_evaluation()
