"""
Phase 7 - Part 5 Reliability & Failure Recovery Suite
Executes the 10 failure and recovery scenarios and validates resilient behavior.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import (
    CHROMA_DIR,
    INDEX_MANIFEST_PATH,
    OLLAMA_BASE_URL,
    PRIMARY_EMBEDDING_MODEL,
    DOCUMENT_COLLECTION_NAME,
    CODEBASE_COLLECTION_NAME,
)
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager
from rag.retriever import UnifiedRetriever, RetrievalStatus
from rag.indexer import IncrementalIndexer
from rag.service import RAGService


def test_scenario_1_ollama_unavailable():
    """Scenario 1: Ollama service is unavailable/down."""
    print("\n--- Scenario 1: Ollama Down ---")
    mock_service = EmbeddingService(base_url="http://127.0.0.1:99999", model_name="embeddinggemma")
    vector_store = VectorStoreManager(CHROMA_DIR)
    retriever = UnifiedRetriever(vector_store, mock_service)
    rag_service = RAGService(retriever)
    
    result = rag_service.retrieve_context("What is the network latency?", mode="documents")
    assert result.status in ("error", "embedding_unavailable", "no_results")
    assert not result.has_context
    assert result.context_text == ""
    print("PASS: System returned graceful fallback result with status:", result.status)
    return True


def test_scenario_2_embedding_model_missing():
    """Scenario 2: Embedding model is missing."""
    print("\n--- Scenario 2: Embedding Model Missing ---")
    bad_service = EmbeddingService(OLLAMA_BASE_URL, "non_existent_model_12345")
    vector_store = VectorStoreManager(CHROMA_DIR)
    retriever = UnifiedRetriever(vector_store, bad_service)
    
    response = retriever.retrieve("Test query", mode="documents")
    assert response.status == RetrievalStatus.EMBEDDING_UNAVAILABLE
    assert response.error_message is not None
    print("PASS: System returned EMBEDDING_UNAVAILABLE with error:", response.error_message)
    return True


def test_scenario_3_chromadb_unavailable():
    """Scenario 3: ChromaDB is unavailable/corrupt."""
    print("\n--- Scenario 3: ChromaDB Unavailable ---")
    bad_store = VectorStoreManager(persist_dir="Z:\\invalid_path_does_not_exist\\chroma")
    embedding_service = EmbeddingService(OLLAMA_BASE_URL, PRIMARY_EMBEDDING_MODEL)
    retriever = UnifiedRetriever(bad_store, embedding_service)
    
    with patch.object(bad_store, "get_collection", side_effect=Exception("ChromaDB connection refused")):
        response = retriever.retrieve("Test query", mode="documents")
        assert response.status == RetrievalStatus.DATABASE_ERROR
        print("PASS: System safely trapped ChromaDB exception and returned DATABASE_ERROR")
    return True


def test_scenario_4_missing_or_malformed_manifest():
    """Scenario 4: Missing or malformed manifest."""
    print("\n--- Scenario 4: Missing or Malformed Manifest ---")
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        tmp.write("{ invalid json syntax ...")
        malformed_path = tmp.name

    try:
        vector_store = VectorStoreManager(CHROMA_DIR)
        embedding_service = EmbeddingService(OLLAMA_BASE_URL, PRIMARY_EMBEDDING_MODEL)
        retriever = UnifiedRetriever(vector_store, embedding_service, manifest_path=malformed_path)
        
        response = retriever.retrieve("What is network latency?", mode="documents")
        assert response.status == RetrievalStatus.DATABASE_ERROR
        assert len(response.chunks) == 0
        print("PASS: Malformed manifest safely yielded DATABASE_ERROR with 0 unfiltered chunk leakage.")
    finally:
        if os.path.exists(malformed_path):
            os.remove(malformed_path)
    return True


def test_scenario_5_empty_collections():
    """Scenario 5: Empty collections."""
    print("\n--- Scenario 5: Empty Collections ---")
    tmp_chroma = tempfile.mkdtemp()
    try:
        vector_store = VectorStoreManager(tmp_chroma)
        vector_store.initialize()
        embedding_service = EmbeddingService(OLLAMA_BASE_URL, PRIMARY_EMBEDDING_MODEL)
        retriever = UnifiedRetriever(vector_store, embedding_service)
        
        response = retriever.retrieve("Any query in empty store", mode="documents")
        assert response.status in (RetrievalStatus.NO_RESULTS, RetrievalStatus.SUCCESS)
        assert len(response.chunks) == 0
        print("PASS: Empty collection returned 0 chunks without error.")
    finally:
        shutil.rmtree(tmp_chroma, ignore_errors=True)
    return True


def test_scenario_6_deleted_source_files():
    """Scenario 6: Deleted source files handling."""
    print("\n--- Scenario 6: Deleted Source Files ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        test_file = os.path.join(tmp_dir, "temp_doc.txt")
        with open(test_file, "w") as f:
            f.write("Temporary content for deletion test.")

        manifest_file = os.path.join(tmp_dir, "manifest.json")
        chroma_dir = os.path.join(tmp_dir, "chroma")
        vsm = VectorStoreManager(chroma_dir)
        vsm.initialize()

        class MockEmbeddingService:
            def embed_documents(self, t): return [[0.1]*768]*len(t)
            def embed_batch(self, t): return [[0.1]*768]*len(t)
            def embed_query(self, t): return [0.1]*768
            def embed_text(self, t): return [0.1]*768
            def get_dimension(self): return 768

        indexer = IncrementalIndexer(
            vector_store=vsm,
            embedding_service=MockEmbeddingService(),
            manifest_path=manifest_file,
            base_dir=tmp_dir,
        )

        # Index once
        s1 = indexer.index_documents(target_dir=tmp_dir)
        assert s1.get("indexed", 0) >= 1

        # Delete file
        os.remove(test_file)

        # Re-index
        s2 = indexer.index_documents(target_dir=tmp_dir)
        assert s2.get("deleted", 0) >= 1
        print("PASS: File deletion detected and purged from index during subsequent indexing run.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def test_scenario_7_modified_files_reindexing():
    """Scenario 7: Modified files re-indexing."""
    print("\n--- Scenario 7: Modified Files Re-Indexing ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        test_file = os.path.join(tmp_dir, "doc_v1.txt")
        with open(test_file, "w") as f:
            f.write("Initial version of file.")

        manifest_file = os.path.join(tmp_dir, "manifest.json")
        chroma_dir = os.path.join(tmp_dir, "chroma")
        vsm = VectorStoreManager(chroma_dir)
        vsm.initialize()

        class MockEmbeddingService:
            def embed_documents(self, t): return [[0.1]*768]*len(t)
            def embed_batch(self, t): return [[0.1]*768]*len(t)
            def embed_query(self, t): return [0.1]*768
            def embed_text(self, t): return [0.1]*768
            def get_dimension(self): return 768

        indexer = IncrementalIndexer(
            vector_store=vsm,
            embedding_service=MockEmbeddingService(),
            manifest_path=manifest_file,
            base_dir=tmp_dir,
        )

        s1 = indexer.index_documents(target_dir=tmp_dir)
        assert s1.get("indexed", 0) == 1

        with open(manifest_file, "r") as f:
            m1 = json.load(f)
        doc_entries1 = m1["documents"]
        entry1 = next(iter(doc_entries1.values()))
        gen1 = entry1.get("generation_id") or entry1.get("active_generation")

        # Modify file
        with open(test_file, "w") as f:
            f.write("Second modified version with altered content.")

        s2 = indexer.index_documents(target_dir=tmp_dir)
        assert s2.get("updated", 0) == 1 or s2.get("indexed", 0) == 1

        with open(manifest_file, "r") as f:
            m2 = json.load(f)
        doc_entries2 = m2["documents"]
        entry2 = next(iter(doc_entries2.values()))
        gen2 = entry2.get("generation_id") or entry2.get("active_generation")

        assert gen1 != gen2, f"Expected generation change upon file modification: {gen1} vs {gen2}"
        print(f"PASS: Modified file transitioned from generation {gen1} to {gen2} successfully.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def test_scenario_8_repeated_indexing_idempotency():
    """Scenario 8: Repeated indexing without changes (idempotency)."""
    print("\n--- Scenario 8: Repeated Indexing Idempotency ---")
    indexer = IncrementalIndexer()
    doc_stats = indexer.index_documents()
    
    assert doc_stats.get("indexed", 0) == 0, f"Expected 0 new indexed doc files on repeat run, got {doc_stats}"
    assert doc_stats.get("skipped", 0) >= 3, f"Expected doc files skipped, got {doc_stats}"
    print(f"PASS: 100% idempotent. Docs skipped: {doc_stats['skipped']}, newly indexed: 0.")
    return True


def test_scenario_9_stale_generation_cleanup():
    """Scenario 9: Stale generation cleanup."""
    print("\n--- Scenario 9: Stale Generation Cleanup ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        manifest_file = os.path.join(tmp_dir, "manifest.json")
        chroma_dir = os.path.join(tmp_dir, "chroma")
        vsm = VectorStoreManager(chroma_dir)
        vsm.initialize()

        class MockEmbeddingService:
            def embed_documents(self, t): return [[0.1]*768]*len(t)
            def embed_batch(self, t): return [[0.1]*768]*len(t)
            def embed_query(self, t): return [0.1]*768
            def embed_text(self, t): return [0.1]*768
            def get_dimension(self): return 768

        indexer = IncrementalIndexer(
            vector_store=vsm,
            embedding_service=MockEmbeddingService(),
            manifest_path=manifest_file,
            base_dir=tmp_dir,
        )

        # Insert active chunk and orphaned chunk
        doc_col = vsm.get_collection(DOCUMENT_COLLECTION_NAME)
        active_id = "doc:file1.txt:c001:g11111111"
        orphan_id = "doc:file1.txt:c001:g99999999"

        doc_col.add(
            ids=[active_id, orphan_id],
            documents=["Active chunk content", "Orphaned chunk content"],
            embeddings=[[0.1]*768, [0.1]*768],
            metadatas=[
                {"generation": "g11111111", "file_name": "file1.txt", "file_path": os.path.abspath(os.path.join(tmp_dir, "file1.txt"))},
                {"generation": "g99999999", "file_name": "file1.txt", "file_path": os.path.abspath(os.path.join(tmp_dir, "file1.txt"))},
            ]
        )

        # Manifest only tracks g11111111
        manifest_data = {
            "version": "1.0",
            "documents": {
                "file1.txt": {
                    "active_generation": "11111111",
                    "chunk_ids": [active_id]
                }
            },
            "codebase": {}
        }
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f)

        # Reload indexer's manifest in memory
        indexer.manifest = indexer.load_manifest()

        cleanup_stats = indexer.clean_orphaned_generations()
        assert cleanup_stats.get("orphaned_chunks_deleted", 0) >= 1

        remaining = doc_col.get(ids=[active_id, orphan_id])
        assert active_id in remaining["ids"]
        assert orphan_id not in remaining["ids"]
        print(f"PASS: Orphaned chunk {orphan_id} purged; active chunk {active_id} preserved.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def test_scenario_10_persistence_across_restarts():
    """Scenario 10: Vector store and manifest persistence across restarts."""
    print("\n--- Scenario 10: Persistence Across Restarts ---")
    vsm = VectorStoreManager(CHROMA_DIR)
    vsm.initialize()
    doc_col = vsm.get_collection(DOCUMENT_COLLECTION_NAME)
    code_col = vsm.get_collection(CODEBASE_COLLECTION_NAME)

    doc_count = doc_col.count()
    code_count = code_col.count()
    assert doc_count >= 10, f"Expected at least 10 document chunks, got {doc_count}"
    assert code_count >= 200, f"Expected at least 200 codebase chunks, got {code_count}"

    assert os.path.exists(INDEX_MANIFEST_PATH)
    with open(INDEX_MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    assert len(manifest.get("documents", {})) >= 3
    assert len(manifest.get("codebase", {})) >= 24

    print(f"PASS: Persistence verified. Document chunks: {doc_count}, Codebase chunks: {code_count}, Manifest entries: {len(manifest['documents']) + len(manifest['codebase'])}.")
    return True


def run_all_scenarios():
    print("="*70)
    print("RUNNING 10 RELIABILITY & FAILURE RECOVERY SCENARIOS")
    print("="*70)
    
    results = {}
    scenarios = [
        ("Scenario 1: Ollama Unavailable", test_scenario_1_ollama_unavailable),
        ("Scenario 2: Embedding Model Missing", test_scenario_2_embedding_model_missing),
        ("Scenario 3: ChromaDB Unavailable", test_scenario_3_chromadb_unavailable),
        ("Scenario 4: Missing/Malformed Manifest", test_scenario_4_missing_or_malformed_manifest),
        ("Scenario 5: Empty Collections", test_scenario_5_empty_collections),
        ("Scenario 6: Deleted Source Files", test_scenario_6_deleted_source_files),
        ("Scenario 7: Modified Source Files", test_scenario_7_modified_files_reindexing),
        ("Scenario 8: Repeated Indexing (Idempotency)", test_scenario_8_repeated_indexing_idempotency),
        ("Scenario 9: Stale Generation Cleanup", test_scenario_9_stale_generation_cleanup),
        ("Scenario 10: Persistence Across Restarts", test_scenario_10_persistence_across_restarts),
    ]

    for name, fn in scenarios:
        try:
            passed = fn()
            results[name] = "PASSED" if passed else "FAILED"
        except Exception as e:
            print(f"FAIL: {name} raised exception: {e}")
            results[name] = f"FAILED: {e}"

    print("\n" + "="*70)
    print("RELIABILITY TEST SUITE SUMMARY:")
    print("="*70)
    for name, status in results.items():
        print(f"  {name}: {status}")

    passed_count = sum(1 for s in results.values() if s == "PASSED")
    print(f"\nTotal: {passed_count}/{len(scenarios)} Scenarios Passed.")
    return results


if __name__ == "__main__":
    run_all_scenarios()
