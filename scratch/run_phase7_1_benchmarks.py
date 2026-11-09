"""
Phase 7.1 - RAG Retrieval Quality Refinement & Performance Benchmark Suite
- Evaluates 50 queries (30 original + 20 new domain-specific queries)
- Compares baseline (unranked / distance-only) vs Phase 7.1 source-aware ranking
- Separates metrics:
  1. Retrieval Success / Hit@K on answerable queries
  2. Precision@K
  3. Recall@K
  4. Citation Accuracy
  5. Answer Groundedness
  6. Correct Abstention Rate on unanswerable queries (NEVER counted as positive retrieval relevance)
- Measures:
  - Cold vs warm embedding latency
  - Batch embedding throughput (batch sizes 8, 16, 32)
  - Retrieval latency with vs without query embedding (pure ChromaDB search)
  - Time-To-First-Token (TTFT)
  - Memory footprints (Flask RSS, VMS, ChromaDB disk footprint)
"""

import sys
import os
import time
import json
import statistics
import psutil
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import (
    CHROMA_DIR,
    DEFAULT_RANKING_CONFIG,
    INDEX_MANIFEST_PATH,
    OLLAMA_BASE_URL,
    PRIMARY_EMBEDDING_MODEL,
    SIMILARITY_DISTANCE_THRESHOLD,
    DEFAULT_TOP_K,
)
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager
from rag.retriever import UnifiedRetriever, RetrievalStatus, RetrievedChunk
from rag.service import RAGService, get_rag_service

# Complete 50-Query Evaluation Dataset
EVALUATION_QUERIES = [
    # ── Original Phase 7 Queries (Q01-Q30) ───────────────────────────────────
    # 1. Document RAG (10 queries)
    {
        "id": "Q01",
        "domain": "document",
        "rag_mode": "documents",
        "query": "What is the maximum allowable network latency between availability zones in the cluster topology?",
        "expected_source": "cloud_deployment_guide.pdf",
        "expected_page_line": "Page 1",
        "key_facts": ["2.5 milliseconds", "2.5 ms", "availability zones"],
        "is_answerable": True,
    },
    {
        "id": "Q02",
        "domain": "document",
        "rag_mode": "documents",
        "query": "Which TLS cipher suites are mandatory for all internal microservice mTLS communication?",
        "expected_source": "cloud_deployment_guide.pdf",
        "expected_page_line": "Page 2",
        "key_facts": ["TLS 1.3", "ECDHE-RSA-AES256-GCM-SHA384", "TLS_AES_256_GCM_SHA384"],
        "is_answerable": True,
    },
    {
        "id": "Q03",
        "domain": "document",
        "rag_mode": "documents",
        "query": "What is the certificate rotation interval and when must automated renewal be triggered?",
        "expected_source": "cloud_deployment_guide.pdf",
        "expected_page_line": "Page 2",
        "key_facts": ["24 hours", "50% lifetime", "12 hours", "SPIFFE"],
        "is_answerable": True,
    },
    {
        "id": "Q04",
        "domain": "document",
        "rag_mode": "documents",
        "query": "What are the Recovery Time Objective (RTO) and Recovery Point Objective (RPO) targets for tier-1 databases?",
        "expected_source": "cloud_deployment_guide.pdf",
        "expected_page_line": "Page 3",
        "key_facts": ["RTO < 15 minutes", "RPO < 60 seconds", "tier-1"],
        "is_answerable": True,
    },
    {
        "id": "Q05",
        "domain": "document",
        "rag_mode": "documents",
        "query": "Under what conditions is automated DNS failover between primary and secondary regions triggered?",
        "expected_source": "cloud_deployment_guide.pdf",
        "expected_page_line": "Page 3",
        "key_facts": ["3 consecutive health checks", "90-second sliding evaluation window"],
        "is_answerable": True,
    },
    {
        "id": "Q06",
        "domain": "document",
        "rag_mode": "documents",
        "query": "What are the production and staging base URLs specified for the API?",
        "expected_source": "api_specifications.md",
        "expected_page_line": "Lines 10-25",
        "key_facts": ["https://api.internal.enterprise.net/v1", "https://api-staging.internal.enterprise.net/v1"],
        "is_answerable": True,
    },
    {
        "id": "Q07",
        "domain": "document",
        "rag_mode": "documents",
        "query": "What are the rate limit quota and burst allowance per client organization?",
        "expected_source": "api_specifications.md",
        "expected_page_line": "Lines 25-45",
        "key_facts": ["1,200 requests/minute", "burst bucket", "150 requests"],
        "is_answerable": True,
    },
    {
        "id": "Q08",
        "domain": "document",
        "rag_mode": "documents",
        "query": "What is the maximum permissible request payload size and correlation header for API requests?",
        "expected_source": "api_specifications.md",
        "expected_page_line": "Lines 40-60",
        "key_facts": ["15 MB", "15728640", "X-Request-Correlation-ID"],
        "is_answerable": True,
    },
    {
        "id": "Q09",
        "domain": "document",
        "rag_mode": "documents",
        "query": "What are the symptoms and error code for database connection pool starvation?",
        "expected_source": "incident_runbook.txt",
        "expected_page_line": "Lines 5-20",
        "key_facts": ["ERR-503-POOL", "503 Service Unavailable", "ConnectionPoolExhaustedException"],
        "is_answerable": True,
    },
    {
        "id": "Q10",
        "domain": "document",
        "rag_mode": "documents",
        "query": "What remediation commands are prescribed in Step 1 and Step 2 of the connection pool exhaustion runbook?",
        "expected_source": "incident_runbook.txt",
        "expected_page_line": "Lines 20-40",
        "key_facts": ["poolctl stat --cluster db-primary-01", "poolctl resize --cluster db-primary-01 --max-connections 250"],
        "is_answerable": True,
    },

    # 2. Codebase RAG Original (Q11-Q20)
    {
        "id": "Q11",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "What HTTP endpoints and routes are defined in app.py?",
        "expected_source": "app.py",
        "expected_page_line": "Lines 120-200",
        "key_facts": ["/api/chat", "/api/settings", "/api/models", "/api/health"],
        "is_answerable": True,
    },
    {
        "id": "Q12",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "How does classify_intent autonomously determine whether to route a prompt to gemini versus local models?",
        "expected_source": "app.py",
        "expected_page_line": "Lines 205-245",
        "key_facts": ["def classify_intent", "gemini", "pipeline"],
        "is_answerable": True,
    },
    {
        "id": "Q13",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "What model names and priorities are defined in CODER_MODELS in app.py?",
        "expected_source": "app.py",
        "expected_page_line": "Lines 75-95",
        "key_facts": ["CODER_MODELS", "deepseek-coder-v2:16b", "Qwen2.5-Coder:7B"],
        "is_answerable": True,
    },
    {
        "id": "Q14",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "How does streamBotPipeline in templates/index.html process SSE events like route and done?",
        "expected_source": "templates/index.html",
        "expected_page_line": "Lines 2400-2650",
        "key_facts": ["streamBotPipeline", "route", "token", "done"],
        "is_answerable": True,
    },
    {
        "id": "Q15",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "What default ignore patterns and file size limits are enforced in rag/config.py?",
        "expected_source": "rag/config.py",
        "expected_page_line": "Lines 25-65",
        "key_facts": ["HARDCODED_SECURITY_BLACKLIST", "MAX_FILE_SIZE_BYTES", "10 * 1024 * 1024"],
        "is_answerable": True,
    },
    {
        "id": "Q16",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "How does IncrementalIndexer calculate file hashes and determine whether a file requires re-indexing?",
        "expected_source": "rag/indexer.py",
        "expected_page_line": "Lines 70-130",
        "key_facts": ["hashlib.sha256", "file_hash", "compute_file_hash"],
        "is_answerable": True,
    },
    {
        "id": "Q17",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "How does versioned chunk replacement in _safe_replace_file_entry stage new generations before committing?",
        "expected_source": "rag/indexer.py",
        "expected_page_line": "Lines 180-260",
        "key_facts": ["uuid.uuid4().hex[:8]", "generation_id", "_safe_replace_file_entry"],
        "is_answerable": True,
    },
    {
        "id": "Q18",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "How does UnifiedRetriever enforce active generation filtering using the committed manifest to prevent stale chunks?",
        "expected_source": "rag/retriever.py",
        "expected_page_line": "Lines 110-185",
        "key_facts": ["manifest", "generation_id", "active_entry"],
        "is_answerable": True,
    },
    {
        "id": "Q19",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "How does PromptBuilder format retrieved context chunks and guard against prompt injection?",
        "expected_source": "rag/prompt_builder.py",
        "expected_page_line": "Lines 20-80",
        "key_facts": ["<retrieved_context>", "build_rag_prompt", "untrusted"],
        "is_answerable": True,
    },
    {
        "id": "Q20",
        "domain": "codebase",
        "rag_mode": "codebase",
        "query": "How does RAGService retrieve context based on the requested rag_mode?",
        "expected_source": "rag/service.py",
        "expected_page_line": "Lines 45-120",
        "key_facts": ["retrieve_context", "rag_mode", "auto", "documents", "codebase", "both"],
        "is_answerable": True,
    },

    # 3. Combined RAG (Q21-Q25)
    {
        "id": "Q21",
        "domain": "combined",
        "rag_mode": "both",
        "query": "Compare the X-Request-Correlation-ID header in api_specifications.md with request handling in app.py.",
        "expected_source": "api_specifications.md AND app.py",
        "expected_page_line": "Multi-source",
        "key_facts": ["X-Request-Correlation-ID", "request", "app.py"],
        "is_answerable": True,
    },
    {
        "id": "Q22",
        "domain": "combined",
        "rag_mode": "both",
        "query": "Compare the timeout in incident_runbook.txt with OLLAMA_TIMEOUT in rag/config.py and app.py.",
        "expected_source": "incident_runbook.txt AND rag/config.py",
        "expected_page_line": "Multi-source",
        "key_facts": ["timeout", "OLLAMA_TIMEOUT", "60", "seconds"],
        "is_answerable": True,
    },
    {
        "id": "Q23",
        "domain": "combined",
        "rag_mode": "both",
        "query": "Compare the maximum API request payload size in api_specifications.md with MAX_FILE_SIZE_BYTES in rag/config.py.",
        "expected_source": "api_specifications.md AND rag/config.py",
        "expected_page_line": "Multi-source",
        "key_facts": ["15 MB", "10 * 1024 * 1024", "MAX_FILE_SIZE_BYTES"],
        "is_answerable": True,
    },
    {
        "id": "Q24",
        "domain": "combined",
        "rag_mode": "both",
        "query": "Compare the multi-region topology in cloud_deployment_guide.pdf with local deployment in run_app.py and app.py.",
        "expected_source": "cloud_deployment_guide.pdf AND run_app.py",
        "expected_page_line": "Multi-source",
        "key_facts": ["multi-region", "localhost", "127.0.0.1", "5000"],
        "is_answerable": True,
    },
    {
        "id": "Q25",
        "domain": "combined",
        "rag_mode": "both",
        "query": "Compare database connection pooling in incident_runbook.txt with ChromaDB storage in rag/vector_store.py.",
        "expected_source": "incident_runbook.txt AND rag/vector_store.py",
        "expected_page_line": "Multi-source",
        "key_facts": ["poolctl", "max-connections", "PersistentClient", "VectorStoreManager"],
        "is_answerable": True,
    },

    # 4. Absent / Unanswerable (Q26-Q30)
    {
        "id": "Q26",
        "domain": "absent",
        "rag_mode": "auto",
        "query": "What is the Kubernetes cluster autoscaler scale-down delay threshold configured for worker nodes?",
        "expected_source": "None (Absent)",
        "expected_page_line": "N/A",
        "key_facts": ["kubernetes", "autoscaler", "scale-down"],
        "is_answerable": False,
    },
    {
        "id": "Q27",
        "domain": "absent",
        "rag_mode": "auto",
        "query": "What is the Redis cache eviction policy configured in the repository?",
        "expected_source": "None (Absent)",
        "expected_page_line": "N/A",
        "key_facts": ["redis", "eviction", "allkeys-lru"],
        "is_answerable": False,
    },
    {
        "id": "Q28",
        "domain": "absent",
        "rag_mode": "auto",
        "query": "What is the OAuth2 client secret and redirect URI for Google Drive cloud synchronization?",
        "expected_source": "None (Absent)",
        "expected_page_line": "N/A",
        "key_facts": ["oauth2", "google drive", "client secret"],
        "is_answerable": False,
    },
    {
        "id": "Q29",
        "domain": "absent",
        "rag_mode": "auto",
        "query": "How is the Stripe payment webhook HMAC signature verified in the payment gateway module?",
        "expected_source": "None (Absent)",
        "expected_page_line": "N/A",
        "key_facts": ["stripe", "webhook", "hmac"],
        "is_answerable": False,
    },
    {
        "id": "Q30",
        "domain": "absent",
        "rag_mode": "auto",
        "query": "What GraphQL mutation is used to update user billing addresses and tax identification numbers?",
        "expected_source": "None (Absent)",
        "expected_page_line": "N/A",
        "key_facts": ["graphql", "mutation", "billing"],
        "is_answerable": False,
    },

    # ── 20 New Specific Queries (Q31-Q50) ────────────────────────────────────
    # 5. Implementation-Specific Queries (Q31-Q35)
    {
        "id": "Q31",
        "domain": "implementation",
        "rag_mode": "codebase",
        "query": "How does VectorStoreManager initialize persistent client and isolated collections in rag/vector_store.py?",
        "expected_source": "rag/vector_store.py",
        "expected_page_line": "Lines 35-90",
        "key_facts": ["PersistentClient", "get_or_create_collection", "documents_collection", "codebase_collection"],
        "is_answerable": True,
    },
    {
        "id": "Q32",
        "domain": "implementation",
        "rag_mode": "codebase",
        "query": "What parameters and endpoints does EmbeddingService pass to Ollama in rag/embeddings.py?",
        "expected_source": "rag/embeddings.py",
        "expected_page_line": "Lines 50-110",
        "key_facts": ["/api/embed", "model", "input", "embeddinggemma", "EMBEDDING_BATCH_SIZE"],
        "is_answerable": True,
    },
    {
        "id": "Q33",
        "domain": "implementation",
        "rag_mode": "codebase",
        "query": "How does IncrementalIndexer roll back staged chunk IDs if manifest persistence fails in rag/indexer.py?",
        "expected_source": "rag/indexer.py",
        "expected_page_line": "Lines 265-285",
        "key_facts": ["delete_by_ids", "rollback", "cat_manifest", "prev_entry"],
        "is_answerable": True,
    },
    {
        "id": "Q34",
        "domain": "implementation",
        "rag_mode": "codebase",
        "query": "How does PromptBuilder construct system directives and wrap context chunks in rag/prompt_builder.py?",
        "expected_source": "rag/prompt_builder.py",
        "expected_page_line": "Lines 30-75",
        "key_facts": ["<retrieved_context>", "build_rag_prompt", "system", "untrusted"],
        "is_answerable": True,
    },
    {
        "id": "Q35",
        "domain": "implementation",
        "rag_mode": "codebase",
        "query": "How does RAGService manage singleton lifecycle and handle off or empty retrieval modes in rag/service.py?",
        "expected_source": "rag/service.py",
        "expected_page_line": "Lines 40-85",
        "key_facts": ["get_rag_service", "_rag_service_instance", "rag_mode", "off"],
        "is_answerable": True,
    },

    # 6. Test-Specific Queries (Q36-Q40)
    {
        "id": "Q36",
        "domain": "test",
        "rag_mode": "codebase",
        "query": "What unit test methods verify active generation filtering in tests_rag_phase4_2.py?",
        "expected_source": "tests_rag_phase4_2.py",
        "expected_page_line": "Lines 40-100",
        "key_facts": ["test_active_generation_filtering", "test_orphan_cleanup", "generation_id"],
        "is_answerable": True,
    },
    {
        "id": "Q37",
        "domain": "test",
        "rag_mode": "codebase",
        "query": "How does tests_rag_phase3.py test unreadable or corrupted manifest recovery?",
        "expected_source": "tests_rag_phase3.py",
        "expected_page_line": "Lines 75-135",
        "key_facts": ["test_corrupted_manifest", "test_unreadable_manifest", "bak"],
        "is_answerable": True,
    },
    {
        "id": "Q38",
        "domain": "test",
        "rag_mode": "codebase",
        "query": "What assertions verify failure injection rollback during vector insertion in tests_rag_phase4_1.py?",
        "expected_source": "tests_rag_phase4_1.py",
        "expected_page_line": "Lines 60-120",
        "key_facts": ["test_rollback_on_insertion_failure", "delete_by_ids", "assertRaises"],
        "is_answerable": True,
    },
    {
        "id": "Q39",
        "domain": "test",
        "rag_mode": "codebase",
        "query": "How does tests_rag_phase5.py test simulated Ollama downtime and embedding service failure?",
        "expected_source": "tests_rag_phase5.py",
        "expected_page_line": "Lines 50-100",
        "key_facts": ["EMBEDDING_UNAVAILABLE", "Ollama server unavailable", "mock"],
        "is_answerable": True,
    },
    {
        "id": "Q40",
        "domain": "test",
        "rag_mode": "codebase",
        "query": "What unit tests verify the frontend RAG mode selector and SSE citation handling in tests_rag_phase6.py?",
        "expected_source": "tests_rag_phase6.py",
        "expected_page_line": "Lines 40-95",
        "key_facts": ["test_frontend_rag_mode_selector", "rag_mode", "citation", "templates/index.html"],
        "is_answerable": True,
    },

    # 7. Frontend-Specific Queries (Q41-Q45)
    {
        "id": "Q41",
        "domain": "frontend",
        "rag_mode": "codebase",
        "query": "Where is the rag-mode-select element defined in templates/index.html?",
        "expected_source": "templates/index.html",
        "expected_page_line": "Lines 160-220",
        "key_facts": ["rag-mode-select", "rag_mode", "Auto", "Documents", "Codebase", "Both", "Off"],
        "is_answerable": True,
    },
    {
        "id": "Q42",
        "domain": "frontend",
        "rag_mode": "codebase",
        "query": "How is the pause and resume streaming control button implemented in templates/index.html?",
        "expected_source": "templates/index.html",
        "expected_page_line": "Lines 2200-2450",
        "key_facts": ["btn-pause-stream", "pauseStream", "resumeStream", "isPaused"],
        "is_answerable": True,
    },
    {
        "id": "Q43",
        "domain": "frontend",
        "rag_mode": "codebase",
        "query": "How does streamBotPipeline handle the citation SSE event and render source pills in templates/index.html?",
        "expected_source": "templates/index.html",
        "expected_page_line": "Lines 2500-2750",
        "key_facts": ["event === 'citation'", "citation-pill", "renderCitations", "source_path"],
        "is_answerable": True,
    },
    {
        "id": "Q44",
        "domain": "frontend",
        "rag_mode": "codebase",
        "query": "Where are the CSS variables for dark theme and light theme defined in templates/index.html?",
        "expected_source": "templates/index.html",
        "expected_page_line": "Lines 30-150",
        "key_facts": [":root", "[data-theme=\"light\"]", "--bg-primary", "--accent-primary"],
        "is_answerable": True,
    },
    {
        "id": "Q45",
        "domain": "frontend",
        "rag_mode": "codebase",
        "query": "How does the model dropdown populate available models dynamically in templates/index.html?",
        "expected_source": "templates/index.html",
        "expected_page_line": "Lines 2100-2300",
        "key_facts": ["loadAvailableModels", "/api/models", "model-select", "fetch"],
        "is_answerable": True,
    },

    # 8. Configuration-Specific Queries (Q46-Q50)
    {
        "id": "Q46",
        "domain": "config",
        "rag_mode": "codebase",
        "query": "What is the default SIMILARITY_DISTANCE_THRESHOLD in rag/config.py?",
        "expected_source": "rag/config.py",
        "expected_page_line": "Lines 35-50",
        "key_facts": ["SIMILARITY_DISTANCE_THRESHOLD", "0.70", "0.7"],
        "is_answerable": True,
    },
    {
        "id": "Q47",
        "domain": "config",
        "rag_mode": "codebase",
        "query": "What file extensions are supported in SUPPORTED_CODE_EXTENSIONS in rag/config.py?",
        "expected_source": "rag/config.py",
        "expected_page_line": "Lines 55-75",
        "key_facts": [".py", ".json", ".yaml", ".html", ".htm", ".css", ".js", "SUPPORTED_CODE_EXTENSIONS"],
        "is_answerable": True,
    },
    {
        "id": "Q48",
        "domain": "config",
        "rag_mode": "codebase",
        "query": "What security patterns are hardcoded in HARDCODED_SECURITY_BLACKLIST in rag/config.py?",
        "expected_source": "rag/config.py",
        "expected_page_line": "Lines 100-135",
        "key_facts": [".env", ".git", "server.log", "HARDCODED_SECURITY_BLACKLIST", "HARDCODED_EXT_BLACKLIST"],
        "is_answerable": True,
    },
    {
        "id": "Q49",
        "domain": "config",
        "rag_mode": "codebase",
        "query": "What are the default chunk size and overlap configurations in DOC_CHUNK_SIZE and DOC_CHUNK_OVERLAP in rag/config.py?",
        "expected_source": "rag/config.py",
        "expected_page_line": "Lines 30-40",
        "key_facts": ["DOC_CHUNK_SIZE", "750", "DOC_CHUNK_OVERLAP", "120"],
        "is_answerable": True,
    },
    {
        "id": "Q50",
        "domain": "config",
        "rag_mode": "codebase",
        "query": "How does is_path_ignored in rag/config.py evaluate relative paths against .ragignore rules?",
        "expected_source": "rag/config.py",
        "expected_page_line": "Lines 145-200",
        "key_facts": ["fnmatch", "is_path_ignored", "load_ragignore_patterns", "HARDCODED_SECURITY_BLACKLIST"],
        "is_answerable": True,
    },
]


def evaluate_query(
    retriever: UnifiedRetriever,
    query_item: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a single query and evaluate separated metrics."""
    qid = query_item["id"]
    domain = query_item["domain"]
    rag_mode = query_item["rag_mode"]
    query = query_item["query"]
    expected_src = query_item["expected_source"]
    expected_loc = query_item["expected_page_line"]
    key_facts = query_item["key_facts"]
    is_answerable = query_item["is_answerable"]

    t0 = time.perf_counter()
    resp = retriever.retrieve(query=query, mode=rag_mode, top_k=4)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    chunks = resp.chunks
    top_chunk: Optional[RetrievedChunk] = chunks[0] if chunks else None
    top_file = top_chunk.file_name if top_chunk else "None"
    top_sim = top_chunk.similarity if top_chunk else 0.0
    top_score = top_chunk.score if top_chunk else top_sim

    retrieved_files = [c.file_name for c in chunks]
    retrieved_sources = []
    for c in chunks:
        loc = f"p. {c.page_number}" if c.page_number else (f"L{c.start_line}-{c.end_line}" if c.start_line else "")
        retrieved_sources.append(f"{c.file_name} ({loc}, sim={c.similarity:.3f}, score={c.score or c.similarity:.3f})")

    # Metric evaluation
    retrieval_success = False
    precision_at_k = 0.0
    recall_at_k = 0.0
    citation_accurate = False
    answer_grounded = False
    abstention_correct = False
    failure_reason = ""

    if not is_answerable:
        # Unanswerable query evaluation
        if len(chunks) == 0:
            abstention_correct = True
            failure_reason = "Clean abstention (0 chunks returned)"
        else:
            # If chunks returned, verify if similarity is appropriately low (< 0.45)
            if top_sim < 0.45:
                abstention_correct = True
                failure_reason = f"Low confidence abstention (top_sim={top_sim:.3f})"
            else:
                abstention_correct = False
                failure_reason = f"Spurious retrieval on absent topic (top_sim={top_sim:.3f})"
    else:
        # Answerable query evaluation
        if len(chunks) == 0:
            retrieval_success = False
            precision_at_k = 0.0
            recall_at_k = 0.0
            citation_accurate = False
            answer_grounded = False
            failure_reason = "Starvation: 0 chunks returned for answerable query"
        else:
            if domain in ("document", "codebase", "implementation", "test", "frontend", "config"):
                target_base = os.path.basename(expected_src)
                # Check top chunk or top K
                matching_chunks = [c for c in chunks if target_base in (c.file_name or "")]
                if matching_chunks:
                    retrieval_success = True
                    precision_at_k = len(matching_chunks) / len(chunks)
                    recall_at_k = 1.0  # target file found
                    citation_accurate = True
                    # Check text grounding
                    combined_text = " ".join(c.content for c in chunks).lower()
                    if any(fact.lower() in combined_text for fact in key_facts):
                        answer_grounded = True
                    else:
                        answer_grounded = False
                        failure_reason = "Key fact missing in chunk snippets"
                else:
                    retrieval_success = False
                    precision_at_k = 0.0
                    recall_at_k = 0.0
                    citation_accurate = False
                    answer_grounded = False
                    failure_reason = f"Expected '{target_base}' not in top {len(chunks)} chunks"

            elif domain == "combined":
                # Expecting multi-source (e.g. document and code)
                doc_matches = [c for c in chunks if (c.file_name or "").endswith((".pdf", ".md", ".txt"))]
                code_matches = [c for c in chunks if (c.file_name or "").endswith((".py", ".html", ".json"))]
                if doc_matches and code_matches:
                    retrieval_success = True
                    precision_at_k = (len(doc_matches) + len(code_matches)) / len(chunks)
                    recall_at_k = 1.0
                    citation_accurate = True
                    answer_grounded = True
                elif doc_matches or code_matches:
                    retrieval_success = True  # partial hit
                    precision_at_k = max(len(doc_matches), len(code_matches)) / len(chunks)
                    recall_at_k = 0.5
                    citation_accurate = False
                    answer_grounded = False
                    failure_reason = "Only one half of cross-modal sources retrieved"
                else:
                    retrieval_success = False
                    precision_at_k = 0.0
                    recall_at_k = 0.0
                    citation_accurate = False
                    answer_grounded = False
                    failure_reason = "Neither domain source retrieved"

    return {
        "id": qid,
        "domain": domain,
        "rag_mode": rag_mode,
        "query": query,
        "is_answerable": is_answerable,
        "expected_source": expected_src,
        "expected_page_line": expected_loc,
        "top_file": top_file,
        "top_sim": round(top_sim, 4),
        "top_score": round(top_score, 4),
        "chunks_count": len(chunks),
        "retrieved_sources": retrieved_sources,
        "latency_ms": round(elapsed_ms, 2),
        "retrieval_success": retrieval_success,
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "citation_accurate": citation_accurate,
        "answer_grounded": answer_grounded,
        "abstention_correct": abstention_correct,
        "failure_reason": failure_reason,
    }


def run_latency_benchmarks(
    embedding_service: EmbeddingService,
    vector_store: VectorStoreManager,
    retriever: UnifiedRetriever,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("RUNNING LATENCY, THROUGHPUT & DECOMPOSITION BENCHMARKS")
    print("=" * 70)

    sample_query = "What is the maximum allowable network latency between availability zones?"

    # 1. Cold vs Warm Embedding Latency
    # For cold simulation, embed a distinctive new string
    cold_str = f"Cold startup embedding benchmark unique string {time.time()}"
    t0 = time.perf_counter()
    _ = embedding_service.embed_query(cold_str)
    cold_embed_ms = (time.perf_counter() - t0) * 1000

    warm_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = embedding_service.embed_query(sample_query)
        warm_times.append((time.perf_counter() - t0) * 1000)
    warm_embed_ms = statistics.mean(warm_times)

    # 2. Batch Embedding Throughput (batch sizes 8, 16, 32)
    batch_results = {}
    for b_size in [8, 16, 32]:
        texts = [f"Text item index {i} in benchmark batch of size {b_size} testing throughput." for i in range(b_size)]
        t0 = time.perf_counter()
        _ = embedding_service.embed_batch(texts, batch_size=b_size)
        elapsed = (time.perf_counter() - t0) * 1000
        per_chunk = elapsed / b_size
        batch_results[f"batch_{b_size}"] = {
            "total_ms": round(elapsed, 2),
            "per_chunk_ms": round(per_chunk, 2),
            "chunks_per_sec": round(1000.0 / per_chunk, 2),
        }

    # 3. Retrieval Latency Decomposition: Total Retrieval vs Pure ChromaDB Query (without embedding)
    query_vec = embedding_service.embed_query(sample_query)

    chroma_doc_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = vector_store.query("documents_collection", query_embeddings=query_vec, n_results=4)
        chroma_doc_times.append((time.perf_counter() - t0) * 1000)
    pure_chroma_doc_ms = statistics.mean(chroma_doc_times)

    chroma_code_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = vector_store.query("codebase_collection", query_embeddings=query_vec, n_results=4)
        chroma_code_times.append((time.perf_counter() - t0) * 1000)
    pure_chroma_code_ms = statistics.mean(chroma_code_times)

    total_retrieval_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = retriever.retrieve(sample_query, mode="both", top_k=4)
        total_retrieval_times.append((time.perf_counter() - t0) * 1000)
    total_retrieval_ms = statistics.mean(total_retrieval_times)

    # 4. Memory and Disk Footprints
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)

    chroma_size_bytes = 0
    if os.path.exists(CHROMA_DIR):
        for root, _, files in os.walk(CHROMA_DIR):
            for f in files:
                chroma_size_bytes += os.path.getsize(os.path.join(root, f))
    chroma_mb = chroma_size_bytes / (1024 * 1024)

    manifest_size_kb = 0.0
    if os.path.exists(INDEX_MANIFEST_PATH):
        manifest_size_kb = os.path.getsize(INDEX_MANIFEST_PATH) / 1024

    return {
        "cold_embed_ms": round(cold_embed_ms, 2),
        "warm_embed_ms": round(warm_embed_ms, 2),
        "batch_benchmarks": batch_results,
        "pure_chroma_doc_ms": round(pure_chroma_doc_ms, 2),
        "pure_chroma_code_ms": round(pure_chroma_code_ms, 2),
        "total_retrieval_ms": round(total_retrieval_ms, 2),
        "embedding_time_fraction": round(warm_embed_ms / max(1.0, total_retrieval_ms) * 100, 1),
        "chroma_time_fraction": round((pure_chroma_doc_ms + pure_chroma_code_ms) / max(1.0, total_retrieval_ms) * 100, 1),
        "memory": {
            "rss_mb": round(rss_mb, 2),
            "vms_mb": round(vms_mb, 2),
            "chroma_dir_mb": round(chroma_mb, 2),
            "manifest_kb": round(manifest_size_kb, 2),
        },
    }


def main():
    print("=" * 70)
    print("PHASE 7.1 COMPREHENSIVE RETRIEVAL QUALITY & BENCHMARK SUITE")
    print("=" * 70)

    # 1. Initialize components
    vector_store = VectorStoreManager()
    vector_store.initialize()
    embedding_service = EmbeddingService()

    # Baseline retriever (source-aware ranking DISABLED)
    baseline_config = dict(DEFAULT_RANKING_CONFIG)
    baseline_config["enabled"] = False
    baseline_retriever = UnifiedRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        ranking_config=baseline_config,
    )

    # Phase 7.1 Refined retriever (source-aware ranking ENABLED)
    refined_retriever = UnifiedRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        ranking_config=DEFAULT_RANKING_CONFIG,
    )

    # 2. Evaluate all 50 queries on Baseline
    print("\n[1/3] Evaluating Baseline (Unranked / Distance-only)...")
    baseline_results = []
    for item in EVALUATION_QUERIES:
        res = evaluate_query(baseline_retriever, item)
        baseline_results.append(res)

    # 3. Evaluate all 50 queries on Refined Retriever
    print("\n[2/3] Evaluating Phase 7.1 Refined (Source-Aware Ranking + Templates)...")
    refined_results = []
    for item in EVALUATION_QUERIES:
        res = evaluate_query(refined_retriever, item)
        refined_results.append(res)

    # 4. Run Performance & Latency Benchmarks
    print("\n[3/3] Running Latency & Throughput Benchmarks...")
    perf_benchmarks = run_latency_benchmarks(embedding_service, vector_store, refined_retriever)

    # 5. Compute Aggregate Separated Metrics
    def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        answerable = [r for r in results if r["is_answerable"]]
        unanswerable = [r for r in results if not r["is_answerable"]]

        total_ans = len(answerable)
        ans_success = sum(1 for r in answerable if r["retrieval_success"])
        mean_p_at_k = statistics.mean([r["precision_at_k"] for r in answerable]) if answerable else 0.0
        mean_r_at_k = statistics.mean([r["recall_at_k"] for r in answerable]) if answerable else 0.0
        citation_acc = sum(1 for r in answerable if r["citation_accurate"]) / total_ans if total_ans else 0.0
        groundedness = sum(1 for r in answerable if r["answer_grounded"]) / total_ans if total_ans else 0.0

        total_unans = len(unanswerable)
        correct_abstentions = sum(1 for r in unanswerable if r["abstention_correct"])
        clean_abstentions = sum(1 for r in unanswerable if r["chunks_count"] == 0)
        abstention_rate = correct_abstentions / total_unans if total_unans else 0.0

        # Sub-domain breakdowns
        domains = set(r["domain"] for r in results)
        domain_breakdown = {}
        for d in sorted(domains):
            d_records = [r for r in results if r["domain"] == d]
            d_ans = [r for r in d_records if r["is_answerable"]]
            d_unans = [r for r in d_records if not r["is_answerable"]]

            d_entry = {"total": len(d_records)}
            if d_ans:
                d_entry["retrieval_success_rate"] = round(sum(1 for r in d_ans if r["retrieval_success"]) / len(d_ans), 4)
                d_entry["precision_at_k"] = round(statistics.mean([r["precision_at_k"] for r in d_ans]), 4)
                d_entry["recall_at_k"] = round(statistics.mean([r["recall_at_k"] for r in d_ans]), 4)
                d_entry["citation_accuracy"] = round(sum(1 for r in d_ans if r["citation_accurate"]) / len(d_ans), 4)
                d_entry["groundedness"] = round(sum(1 for r in d_ans if r["answer_grounded"]) / len(d_ans), 4)
            if d_unans:
                d_entry["abstention_rate"] = round(sum(1 for r in d_unans if r["abstention_correct"]) / len(d_unans), 4)
                d_entry["clean_abstention_rate"] = round(sum(1 for r in d_unans if r["chunks_count"] == 0) / len(d_unans), 4)

            domain_breakdown[d] = d_entry

        return {
            "answerable_queries_total": total_ans,
            "retrieval_success_count": ans_success,
            "retrieval_success_rate": round(ans_success / total_ans, 4) if total_ans else 0.0,
            "mean_precision_at_k": round(mean_p_at_k, 4),
            "mean_recall_at_k": round(mean_r_at_k, 4),
            "citation_accuracy_rate": round(citation_acc, 4),
            "answer_groundedness_rate": round(groundedness, 4),
            "unanswerable_queries_total": total_unans,
            "correct_abstention_count": correct_abstentions,
            "clean_abstention_count": clean_abstentions,
            "correct_abstention_rate": round(abstention_rate, 4),
            "domain_breakdown": domain_breakdown,
        }

    baseline_metrics = compute_metrics(baseline_results)
    refined_metrics = compute_metrics(refined_results)

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(EVALUATION_QUERIES),
        "baseline_metrics": baseline_metrics,
        "refined_metrics": refined_metrics,
        "performance_benchmarks": perf_benchmarks,
        "baseline_detailed_results": baseline_results,
        "refined_detailed_results": refined_results,
    }

    out_file = os.path.join(os.path.dirname(__file__), "phase7_1_evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("PHASE 7.1 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Queries: {len(EVALUATION_QUERIES)}")
    print(f"Answerable Queries: {baseline_metrics['answerable_queries_total']}")
    print(f"  - Baseline Retrieval Success: {baseline_metrics['retrieval_success_count']}/{baseline_metrics['answerable_queries_total']} ({baseline_metrics['retrieval_success_rate']*100:.1f}%)")
    print(f"  - Refined  Retrieval Success: {refined_metrics['retrieval_success_count']}/{refined_metrics['answerable_queries_total']} ({refined_metrics['retrieval_success_rate']*100:.1f}%)")
    print(f"  - Baseline Precision@K:       {baseline_metrics['mean_precision_at_k']:.3f} | Refined: {refined_metrics['mean_precision_at_k']:.3f}")
    print(f"  - Baseline Recall@K:          {baseline_metrics['mean_recall_at_k']:.3f} | Refined: {refined_metrics['mean_recall_at_k']:.3f}")
    print(f"  - Baseline Groundedness:      {baseline_metrics['answer_groundedness_rate']*100:.1f}% | Refined: {refined_metrics['answer_groundedness_rate']*100:.1f}%")
    print(f"Unanswerable Queries: {baseline_metrics['unanswerable_queries_total']}")
    print(f"  - Correct Abstention Rate:    {refined_metrics['correct_abstention_rate']*100:.1f}% ({refined_metrics['correct_abstention_count']}/{refined_metrics['unanswerable_queries_total']})")
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
