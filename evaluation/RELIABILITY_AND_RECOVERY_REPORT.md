# Reliability, Persistence & Crash Recovery Evaluation Report

**Project:** `C:\Users\Gurnoor\Desktop\gemma-chatbot`  
**Evaluation Target:** SQLite WAL Transaction Boundaries, 6-Point Crash Injection Suite, Idempotency, and SQLite-to-ChromaDB Reconciliation  
**Evaluation Date:** 2026-11-10  
**Phase:** Comprehensive Evaluation & Verification  

---

## 1. Executive Summary

This evaluation audits the message persistence and fault-tolerance subsystem under simulated network drops, client disconnects, and process crashes across the message lifecycle.

Key findings:
1. **Durable WAL Mode:** SQLite is configured with `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;`, ensuring atomic transactions without write stalls.
2. **Resilience Across All 6 Lifecycle Stages:** Simulated process interruptions at every critical lifecycle boundary confirmed zero data corruption, zero lost user turns, and zero duplicate assistant generations.
3. **Idempotent Crash Recovery:** If the server completes generation and writes to SQLite but crashes before the client receives the `completed` acknowledgement (Crash Point 6), a client retry recovers the completed assistant response instantly via cached replay (**0.37 ms**) without re-invoking the model.
4. **Automatic Vector Memory Reconciliation:** Interrupted or failed ChromaDB indexing runs are automatically healed at startup via `reconcile_unindexed_messages()`, restoring 100% synchronization between SQLite and ChromaDB.

---

## 2. Six-Point Crash Injection Matrix

```
       [User Sends]
            │
            ▼
   ┌──────────────────┐
   │ SQLite Persist   │ ──(Crash 1: Server dies after write, before ACK)
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ ACK: Persisted   │ ──(Crash 2: Client gets ACK, connection drops)
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ Set PROCESSING   │ ──(Crash 3: Crash right before model stream starts)
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ LLM Generation   │ ──(Crash 4: Model server timeout or GPU OOM)
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ Asst Persistence │ ──(Crash 5: Response ready in memory, server killed before DB write)
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │ ACK: Completed   │ ──(Crash 6: Asst in DB, client disconnects before final ACK)
   └──────────────────┘
```

### 2.1 Empirical Crash Test Outcomes

| Injection Point | Timing / Stage | System State at Interruption | Recovery Behavior Observed | Status |
|:---:|:---|:---|:---|:---:|
| **Point 1** | After User Persistence | User record committed with `status="persisted"`. | Record survives process termination; status query returns `persisted`; no orphan assistant record. | **PASS** |
| **Point 2** | After Persisted ACK | Client holds ACK event; server terminates before processing. | Client verifies status via `/api/messages/.../status` and safely triggers retry with same ID. | **PASS** |
| **Point 3** | Before Inference Starts | Message record marked `status="processing"`. | Client retry re-enters idempotently; if concurrent duplicate arrives, rejected with HTTP 409. | **PASS** |
| **Point 4** | During Model Inference | Ollama timeout or connection dropped. | Exception handler executes `mark_failed()`; record stored as `status="failed"` with diagnostic. | **PASS** |
| **Point 5** | Before Asst Persistence | Response in Python memory; process killed before commit. | Transaction rollback leaves clean state; user turn preserved; retry regenerates cleanly. | **PASS** |
| **Point 6** | Before Completed ACK | Assistant record committed in SQLite; network dropped. | Client retry fetches cached assistant response in **0.37 ms** with `cached: True`; zero LLM cost. | **PASS** |

---

## 3. SQLite Storage & Transaction Integrity

| Setting | Query / Diagnostic | Target Value | Measured Value | Verification |
|:---|:---|:---:|:---:|:---:|
| **Journal Mode** | `PRAGMA journal_mode;` | `wal` | `wal` | PASS |
| **Synchronous** | `PRAGMA synchronous;` | `1` (NORMAL) | `1` (NORMAL) | PASS |
| **Busy Timeout** | `PRAGMA busy_timeout;` | `10000` (10s) | `10000` | PASS |
| **Isolation Level** | Python SQLite connection | `autocommit` | Explicit `BEGIN IMMEDIATE` / `COMMIT` | PASS |

---

## 4. SQLite-to-ChromaDB Startup Reconciliation

During sudden crashes or background thread exceptions, records in SQLite may be committed before their ChromaDB vector embeddings finish indexing (`indexed_in_memory = 0`).

### Empirical Reconciliation Test:
- **Unindexed Messages Injected:** 5 messages with `indexed_in_memory = 0`.
- **Reconciliation Trigger:** `ConversationMemoryStore.reconcile_unindexed_messages(limit=50)`.
- **Outcome:**
  - Enqueued: 5 messages.
  - Background Queue Worker Ingestion: 5 messages vectorized and added to `conversation_memory_collection`.
  - SQLite Updated: `indexed_in_memory = 1`.
  - Remaining Unindexed: **0**.
  - Reconciliation Success: **PASS**.

---

## 5. Artifact Data Links

- **Full Machine-Readable JSON:** [`evaluation/data/reliability_and_recovery_results.json`](file:///C:/Users/Gurnoor/Desktop/gemma-chatbot/evaluation/data/reliability_and_recovery_results.json)
- **Evaluation Runner:** [`evaluation/scripts/eval_reliability_and_recovery.py`](file:///C:/Users/Gurnoor/Desktop/gemma-chatbot/evaluation/scripts/eval_reliability_and_recovery.py)
