"""
Evaluation Script: Intent-Driven Conversational History Retrieval Decision Accuracy
Measures heuristic accuracy, precision, recall, F1, FPR, and FNR across 65 labelled queries
spanning Categories 1-5.
"""

import csv
import json
import os
import sys
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.service import get_rag_service

LABELLED_DATASET: List[Dict[str, Any]] = [
    # ── Category 1: Ordinary messages — retrieval should be skipped (13 queries) ──
    {"id": "CAT1_01", "category": "Category 1: Ordinary", "query": "Hey.", "expected_retrieval": False, "reason": "Standard greeting"},
    {"id": "CAT1_02", "category": "Category 1: Ordinary", "query": "How are you doing today?", "expected_retrieval": False, "reason": "Pleasantry / small talk"},
    {"id": "CAT1_03", "category": "Category 1: Ordinary", "query": "Explain Python decorators with a simple example.", "expected_retrieval": False, "reason": "General programming concept"},
    {"id": "CAT1_04", "category": "Category 1: Ordinary", "query": "What is a binary search tree?", "expected_retrieval": False, "reason": "General computer science question"},
    {"id": "CAT1_05", "category": "Category 1: Ordinary", "query": "Write a basic SQL query to select all users older than 30.", "expected_retrieval": False, "reason": "Generic code generation"},
    {"id": "CAT1_06", "category": "Category 1: Ordinary", "query": "What is the difference between TCP and UDP?", "expected_retrieval": False, "reason": "Networking fundamentals"},
    {"id": "CAT1_07", "category": "Category 1: Ordinary", "query": "How does quicksort work?", "expected_retrieval": False, "reason": "Standard algorithm question"},
    {"id": "CAT1_08", "category": "Category 1: Ordinary", "query": "Can you define polymorphism in object-oriented programming?", "expected_retrieval": False, "reason": "General OOP theory"},
    {"id": "CAT1_09", "category": "Category 1: Ordinary", "query": "What is the capital of France?", "expected_retrieval": False, "reason": "General trivia"},
    {"id": "CAT1_10", "category": "Category 1: Ordinary", "query": "Good morning!", "expected_retrieval": False, "reason": "Greeting"},
    {"id": "CAT1_11", "category": "Category 1: Ordinary", "query": "Explain the difference between synchronous and asynchronous execution.", "expected_retrieval": False, "reason": "General software engineering concept"},
    {"id": "CAT1_12", "category": "Category 1: Ordinary", "query": "Write a regular expression to validate an email address.", "expected_retrieval": False, "reason": "Generic coding problem"},
    {"id": "CAT1_13", "category": "Category 1: Ordinary", "query": "What are the SOLID design principles?", "expected_retrieval": False, "reason": "General design principle query"},

    # ── Category 2: Explicit historical references — retrieval required (13 queries) ──
    {"id": "CAT2_01", "category": "Category 2: Explicit History", "query": "What did we decide about model lifecycle?", "expected_retrieval": True, "reason": "Explicit reference to prior decision ('what did we decide')"},
    {"id": "CAT2_02", "category": "Category 2: Explicit History", "query": "Continue our PDF Intelligence implementation we discussed earlier.", "expected_retrieval": True, "reason": "Explicit continuation from prior discussion ('discussed earlier')"},
    {"id": "CAT2_03", "category": "Category 2: Explicit History", "query": "What was the bug DeepSeek identified?", "expected_retrieval": True, "reason": "Explicit agent attribution reference ('what did DeepSeek...')"},
    {"id": "CAT2_04", "category": "Category 2: Explicit History", "query": "Recall our earlier discussion about message persistence.", "expected_retrieval": True, "reason": "Explicit memory retrieval instruction ('earlier discussion')"},
    {"id": "CAT2_05", "category": "Category 2: Explicit History", "query": "Compare this with the architecture we designed previously.", "expected_retrieval": True, "reason": "Explicit historical comparison ('designed previously')"},
    {"id": "CAT2_06", "category": "Category 2: Explicit History", "query": "What did Qwen implement in the earlier session?", "expected_retrieval": True, "reason": "Explicit agent implementation inquiry ('Qwen earlier implementation')"},
    {"id": "CAT2_07", "category": "Category 2: Explicit History", "query": "Summarize our conversation from earlier today.", "expected_retrieval": True, "reason": "Explicit conversation summary request"},
    {"id": "CAT2_08", "category": "Category 2: Explicit History", "query": "What were the results of the phase 7 benchmark we discussed earlier?", "expected_retrieval": True, "reason": "Explicit reference to earlier benchmark discussion"},
    {"id": "CAT2_09", "category": "Category 2: Explicit History", "query": "Can you check what I said earlier regarding the SQLite schema?", "expected_retrieval": True, "reason": "Explicit reference to user's prior statement ('what I said earlier')"},
    {"id": "CAT2_010", "category": "Category 2: Explicit History", "query": "Continue from where we left off on the RAG pipeline.", "expected_retrieval": True, "reason": "Explicit task continuation from prior checkpoint"},
    {"id": "CAT2_11", "category": "Category 2: Explicit History", "query": "What did Gemma suggest during our initial architecture review?", "expected_retrieval": True, "reason": "Explicit agent advice query ('Gemma suggest earlier')"},
    {"id": "CAT2_12", "category": "Category 2: Explicit History", "query": "As we discussed previously, how should we handle orphaned model leases?", "expected_retrieval": True, "reason": "Explicit reference to previous agreement"},
    {"id": "CAT2_13", "category": "Category 2: Explicit History", "query": "Show me the earlier code DeepSeek wrote for lease management.", "expected_retrieval": True, "reason": "Explicit cross-model code query ('earlier code DeepSeek wrote')"},

    # ── Category 3: Implicit historical references — retrieval required (13 queries) ──
    {"id": "CAT3_01", "category": "Category 3: Implicit History", "query": "Why did we choose SQLite instead?", "expected_retrieval": True, "reason": "Implicit reference to an unstated historical design trade-off"},
    {"id": "CAT3_02", "category": "Category 3: Implicit History", "query": "What was causing that memory issue?", "expected_retrieval": True, "reason": "Implicit demonstrative reference ('that memory issue')"},
    {"id": "CAT3_03", "category": "Category 3: Implicit History", "query": "Did we already solve this problem?", "expected_retrieval": True, "reason": "Implicit inquiry into past resolution status"},
    {"id": "CAT3_04", "category": "Category 3: Implicit History", "query": "Continue from where we stopped.", "expected_retrieval": True, "reason": "Implicit workflow continuation"},
    {"id": "CAT3_05", "category": "Category 3: Implicit History", "query": "What happened to the earlier implementation?", "expected_retrieval": True, "reason": "Implicit inquiry on previous artifact status"},
    {"id": "CAT3_06", "category": "Category 3: Implicit History", "query": "Is the latest change compatible with our existing design?", "expected_retrieval": True, "reason": "Implicit comparison with established design decisions"},
    {"id": "CAT3_07", "category": "Category 3: Implicit History", "query": "Why was that timeout occurring?", "expected_retrieval": True, "reason": "Implicit reference to previously debugged timeout"},
    {"id": "CAT3_08", "category": "Category 3: Implicit History", "query": "Did we fix that unclosed database handle?", "expected_retrieval": True, "reason": "Implicit reference to past bug fix"},
    {"id": "CAT3_09", "category": "Category 3: Implicit History", "query": "How does this compare to our first attempt?", "expected_retrieval": True, "reason": "Implicit comparison with initial attempt"},
    {"id": "CAT3_10", "category": "Category 3: Implicit History", "query": "What was the conclusion on that issue?", "expected_retrieval": True, "reason": "Implicit reference to past conclusion"},
    {"id": "CAT3_11", "category": "Category 3: Implicit History", "query": "Why did that test fail earlier?", "expected_retrieval": True, "reason": "Implicit reference to a past test run"},
    {"id": "CAT3_12", "category": "Category 3: Implicit History", "query": "Are we still using that fallback approach?", "expected_retrieval": True, "reason": "Implicit inquiry on persistence of prior design decision"},
    {"id": "CAT3_13", "category": "Category 3: Implicit History", "query": "What went wrong with the initial indexing run?", "expected_retrieval": True, "reason": "Implicit retrospective on previous failure"},

    # ── Category 4: Project-context-dependent requests (13 queries) ──
    # Note: These may need codebase or document RAG, but historical conversational memory is NOT strictly required
    # unless asking about past discussions/decisions. Here we label whether *conversational memory* retrieval is expected.
    {"id": "CAT4_01", "category": "Category 4: Project Context", "query": "How should we improve the chatbot's TTFT?", "expected_retrieval": False, "reason": "Forward-looking technical question (needs code analysis, not past chat memory)"},
    {"id": "CAT4_02", "category": "Category 4: Project Context", "query": "Review the current RAG architecture.", "expected_retrieval": False, "reason": "Codebase/architectural review (needs codebase/docs RAG, not past chat memory)"},
    {"id": "CAT4_03", "category": "Category 4: Project Context", "query": "Is our model-switching approach efficient?", "expected_retrieval": False, "reason": "Technical evaluation query (needs codebase inspection)"},
    {"id": "CAT4_04", "category": "Category 4: Project Context", "query": "What should we implement next in the project?", "expected_retrieval": True, "reason": "Needs project roadmap and previously decided milestones from conversation"},
    {"id": "CAT4_05", "category": "Category 4: Project Context", "query": "Check whether this new feature conflicts with our existing system.", "expected_retrieval": False, "reason": "Codebase compatibility query (needs codebase RAG)"},
    {"id": "CAT4_06", "category": "Category 4: Project Context", "query": "How does our intent classification work in app.py?", "expected_retrieval": False, "reason": "Direct codebase inspection request (codebase RAG)"},
    {"id": "CAT4_07", "category": "Category 4: Project Context", "query": "What is the bottleneck in our current SSE pipeline?", "expected_retrieval": False, "reason": "Performance analysis query (needs benchmark/code analysis)"},
    {"id": "CAT4_08", "category": "Category 4: Project Context", "query": "Can our current database schema handle voice inputs?", "expected_retrieval": False, "reason": "Schema verification query (codebase RAG)"},
    {"id": "CAT4_09", "category": "Category 4: Project Context", "query": "Explain how our model eviction policy works in model_lifecycle.py.", "expected_retrieval": False, "reason": "Codebase query for model_lifecycle.py"},
    {"id": "CAT4_10", "category": "Category 4: Project Context", "query": "Do we have any tests for duplicate message submission?", "expected_retrieval": False, "reason": "Test suite inspection query (codebase RAG)"},
    {"id": "CAT4_11", "category": "Category 4: Project Context", "query": "What is the maximum token limit configured in our prompt builder?", "expected_retrieval": False, "reason": "Configuration lookup in prompt_builder.py (codebase RAG)"},
    {"id": "CAT4_12", "category": "Category 4: Project Context", "query": "How are PDF files processed in our system?", "expected_retrieval": False, "reason": "Architecture explanation (codebase/doc RAG)"},
    {"id": "CAT4_13", "category": "Category 4: Project Context", "query": "What have we completed so far on the project roadmap?", "expected_retrieval": True, "reason": "Historical progress inquiry ('what have we completed so far')"},

    # ── Category 5: Ambiguous messages (13 queries) ──
    # Ambiguous follow-ups: in a sliding window system, recent turns (last 2-3 turns) are already in live prompt.
    # Long-term historical vector retrieval should ONLY trigger if older context is required.
    {"id": "CAT5_01", "category": "Category 5: Ambiguous", "query": "What about the previous one?", "expected_retrieval": False, "reason": "Immediate predecessor pronoun; resolved by live sliding window without vector retrieval"},
    {"id": "CAT5_02", "category": "Category 5: Ambiguous", "query": "Can we improve that?", "expected_retrieval": False, "reason": "Immediate demonstrative; resolved by immediate prior assistant turn"},
    {"id": "CAT5_03", "category": "Category 5: Ambiguous", "query": "Is it still necessary?", "expected_retrieval": False, "reason": "Immediate pronoun; resolved by recent sliding window context"},
    {"id": "CAT5_04", "category": "Category 5: Ambiguous", "query": "What did we do there?", "expected_retrieval": False, "reason": "Immediate spatial/logical reference; resolved by recent turns"},
    {"id": "CAT5_05", "category": "Category 5: Ambiguous", "query": "Why did that happen?", "expected_retrieval": False, "reason": "Immediate error reference; resolved by previous turn's output"},
    {"id": "CAT5_06", "category": "Category 5: Ambiguous", "query": "Can you check that again?", "expected_retrieval": False, "reason": "Immediate repetition request; resolved by recent turns"},
    {"id": "CAT5_07", "category": "Category 5: Ambiguous", "query": "Is there another way to do this?", "expected_retrieval": False, "reason": "Immediate alternative query; resolved by active context"},
    {"id": "CAT5_08", "category": "Category 5: Ambiguous", "query": "Does it work now?", "expected_retrieval": False, "reason": "Immediate verification prompt; resolved by active context"},
    {"id": "CAT5_09", "category": "Category 5: Ambiguous", "query": "What should we do with it?", "expected_retrieval": False, "reason": "Immediate object pronoun; resolved by active context"},
    {"id": "CAT5_10", "category": "Category 5: Ambiguous", "query": "Could that be faster?", "expected_retrieval": False, "reason": "Immediate optimization query; resolved by recent output"},
    {"id": "CAT5_11", "category": "Category 5: Ambiguous", "query": "Will this break anything?", "expected_retrieval": False, "reason": "Immediate impact assessment; resolved by current code in context"},
    {"id": "CAT5_12", "category": "Category 5: Ambiguous", "query": "Can you elaborate on that point?", "expected_retrieval": False, "reason": "Immediate elaboration request; resolved by last assistant message"},
    {"id": "CAT5_13", "category": "Category 5: Ambiguous", "query": "What was the alternative we rejected earlier?", "expected_retrieval": True, "reason": "Ambiguous noun phrase but explicit historical modifier ('rejected earlier') requiring history retrieval"},
]


def run_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("EVALUATION: INTENT-DRIVEN CONVERSATIONAL HISTORY RETRIEVAL ACCURACY")
    print(f"Total Labelled Test Queries: {len(LABELLED_DATASET)}")
    print("=" * 80)

    rag_service = get_rag_service()

    results = []
    category_counts: Dict[str, Dict[str, int]] = {}

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for item in LABELLED_DATASET:
        qid = item["id"]
        cat = item["category"]
        query = item["query"]
        expected = item["expected_retrieval"]

        # Evaluate against actual RAG service implementation
        detected_mode = rag_service.detect_retrieval_mode(query, explicit_mode="auto")
        # Memory retrieval is triggered if detected mode includes memory
        actual_retrieval = detected_mode in ("memory", "all", "memory+codebase", "memory+documents")

        # Classification outcome
        if expected and actual_retrieval:
            outcome = "TP"
            tp += 1
        elif not expected and not actual_retrieval:
            outcome = "TN"
            tn += 1
        elif not expected and actual_retrieval:
            outcome = "FP"
            fp += 1
        else:
            outcome = "FN"
            fn += 1

        is_correct = (expected == actual_retrieval)

        # Track per-category counts
        if cat not in category_counts:
            category_counts[cat] = {"total": 0, "correct": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
        category_counts[cat]["total"] += 1
        if is_correct:
            category_counts[cat]["correct"] += 1
        category_counts[cat][outcome] += 1

        results.append({
            "id": qid,
            "category": cat,
            "query": query,
            "expected_retrieval": expected,
            "actual_retrieval": actual_retrieval,
            "detected_mode": detected_mode,
            "outcome": outcome,
            "is_correct": is_correct,
            "reason": item["reason"],
        })

    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "total_queries": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "category_breakdown": category_counts,
    }

    print("\nOVERALL METRICS:")
    print(f"Accuracy:              {metrics['accuracy'] * 100:.2f}% ({tp + tn}/{total})")
    print(f"Precision:             {metrics['precision'] * 100:.2f}% ({tp}/{tp + fp} positive predictions)")
    print(f"Recall:                {metrics['recall'] * 100:.2f}% ({tp}/{tp + fn} expected retrievals)")
    print(f"F1 Score:              {metrics['f1_score']:.4f}")
    print(f"False Positive Rate:   {metrics['false_positive_rate'] * 100:.2f}% ({fp}/{fp + tn} negatives)")
    print(f"False Negative Rate:   {metrics['false_negative_rate'] * 100:.2f}% ({fn}/{fn + tp} positives)")

    print("\nPER-CATEGORY BREAKDOWN:")
    for cat, cdata in category_counts.items():
        cat_acc = (cdata['correct'] / cdata['total']) * 100
        print(f"  {cat:32s}: {cat_acc:6.2f}% ({cdata['correct']}/{cdata['total']}) | TP={cdata['TP']} TN={cdata['TN']} FP={cdata['FP']} FN={cdata['FN']}")

    # Export to JSON
    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "data")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "retrieval_decision_dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "queries": results}, f, indent=2)

    # Export to CSV
    csv_path = os.path.join(out_dir, "retrieval_decision_dataset.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "query", "expected_retrieval", "actual_retrieval", "detected_mode", "outcome", "is_correct", "reason"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nMachine-readable datasets saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    return {"metrics": metrics, "queries": results}


if __name__ == "__main__":
    run_evaluation()
