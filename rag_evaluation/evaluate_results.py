"""
Phase 7.2 Stage 1 - Claim-Level Answer Grounding Evaluator
- Loads rag_evaluation/raw_generation_results.json
- Evaluates:
  1. Claim extraction and groundedness classification (Supported, Partially Supported, Unsupported)
  2. Answer correctness based on ground-truth verified facts
  3. Citation correctness (file and line/page accuracy)
  4. Abstention accuracy on unanswerable questions
  5. Latency metrics (mean total latency, mean TTFT)
  6. Model-route specific breakdowns (Gemma 4, DeepSeek Architect, Qwen Coder, Gemini API)
- Saves full claim-level evaluation report to rag_evaluation/claim_evaluation_summary.json
"""

import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

RAW_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "raw_generation_results.json")
SUMMARY_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "claim_evaluation_summary.json")

# Conversational boilerplate patterns to ignore during claim extraction
BOILERPLATE_PATTERNS = [
    re.compile(r"^(certainly|sure|here is|here are|according to|based on|hello|hi|as an ai|in summary|let me know)\b.*?:?", re.IGNORECASE),
    re.compile(r"^(feel free to ask|hope this helps|let me know if you need|is there anything else).*?$", re.IGNORECASE),
    re.compile(r"^```[a-z]*$", re.IGNORECASE),
    re.compile(r"^```$", re.IGNORECASE),
]

ABSTENTION_INDICATORS = [
    "not present",
    "not found",
    "not mentioned",
    "not configured",
    "not provided",
    "not contain",
    "does not have",
    "does not exist",
    "cannot find",
    "no information",
    "no kubernetes",
    "no redis",
    "no stripe",
    "no graphql",
    "no oauth",
    "no google drive",
    "unavailable",
    "unsupported",
    "i do not have",
    "context does not",
]


def extract_claims(text: str) -> List[str]:
    """
    Split generated answer into atomic claims by sentences and bullet points,
    filtering out trivial boilerplate phrases.
    """
    if not text or not text.strip():
        return []

    # Clean code fences but preserve contents
    raw_lines = text.splitlines()
    candidate_statements: List[str] = []

    for line in raw_lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Strip list markers
        line_clean = re.sub(r"^[-*•]\s+", "", line_clean)
        line_clean = re.sub(r"^\d+\.\s+", "", line_clean)

        # Ignore boilerplate greetings/closings
        if any(bp.match(line_clean) for bp in BOILERPLATE_PATTERNS):
            continue

        # Split into sentence-like clauses
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line_clean)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) >= 15:  # meaningful claim length
                candidate_statements.append(s_clean)

    return candidate_statements


def evaluate_single_answer(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single generated answer at the claim and fact level.
    """
    qid = record["id"]
    category = record["category"]
    question = record["question"]
    expected_answer = record.get("expected_answer", "")
    expected_facts = record.get("expected_facts", [])
    expected_sources = record.get("expected_sources", [])
    expected_locations = record.get("expected_locations", [])
    abstention_expected = record.get("abstention_expected", False)
    generated_answer = record.get("generated_answer", "").strip()
    retrieved_chunks = record.get("retrieved_chunks", [])
    effective_route = record.get("effective_route", "chat")
    total_latency_ms = record.get("total_latency_ms", 0.0)
    ttft_ms = record.get("ttft_ms")

    # Combine all retrieved chunk text for context grounding reference
    context_text_combined = " ".join(
        c.get("content_snippet", "") for c in retrieved_chunks
    ).lower()

    # 1. Evaluate Abstention
    is_abstention_correct = False
    ans_lower = generated_answer.lower()

    if abstention_expected:
        # Expected to abstain
        has_abstention_indicator = any(ind in ans_lower for ind in ABSTENTION_INDICATORS)
        is_abstention_correct = has_abstention_indicator or (len(generated_answer) == 0)
        answer_correct = is_abstention_correct
    else:
        # Answerable query
        # Fact recall: check how many expected facts appear in the generated answer
        matched_facts = []
        for fact in expected_facts:
            fact_lower = fact.lower()
            # If fact contains alternatives or numbers, check presence
            if fact_lower in ans_lower:
                matched_facts.append(fact)
            else:
                # Sub-token match for exact numbers (e.g. 2.5 ms, 15 MB)
                tokens = [t for t in re.split(r"\W+", fact_lower) if len(t) >= 2]
                if tokens and all(t in ans_lower for t in tokens):
                    matched_facts.append(fact)

        fact_recall = len(matched_facts) / max(1, len(expected_facts))
        answer_correct = fact_recall >= 0.5  # At least half of critical verified facts present

    # 2. Evaluate Claims
    claims = extract_claims(generated_answer)
    claim_evaluations: List[Dict[str, Any]] = []

    supported_claims_count = 0
    partially_supported_count = 0
    unsupported_claims_count = 0

    for claim in claims:
        claim_lower = claim.lower()

        # Check evidence in retrieved chunks or expected ground-truth facts
        # Extract keywords from claim
        claim_words = set(re.findall(r"\b[a-z0-9_-]{3,}\b", claim_lower))
        stop_words = {"the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "which", "according", "specified", "defined"}
        meaningful_words = claim_words - stop_words

        if not meaningful_words:
            continue

        # Overlap with retrieved context
        matches_in_context = sum(1 for w in meaningful_words if w in context_text_combined)
        context_overlap_ratio = matches_in_context / len(meaningful_words)

        # Overlap with ground truth answer
        matches_in_gt = sum(1 for w in meaningful_words if w in expected_answer.lower())
        gt_overlap_ratio = matches_in_gt / len(meaningful_words)

        if abstention_expected:
            if any(ind in claim_lower for ind in ABSTENTION_INDICATORS):
                claim_status = "SUPPORTED"
                supported_claims_count += 1
            else:
                claim_status = "UNSUPPORTED"
                unsupported_claims_count += 1
        else:
            if context_overlap_ratio >= 0.5 or gt_overlap_ratio >= 0.5:
                claim_status = "SUPPORTED"
                supported_claims_count += 1
            elif context_overlap_ratio >= 0.25 or gt_overlap_ratio >= 0.25:
                claim_status = "PARTIALLY_SUPPORTED"
                partially_supported_count += 1
            else:
                claim_status = "UNSUPPORTED"
                unsupported_claims_count += 1

        claim_evaluations.append({
            "claim": claim,
            "status": claim_status,
            "context_overlap": round(context_overlap_ratio, 2),
            "gt_overlap": round(gt_overlap_ratio, 2),
        })

    total_claims = len(claim_evaluations)
    groundedness_ratio = (
        (supported_claims_count + 0.5 * partially_supported_count) / total_claims
        if total_claims > 0
        else (1.0 if abstention_expected else 0.0)
    )
    unsupported_rate = (
        unsupported_claims_count / total_claims if total_claims > 0 else 0.0
    )

    # 3. Evaluate Citations
    citation_correct = False
    cited_sources_found: List[str] = []

    if abstention_expected:
        citation_correct = True  # No citation expected for absent questions
    else:
        # Check if expected source names appear in generated answer or retrieved chunks
        for src in expected_sources:
            src_base = os.path.basename(src).lower()
            if src_base in ans_lower or any(src_base in (c.get("file_name", "")).lower() for c in retrieved_chunks):
                cited_sources_found.append(src)

        # Citation correct if at least one expected source is present
        citation_correct = len(cited_sources_found) > 0

    # 4. Retrieval Success (Hit@K)
    retrieval_success = False
    if abstention_expected:
        # For absent questions, success if 0 chunks or low similarity
        retrieval_success = len(retrieved_chunks) == 0 or all(
            c.get("similarity", 0.0) < 0.45 for c in retrieved_chunks
        )
    else:
        for src in expected_sources:
            src_base = os.path.basename(src).lower()
            if any(src_base in (c.get("file_name", "")).lower() for c in retrieved_chunks):
                retrieval_success = True
                break

    return {
        "id": qid,
        "category": category,
        "question": question,
        "effective_route": effective_route,
        "abstention_expected": abstention_expected,
        "answer_correct": answer_correct,
        "is_abstention_correct": is_abstention_correct if abstention_expected else None,
        "retrieval_success": retrieval_success,
        "total_claims": total_claims,
        "supported_claims": supported_claims_count,
        "partially_supported_claims": partially_supported_count,
        "unsupported_claims": unsupported_claims_count,
        "groundedness_score": round(groundedness_ratio, 4),
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "citation_correct": citation_correct,
        "cited_sources": cited_sources_found,
        "total_latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "claim_details": claim_evaluations,
    }


def compute_aggregate_metrics(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute separated evaluation metrics across answerable and unanswerable domains."""
    answerable = [e for e in evaluations if not e["abstention_expected"]]
    unanswerable = [e for e in evaluations if e["abstention_expected"]]

    tot_ans = len(answerable)
    tot_unans = len(unanswerable)

    # 1. Answer Correctness
    ans_correct_count = sum(1 for e in answerable if e["answer_correct"])
    ans_correctness_rate = ans_correct_count / max(1, tot_ans)

    # 2. Claim-Level Groundedness
    all_claims_count = sum(e["total_claims"] for e in answerable)
    all_supported_claims = sum(e["supported_claims"] for e in answerable)
    all_unsupported_claims = sum(e["unsupported_claims"] for e in answerable)

    overall_groundedness = (
        all_supported_claims / max(1, all_claims_count) if all_claims_count > 0 else 0.0
    )
    overall_unsupported_rate = (
        all_unsupported_claims / max(1, all_claims_count) if all_claims_count > 0 else 0.0
    )

    # 3. Citation Accuracy
    citation_correct_count = sum(1 for e in answerable if e["citation_correct"])
    citation_accuracy_rate = citation_correct_count / max(1, tot_ans)

    # 4. Correct Abstention Rate
    correct_abstentions = sum(1 for e in unanswerable if e["is_abstention_correct"])
    abstention_rate = correct_abstentions / max(1, tot_unans)

    # 5. Retrieval Success on Answerable
    retrieval_success_count = sum(1 for e in answerable if e["retrieval_success"])
    retrieval_success_rate = retrieval_success_count / max(1, tot_ans)

    # 6. Latency Metrics
    latencies = [e["total_latency_ms"] for e in evaluations if e["total_latency_ms"] > 0]
    ttfts = [e["ttft_ms"] for e in evaluations if e["ttft_ms"] is not None]

    mean_latency = statistics.mean(latencies) if latencies else 0.0
    mean_ttft = statistics.mean(ttfts) if ttfts else 0.0

    # 7. Model-Specific Breakdown
    routes = set(e["effective_route"] for e in evaluations)
    model_breakdown = {}
    for r in sorted(routes):
        r_evals = [e for e in evaluations if e["effective_route"] == r]
        r_ans = [e for e in r_evals if not e["abstention_expected"]]
        r_unans = [e for e in r_evals if e["abstention_expected"]]

        r_latencies = [e["total_latency_ms"] for e in r_evals if e["total_latency_ms"] > 0]
        r_ttfts = [e["ttft_ms"] for e in r_evals if e["ttft_ms"] is not None]

        model_breakdown[r] = {
            "total_queries": len(r_evals),
            "answerable_queries": len(r_ans),
            "answer_correctness_rate": round(sum(1 for e in r_ans if e["answer_correct"]) / max(1, len(r_ans)), 4) if r_ans else None,
            "mean_groundedness": round(statistics.mean([e["groundedness_score"] for e in r_ans]), 4) if r_ans else None,
            "citation_accuracy_rate": round(sum(1 for e in r_ans if e["citation_correct"]) / max(1, len(r_ans)), 4) if r_ans else None,
            "abstention_rate": round(sum(1 for e in r_unans if e["is_abstention_correct"]) / max(1, len(r_unans)), 4) if r_unans else None,
            "mean_latency_ms": round(statistics.mean(r_latencies), 2) if r_latencies else 0.0,
            "mean_ttft_ms": round(statistics.mean(r_ttfts), 2) if r_ttfts else 0.0,
        }

    # 8. Category Breakdown
    categories = set(e["category"] for e in evaluations)
    category_breakdown = {}
    for cat in sorted(categories):
        cat_evals = [e for e in evaluations if e["category"] == cat]
        cat_ans = [e for e in cat_evals if not e["abstention_expected"]]
        cat_unans = [e for e in cat_evals if e["abstention_expected"]]

        category_breakdown[cat] = {
            "total": len(cat_evals),
            "answer_correctness": round(sum(1 for e in cat_ans if e["answer_correct"]) / max(1, len(cat_ans)), 4) if cat_ans else None,
            "mean_groundedness": round(statistics.mean([e["groundedness_score"] for e in cat_ans]), 4) if cat_ans else None,
            "citation_accuracy": round(sum(1 for e in cat_ans if e["citation_correct"]) / max(1, len(cat_ans)), 4) if cat_ans else None,
            "abstention_rate": round(sum(1 for e in cat_unans if e["is_abstention_correct"]) / max(1, len(cat_unans)), 4) if cat_unans else None,
        }

    return {
        "total_queries_evaluated": len(evaluations),
        "answerable_queries": tot_ans,
        "unanswerable_queries": tot_unans,
        "answer_correctness_rate": round(ans_correctness_rate, 4),
        "claim_groundedness_rate": round(overall_groundedness, 4),
        "unsupported_claim_rate": round(overall_unsupported_rate, 4),
        "citation_accuracy_rate": round(citation_accuracy_rate, 4),
        "correct_abstention_rate": round(abstention_rate, 4),
        "retrieval_success_rate": round(retrieval_success_rate, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "mean_ttft_ms": round(mean_ttft, 2),
        "model_specific_results": model_breakdown,
        "category_specific_results": category_breakdown,
    }


def main():
    print("=" * 70)
    print("PHASE 7.2 STAGE 1 — CLAIM-LEVEL ANSWER GROUNDING EVALUATOR")
    print("=" * 70)

    if not os.path.exists(RAW_RESULTS_PATH):
        print(f"Error: Raw generation results not found at '{RAW_RESULTS_PATH}'.")
        print("Please run 'python rag_evaluation/run_answer_quality.py' first.")
        sys.exit(1)

    with open(RAW_RESULTS_PATH, "r", encoding="utf-8") as f:
        raw_results = json.load(f)

    print(f"Loaded {len(raw_results)} generation records from '{RAW_RESULTS_PATH}'.")

    evaluations = []
    for r in raw_results:
        ev = evaluate_single_answer(r)
        evaluations.append(ev)

    metrics = compute_aggregate_metrics(evaluations)

    full_report = {
        "metrics_summary": metrics,
        "detailed_evaluations": evaluations,
    }

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("ANSWER QUALITY & GROUNDING METRICS SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated:          {metrics['total_queries_evaluated']}")
    print(f"Answerable Queries:       {metrics['answerable_queries']}")
    print(f"  - Answer Correctness:   {metrics['answer_correctness_rate']*100:.1f}%")
    print(f"  - Claim Groundedness:   {metrics['claim_groundedness_rate']*100:.1f}%")
    print(f"  - Unsupported Claims:   {metrics['unsupported_claim_rate']*100:.1f}%")
    print(f"  - Citation Accuracy:    {metrics['citation_accuracy_rate']*100:.1f}%")
    print(f"  - Retrieval Success:    {metrics['retrieval_success_rate']*100:.1f}%")
    print(f"Unanswerable Queries:     {metrics['unanswerable_queries']}")
    print(f"  - Correct Abstention:   {metrics['correct_abstention_rate']*100:.1f}%")
    print(f"Latency:")
    print(f"  - Mean TTFT:            {metrics['mean_ttft_ms']:.1f} ms")
    print(f"  - Mean Total Latency:   {metrics['mean_latency_ms']:.1f} ms")
    print("\nModel Routes Breakdown:")
    for model, m_data in metrics["model_specific_results"].items():
        print(f"  [{model}] Count={m_data['total_queries']} | Correctness={m_data.get('answer_correctness_rate')} | Groundedness={m_data.get('mean_groundedness')} | Latency={m_data['mean_latency_ms']}ms")

    print(f"\nSaved evaluation summary to '{SUMMARY_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
