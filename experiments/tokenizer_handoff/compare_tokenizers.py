"""
Compare Tokenizers Script
Phase 3 of EmbeddingGemma 2 <-> Gemma 4 Tokenizer Compatibility & Direct Injection Investigation.
Performs exact 1-to-1 comparisons across vocabulary, scores, token types, merges, special tokens,
and encodes test suites (short, code, unicode, json, chat templates) to verify token ID equivalence.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple, Any

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_vocab_artifact(model_slug: str) -> Dict[str, Any]:
    filepath = os.path.join(OUTPUT_DIR, f"vocab_{model_slug}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing vocabulary file: {filepath}. Run extract_tokenizers.py first.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def build_tokenizer_engine(data: Dict[str, Any]) -> Tokenizer:
    """Builds a HuggingFace Tokenizer using the exact GGUF extracted parameters."""
    tokens = data["tokens"]
    merges_raw = data.get("merges", [])
    meta = data["metadata"]

    vocab = {token: i for i, token in enumerate(tokens)}
    merges = []
    for m in merges_raw:
        parts = m.split(" ")
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
        elif len(parts) > 2:
            merges.append((parts[0], " ".join(parts[1:])))

    unk_id = meta.get("tokenizer.ggml.unknown_token_id", 3)
    unk_token = tokens[unk_id] if unk_id < len(tokens) else "<unk>"

    model = BPE(vocab=vocab, merges=merges, unk_token=unk_token)
    tok = Tokenizer(model)

    # Pre-tokenizer: Metaspace with no leading space prepend (matching ggml.add_space_prefix: False)
    tok.pre_tokenizer = Metaspace(replacement=" ", prepend_scheme="never")

    # Post-processor: add BOS token if configured
    add_bos = meta.get("tokenizer.ggml.add_bos_token", True)
    bos_id = meta.get("tokenizer.ggml.bos_token_id", 2)
    eos_id = meta.get("tokenizer.ggml.eos_token_id", 1)
    bos_token = tokens[bos_id] if bos_id < len(tokens) else "<bos>"
    eos_token = tokens[eos_id] if eos_id < len(tokens) else "<eos>"

    if add_bos:
        tok.post_processor = TemplateProcessing(
            single=f"{bos_token} $A",
            pair=f"{bos_token} $A {bos_token} $B",
            special_tokens=[
                (bos_token, bos_id),
                (eos_token, eos_id),
            ],
        )

    return tok

def run_compatibility_suite():
    print("=" * 75)
    print("PHASE 3: EXACT TOKENIZER COMPATIBILITY INVESTIGATION")
    print("=" * 75)

    # 1. Load artifacts
    print("\n[1] Loading extracted vocabulary artifacts...")
    data_emb2 = load_vocab_artifact("embeddinggemma_2_270m")
    data_gemma4 = load_vocab_artifact("gemma4_12b")
    data_old_emb = load_vocab_artifact("embeddinggemma_latest")

    tokens_emb2 = data_emb2["tokens"]
    tokens_gemma4 = data_gemma4["tokens"]
    tokens_old = data_old_emb["tokens"]

    print(f"    - EmbeddingGemma 2 (270M) vocab size : {len(tokens_emb2):,}")
    print(f"    - Gemma 4 (12B) vocab size           : {len(tokens_gemma4):,}")
    print(f"    - Old EmbeddingGemma vocab size      : {len(tokens_old):,}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_compared": {
            "target": "embeddinggemma-2:270m",
            "reference": "gemma4:12b",
            "legacy": "embeddinggemma:latest"
        },
        "structural_comparison": {},
        "special_tokens_comparison": {},
        "test_cases": [],
        "overall_classification": "UNVERIFIED"
    }

    # 2. Structural & Vocabulary Identity
    print("\n[2] Verifying structural token identity (EmbeddingGemma 2 vs Gemma 4)...")
    exact_vocab_match = (tokens_emb2 == tokens_gemma4)
    exact_scores_match = (data_emb2.get("scores") == data_gemma4.get("scores"))
    exact_types_match = (data_emb2.get("token_types") == data_gemma4.get("token_types"))
    exact_merges_match = (data_emb2.get("merges") == data_gemma4.get("merges"))

    first_diff_idx = None
    if not exact_vocab_match:
        for idx in range(min(len(tokens_emb2), len(tokens_gemma4))):
            if tokens_emb2[idx] != tokens_gemma4[idx]:
                first_diff_idx = idx
                break

    report["structural_comparison"] = {
        "exact_vocab_match": exact_vocab_match,
        "exact_scores_match": exact_scores_match,
        "exact_types_match": exact_types_match,
        "exact_merges_match": exact_merges_match,
        "first_difference_index": first_diff_idx,
        "vocab_size_emb2": len(tokens_emb2),
        "vocab_size_gemma4": len(tokens_gemma4),
        "merges_count_emb2": len(data_emb2.get("merges", [])),
        "merges_count_gemma4": len(data_gemma4.get("merges", [])),
    }

    print(f"    - All 262,144 tokens match exactly  : {exact_vocab_match}")
    print(f"    - All 262,144 BPE scores match       : {exact_scores_match}")
    print(f"    - All 262,144 token types match      : {exact_types_match}")
    print(f"    - All 514,906 merge rules match      : {exact_merges_match}")
    if first_diff_idx is not None:
        print(f"    ! First difference detected at index: {first_diff_idx}")

    # Legacy comparison
    print("\n[3] Comparing Legacy EmbeddingGemma vs Gemma 4 (Divergence Check)...")
    legacy_vocab_match = (tokens_old == tokens_gemma4)
    legacy_diff_idx = None
    for idx in range(min(len(tokens_old), len(tokens_gemma4))):
        if tokens_old[idx] != tokens_gemma4[idx]:
            legacy_diff_idx = idx
            break
    print(f"    - Legacy vocab matches Gemma 4      : {legacy_vocab_match}")
    print(f"    - Legacy first token difference at  : ID {legacy_diff_idx} (old='{repr(tokens_old[legacy_diff_idx])}', gemma4='{repr(tokens_gemma4[legacy_diff_idx])}')")
    print(f"    - Legacy merge rules count          : {len(data_old_emb.get('merges', []))} (Gemma 3 uses Unigram; Gemma 4 uses BPE)")

    # 4. Special Tokens Comparison
    print("\n[4] Comparing Special Token Identifiers...")
    meta_e2 = data_emb2["metadata"]
    meta_g4 = data_gemma4["metadata"]

    special_keys = [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.mask_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_space_prefix",
    ]
    spec_comp = {}
    all_spec_match = True
    for sk in special_keys:
        v_e2 = meta_e2.get(sk)
        v_g4 = meta_g4.get(sk)
        match = (v_e2 == v_g4)
        if not match:
            all_spec_match = False
        spec_comp[sk] = {"embeddinggemma_2": v_e2, "gemma4": v_g4, "match": match}
        print(f"    - {sk:<32}: Emb2={v_e2!s:<6} | Gemma4={v_g4!s:<6} | Match={match}")
    report["special_tokens_comparison"] = spec_comp

    # 5. Build Tokenizer Engines
    print("\n[5] Initializing Tokenizer Engines from extracted assets...")
    tok_emb2 = build_tokenizer_engine(data_emb2)
    tok_gemma4 = build_tokenizer_engine(data_gemma4)
    print("    - Both Tokenizer engines initialized with Metaspace & BOS templates.")

    # 6. Test Suite Execution
    print("\n[6] Running Exact Token ID Sequence Tests across diverse inputs...")

    test_inputs = [
        {
            "category": "Basic Greeting",
            "name": "hey",
            "text": "hey"
        },
        {
            "category": "Standard Conversational",
            "name": "Hello, how are you?",
            "text": "Hello, how are you?"
        },
        {
            "category": "Domain Terminology",
            "name": "Explain retrieval augmented generation.",
            "text": "Explain retrieval augmented generation."
        },
        {
            "category": "Source Code",
            "name": "Python Function Definition",
            "text": 'def hello_world(): return "hello"'
        },
        {
            "category": "Unicode & Multilingual",
            "name": "Multilingual Polyglot with Emojis",
            "text": "नमस्ते दुनिया! こんにちは世界 🌍 مرحبا بالعالم. Über die Brücke gehen: ä ö ü ß. Code: 🚀🔥⚡"
        },
        {
            "category": "Escaped JSON & SQL",
            "name": "Escaped Characters in Payload",
            "text": '{"query": "SELECT * FROM \\"users\\" WHERE status = \'active\' AND notes LIKE \'%hello\\\\nworld%\'", "max_tokens": 100, "escapes": "\\"\\\\/\\b\\f\\n\\r\\t"}'
        },
        {
            "category": "Complete Chatbot Prompt",
            "name": "Full Realistic Prompt with System, RAG Context, and History",
            "text": (
                "<start_of_turn>system\n"
                "You are a helpful, friendly, and knowledgeable AI assistant powered by Gemma 4.\n"
                "Provide clear, concise, and structured answers.\n"
                "Context from knowledge base:\n"
                "- Unified RAG manages documents and codebase collections in ChromaDB.\n"
                "- Gemma 4 12B operates locally via Ollama with speculative drafting.\n"
                "<end_of_turn>\n"
                "<start_of_turn>user\n"
                "Can EmbeddingGemma 2 pre-tokenize inputs directly for Gemma 4 to accelerate TTFT?\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        }
    ]

    all_tests_passed = True

    for i, test in enumerate(test_inputs, 1):
        enc_e2 = tok_emb2.encode(test["text"])
        enc_g4 = tok_gemma4.encode(test["text"])

        ids_e2 = enc_e2.ids
        ids_g4 = enc_g4.ids
        tokens_e2 = enc_e2.tokens
        tokens_g4 = enc_g4.tokens

        is_identical = (ids_e2 == ids_g4)
        if not is_identical:
            all_tests_passed = False

        first_diff = None
        if not is_identical:
            for d_idx in range(min(len(ids_e2), len(ids_g4))):
                if ids_e2[d_idx] != ids_g4[d_idx]:
                    first_diff = {
                        "position": d_idx,
                        "emb2_id": ids_e2[d_idx],
                        "gemma4_id": ids_g4[d_idx],
                        "emb2_tok": tokens_e2[d_idx],
                        "gemma4_tok": tokens_g4[d_idx]
                    }
                    break

        case_res = {
            "test_num": i,
            "category": test["category"],
            "name": test["name"],
            "input_preview": test["text"][:60] + "..." if len(test["text"]) > 60 else test["text"],
            "token_count": len(ids_e2),
            "identical": is_identical,
            "first_diff": first_diff,
            "sample_ids": ids_e2[:8]
        }
        report["test_cases"].append(case_res)

        status_str = "PASS (100% Identical)" if is_identical else f"FAIL (Diff at {first_diff['position']})"
        print(f"    Test {i:02d} [{test['category']:<24}]: {status_str} ({len(ids_e2)} tokens)")
        print(f"            IDs: {ids_e2[:10]}")

    # 7. Classification
    if exact_vocab_match and exact_merges_match and all_spec_match and all_tests_passed:
        report["overall_classification"] = "EXACTLY COMPATIBLE"
    elif exact_vocab_match and all_tests_passed:
        report["overall_classification"] = "PARTIALLY COMPATIBLE"
    else:
        report["overall_classification"] = "INCOMPATIBLE"

    print("\n" + "=" * 75)
    print(f"FINAL CLASSIFICATION: {report['overall_classification']}")
    print("=" * 75)

    out_file = os.path.join(OUTPUT_DIR, "compatibility_report.json")
    with open(out_file, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)
    print(f"\nDetailed compatibility report saved to: {out_file}")

    return report

if __name__ == "__main__":
    run_compatibility_suite()
