# Intent-Driven Conversational History Retrieval Decision Accuracy Report

**Project:** `C:\Users\Gurnoor\Desktop\gemma-chatbot`  
**Evaluation Target:** `rag/service.py` (`detect_retrieval_mode`), `app.py` (`classify_intent`)  
**Evaluation Date:** 2026-11-10  
**Phase:** Strict Evaluation & Benchmarking  
**Total Labelled Test Queries:** 65 across 5 categories  

---

## 1. Executive Summary & Evaluation Objective

The Gemma Chatbot architecture specifies that conversational history retrieval must not run indiscriminately for every message. Rather, it must follow an intent-driven pipeline:

$$\text{User Request} \longrightarrow \text{Intent Detection} \longrightarrow \text{Determine History Retrieval} \longrightarrow \text{Selective Retrieval} \longrightarrow \text{Relevance Filtering} \longrightarrow \text{Inference}$$

This evaluation rigorously audits the **retrieval decision stage** against a manually labelled dataset of 65 queries covering:
- **Category 1:** Ordinary messages (retrieval should be skipped)
- **Category 2:** Explicit historical references (retrieval required)
- **Category 3:** Implicit historical references (retrieval required)
- **Category 4:** Project-context-dependent requests (retrieval required only if historical decisions needed)
- **Category 5:** Ambiguous messages (resolved via live sliding window vs vector history)

---

## 2. Quantitative Evaluation Metrics

### 2.1 Confusion Matrix

| | Actual Retrieval: Triggered (Positive) | Actual Retrieval: Skipped (Negative) | Total |
|---|:---:|:---:|:---:|
| **Expected Retrieval (Positive)** | **16 (TP)** | **13 (FN)** | **29** |
| **Expected Skip (Negative)** | **0 (FP)** | **36 (TN)** | **36** |
| **Total** | **16** | **49** | **65** |

### 2.2 Global Classification Performance

$$\text{Accuracy} = \frac{TP + TN}{N} = \frac{16 + 36}{65} = \mathbf{80.00\%}$$

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{16}{16 + 0} = \mathbf{100.00\%}$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{16}{16 + 13} = \mathbf{55.17\%}$$

$$F_1\text{ Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{0.7111}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN} = \frac{0}{36} = \mathbf{0.00\%}$$

$$\text{False Negative Rate (FNR)} = \frac{FN}{FN + TP} = \frac{13}{29} = \mathbf{44.83\%}$$

---

## 3. Category Breakdown & Empirical Analysis

| Category | Total Queries | Correct Predictions | Category Accuracy | True Positives | True Negatives | False Positives | False Negatives |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cat 1: Ordinary Messages** | 13 | 13 | **100.00%** | 0 | 13 | 0 | 0 |
| **Cat 2: Explicit History** | 13 | 12 | **92.31%** | 12 | 0 | 0 | 1 |
| **Cat 3: Implicit History** | 13 | 2 | **15.38%** | 2 | 0 | 0 | 11 |
| **Cat 4: Project Context** | 13 | 12 | **92.31%** | 1 | 11 | 0 | 1 |
| **Cat 5: Ambiguous Queries** | 13 | 13 | **100.00%** | 1 | 12 | 0 | 0 |
| **Total** | **65** | **52** | **80.00%** | **16** | **36** | **0** | **13** |

---

## 4. In-Depth Failure Analysis: The False Negative Crisis

### 4.1 Key Finding: Ultra-Conservative Bias
The intent classification heuristic (`MEMORY_PATTERN` in `rag/service.py`) was designed with an ultra-conservative bias to prevent unnecessary vector lookups for casual greetings and small talk. 
- **The Good:** Zero False Positives ($\text{FPR} = 0.00\%$). Greetings like *"Hey."*, *"How are you?"*, and coding queries like *"Explain Python decorators"* never trigger vector memory lookups.
- **The Critical Bug:** A severe False Negative Rate ($\text{FNR} = 44.83\%$). Out of 29 queries requiring historical context, **13 were completely missed**.

### 4.2 Category 3 Breakdown: Implicit Historical References
In Category 3, accuracy collapsed to **15.38%** (11 false negatives out of 13 queries):

| Query ID | Prompt | Expected | Actual Mode | Reason for Failure |
|---|---|:---:|:---:|---|
| `CAT3_01` | *"Why did we choose SQLite instead?"* | True | `off` | Heuristic lacks pattern for *"why did we choose [X]"* |
| `CAT3_02` | *"What was causing that memory issue?"* | True | `off` | Demonstrative pronoun *"that [issue]"* not recognized as history |
| `CAT3_03` | *"Did we already solve this problem?"* | True | `off` | Aspectual adverb *"already solve"* not in regex vocabulary |
| `CAT3_04` | *"Continue from where we stopped."* | True | `off` | Matches *"where we left off"* but misses *"where we stopped"* |
| `CAT3_06` | *"Is the latest change compatible with our existing design?"* | True | `off` | Misses *"existing design"* as a conversation checkpoint |
| `CAT3_07` | *"Why was that timeout occurring?"* | True | `off` | Past-tense causal question without temporal marker |
| `CAT3_08` | *"Did we fix that unclosed database handle?"* | True | `off` | Past resolution check *"did we fix that"* uncaptured |
| `CAT3_09` | *"How does this compare to our first attempt?"* | True | `off` | Historical ordinal *"first attempt"* uncaptured |
| `CAT3_10` | *"What was the conclusion on that issue?"* | True | `off` | Noun *"conclusion on that issue"* uncaptured |
| `CAT3_11` | *"Why did that test fail earlier?"* | True | `codebase` | Matched `codebase` ("test") but missed `memory` ("earlier") |
| `CAT3_12` | *"Are we still using that fallback approach?"* | True | `off` | Continual state inquiry *"are we still using"* uncaptured |

### 4.3 Severity Assessment
When a false negative occurs, the model is denied historical context. Consequently:
1. The model hallucinates or re-invents architecture decisions already finalized in previous turns.
2. The user is forced to re-explain context or use artificial phrasing (e.g. typing *"As we discussed earlier..."* instead of natural speech *"Why did we choose SQLite instead?"*).
3. In multi-agent interactions, outputs from DeepSeek and Qwen are orphaned because follow-up questions fail to pull their contributions into Gemma's prompt.

---

## 5. Summary of Datasets Generated

The full evaluation data is preserved in the repository for reproducibility:
- **JSON Dataset:** [`evaluation/data/retrieval_decision_dataset.json`](file:///C:/Users/Gurnoor/Desktop/gemma-chatbot/evaluation/data/retrieval_decision_dataset.json)
- **CSV Dataset:** [`evaluation/data/retrieval_decision_dataset.csv`](file:///C:/Users/Gurnoor/Desktop/gemma-chatbot/evaluation/data/retrieval_decision_dataset.csv)
- **Evaluation Runner:** [`evaluation/scripts/eval_retrieval_decision.py`](file:///C:/Users/Gurnoor/Desktop/gemma-chatbot/evaluation/scripts/eval_retrieval_decision.py)
