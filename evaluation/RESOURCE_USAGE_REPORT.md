# Hardware Resource Utilization & Model Residency Report

**Project:** `C:\Users\Gurnoor\Desktop\gemma-chatbot`  
**Host Environment:** Windows 11, Intel Core i7-13620H, 16.0 GB Total Physical RAM, Intel UHD Graphics  
**Evaluation Date:** 2026-11-10  
**Phase:** Comprehensive Evaluation & Verification  

---

## 1. Executive Summary

This report evaluates system resource utilization across the Flask backend, SQLite persistence store, ChromaDB vector collections, and Ollama model runner on a resource-constrained 16 GB RAM laptop.

Key findings:
1. **Application Memory Footprint:** The combined Python runtime (Flask, SQLite in WAL mode, and 3 initialized ChromaDB collections) maintains an idle memory footprint of **$\sim 108 – 134\text{ MB}$**, scaling to **$\sim 168\text{ MB}$** when 1,000 conversation turns are indexed.
2. **CPU Utilization:** Backend orchestration, database transactions, and vector searches consume **$0.0\% – 9.4\%$ CPU**, leaving the host processor unburdened for model inference.
3. **Model Memory Contention on 16 GB RAM:**
   - Available Host Free Memory: $\sim 6.5\text{ GB}$ (out of 16.0 GB total, with OS and background processes using $\sim 9.5\text{ GB}$).
   - `gemma4:12b` requires **8.0 GB**.
   - `deepseek-coder-v2:16b` requires **8.9 GB**.
   - Running both models simultaneously would require **$> 16.9\text{ GB}$**, instantly triggering OS memory exhaustion and disk paging thrash.
4. **Residency Policy Validation:** The `OllamaLifecycleManager` strictly enforces a single-resident generation model policy (`max_concurrent_generation = 1`). Eviction of active generation models before swapping is essential to prevent out-of-memory (OOM) fatal crashes.

---

## 2. Quantitative System Resource Profile

### 2.1 Backend Process Footprint Under Benchmark Load

| Component | State / Activity | RSS Memory (MB) | System RAM Used (GB) | System RAM % | Host CPU % |
|:---|:---|:---:|:---:|:---:|:---:|
| **Baseline Idle** | Python process started, no DB | **107.52 MB** | 9.22 GB | 58.9% | 0.0% |
| **SQLite WAL Loaded** | 100 persistent turns written | **108.45 MB** (+0.93 MB) | 9.45 GB | 60.3% | 0.0% |
| **ChromaDB Initialized** | 3 vector collections active | **133.49 MB** (+25.04 MB)| 9.47 GB | 60.4% | 0.0% |
| **Active Scaling (1K turns)** | 1,000 history turns indexed | **168.71 MB** (+35.22 MB)| 9.65 GB | 61.6% | 2.1% |
| **Full Evaluation Load** | Continuous queries & SSE streams | **123.98 MB** | 9.24 GB | 59.0% | 9.4% |

---

## 3. Local Model Memory Demands & Contention Analysis

### 3.1 Installed Ollama Model Inventory

| Model Name | Parameter / Quantization Scale | Disk / VRAM Size | Context Window | Minimum RAM Required | Concurrency Safety on 16 GB Host |
|:---|:---:|:---:|:---:|:---:|:---:|
| `embeddinggemma:latest` | 300M / Embedding | **621 MB** | 2,048 tokens | 1.0 GB | Safe (Can remain resident) |
| `embeddinggemma-2:270m` | 270M / Embedding | **303 MB** | 2,048 tokens | 0.8 GB | Safe (Can remain resident) |
| `qwen:latest` | 1.8B / Small Chat | **2.3 GB** | 4,096 tokens | 4.0 GB | Safe for standalone execution |
| `Qwen2.5-Coder:7B` | 7B / Coder | **4.7 GB** | 8,192 tokens | 7.0 GB | Borderline (Requires cold swap) |
| `gemma3:latest` | 4B / Fallback Chat | **3.3 GB** | 4,096 tokens | 5.5 GB | Safe for standalone execution |
| `gemma4:12b` | 12B / Primary Assistant | **8.0 GB** | 8,192 tokens | **10.5 GB** | **Exhausts free RAM without eviction** |
| `deepseek-coder-v2:16b` | 16B MoE / Architect | **8.9 GB** | 8,192 tokens | **11.5 GB** | **Exhausts free RAM without eviction** |

### 3.2 Model Switching Overhead & Thrashing Risk
Because the host machine possesses 16 GB of physical RAM with ~6.5 GB free before model load:
- When switching from `gemma4:12b` to `deepseek-coder-v2:16b`, `gemma4` **MUST** be completely unloaded from memory (`keep_alive: 0`) before `deepseek` can begin allocating tensors.
- Unloading and loading takes **10 – 25 seconds** on CPU/integrated graphics.
- If model switching occurs frequently (e.g., alternating between general assistant and architect every turn), user-perceived TTFT degrades drastically to $> 20\text{ seconds}$ purely due to model loading.

---

## 4. Architectural Safeguards Verified

1. **Active Model Protection:** While a model lease is held, no competing request can evict the active model.
2. **Sequential Queueing:** Competing generation requests block on `self._cv.wait()` with a bounded monotonic deadline rather than spawning concurrent Ollama processes.
3. **Adaptive Embedding Draining:** When a heavy generation model (`deepseek-coder-v2:16b`) acquires a lease, active embedding leases are drained and `embeddinggemma` is evicted to maximize headroom.

---

## 5. Artifact Data Links

- **Full Resource Data JSON:** [`evaluation/data/performance_benchmark_data.json`](file:///C:/Users/Gurnoor/Desktop/gemma-chatbot/evaluation/data/performance_benchmark_data.json)
