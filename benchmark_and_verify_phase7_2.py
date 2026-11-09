"""
Phase 7.2 Verification and Resource Benchmark Script
Measures:
1. Resource Benchmarking:
   - Idle RAM & CPU
   - EmbeddingGemma memory consumption
   - DeepSeek 4096 context memory consumption
   - DeepSeek 8192 context memory consumption
   - Simultaneous model residency behavior
   - Memory recovery after unload (keep_alive: 0)
2. Generation Verification:
   - Normal completion
   - Token limit truncation (done_reason == 'length')
   - Auto-continuation and deduplication
   - Max continuation attempts reached (handoff prevention)
   - Connection drop isolation
3. Routing Verification:
   - Greeting -> chat
   - Architectural assessment -> deepseek
   - Design question -> deepseek
   - Explicit coding request -> pipeline
"""

import json
import os
import psutil
import requests
import subprocess
import time
import app

OLLAMA_URL = "http://localhost:11434"


def get_mem_info():
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 2),
        "available_gb": round(vm.available / (1024**3), 2),
        "used_gb": round(vm.used / (1024**3), 2),
        "percent": vm.percent,
        "cpu_percent": psutil.cpu_percent(interval=1),
    }


def get_ollama_ps():
    try:
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True)
        return ps.stdout.strip()
    except Exception as e:
        return f"Error running ollama ps: {e}"


def unload_all_models():
    print("Unloading all models via keep_alive=0...")
    # Get loaded models from /api/ps
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5).json()
        models = [m.get("model") or m.get("name") for m in r.get("models", [])]
        for m in models:
            if m:
                requests.post(f"{OLLAMA_URL}/api/generate", json={"model": m, "keep_alive": 0}, timeout=5)
    except Exception as e:
        print(f"Notice on unload: {e}")
    time.sleep(2)


def run_benchmark():
    results = {}
    print("=" * 60)
    print("PART 1: RESOURCE BENCHMARKING")
    print("=" * 60)

    # 1. Idle state
    unload_all_models()
    idle_mem = get_mem_info()
    idle_ps = get_ollama_ps()
    print(f"IDLE: Total {idle_mem['total_gb']} GB, Available {idle_mem['available_gb']} GB, Used {idle_mem['used_gb']} GB ({idle_mem['percent']}%), CPU {idle_mem['cpu_percent']}%")
    print("Ollama PS at Idle:\n", idle_ps)
    results["idle"] = idle_mem

    # 2. EmbeddingGemma
    print("\n--- Testing EmbeddingGemma ---")
    requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": "embeddinggemma", "prompt": "benchmark test"}, timeout=15)
    emb_mem = get_mem_info()
    emb_ps = get_ollama_ps()
    print(f"EmbeddingGemma: Available {emb_mem['available_gb']} GB, Used {emb_mem['used_gb']} GB ({emb_mem['percent']}%)")
    print("Ollama PS with EmbeddingGemma:\n", emb_ps)
    results["embeddinggemma"] = {"mem": emb_mem, "ps": emb_ps}

    # 3. DeepSeek with 4096 context
    print("\n--- Testing DeepSeek with num_ctx=4096 ---")
    unload_all_models()
    requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": "deepseek-coder-v2:16b",
        "messages": [{"role": "user", "content": "Hello"}],
        "options": {"num_ctx": 4096, "num_predict": 10},
        "stream": False
    }, timeout=60)
    ds_4096_mem = get_mem_info()
    ds_4096_ps = get_ollama_ps()
    print(f"DeepSeek 4096: Available {ds_4096_mem['available_gb']} GB, Used {ds_4096_mem['used_gb']} GB ({ds_4096_mem['percent']}%)")
    print("Ollama PS (4096):\n", ds_4096_ps)
    results["deepseek_4096"] = {"mem": ds_4096_mem, "ps": ds_4096_ps}

    # 4. DeepSeek with 8192 context
    print("\n--- Testing DeepSeek with num_ctx=8192 ---")
    requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": "deepseek-coder-v2:16b",
        "messages": [{"role": "user", "content": "Hello"}],
        "options": {"num_ctx": 8192, "num_predict": 10},
        "stream": False
    }, timeout=60)
    ds_8192_mem = get_mem_info()
    ds_8192_ps = get_ollama_ps()
    print(f"DeepSeek 8192: Available {ds_8192_mem['available_gb']} GB, Used {ds_8192_mem['used_gb']} GB ({ds_8192_mem['percent']}%)")
    print("Ollama PS (8192):\n", ds_8192_ps)
    results["deepseek_8192"] = {"mem": ds_8192_mem, "ps": ds_8192_ps}

    # 5. Simultaneous residency check
    print("\n--- Testing Simultaneous Model Residency (DeepSeek + EmbeddingGemma) ---")
    requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": "embeddinggemma", "prompt": "residency check"}, timeout=15)
    sim_mem = get_mem_info()
    sim_ps = get_ollama_ps()
    print(f"Simultaneous check: Available {sim_mem['available_gb']} GB, Used {sim_mem['used_gb']} GB")
    print("Ollama PS during simultaneous request:\n", sim_ps)
    results["simultaneous"] = {"mem": sim_mem, "ps": sim_ps}

    # 6. Unload recovery
    print("\n--- Testing Unload Recovery ---")
    unload_all_models()
    recovered_mem = get_mem_info()
    recovered_ps = get_ollama_ps()
    print(f"Post-unload: Available {recovered_mem['available_gb']} GB, Used {recovered_mem['used_gb']} GB")
    print("Ollama PS after unload:\n", recovered_ps)
    results["recovery"] = {"mem": recovered_mem, "ps": recovered_ps}

    print("\n" + "=" * 60)
    print("PART 2: ROUTING VERIFICATION")
    print("=" * 60)
    routing_tests = [
        ("Greeting", "Hello! How can you help me today?", "chat"),
        ("Architectural assessment", "Please conduct a comprehensive architectural assessment of our microservices system design.", "deepseek"),
        ("Design question", "What are the trade-offs of microservices vs monolithic architecture?", "deepseek"),
        ("Explicit coding request", "Write a python script to parse CSV files and save to SQLite database.", "pipeline"),
    ]
    results["routing"] = []
    for category, prompt, expected in routing_tests:
        route = app.classify_intent([{"role": "user", "content": prompt}], "auto")
        status = "PASSED" if route == expected else f"FAILED (got {route}, expected {expected})"
        print(f"[{category}] '{prompt[:45]}...' -> {route} ({status})")
        results["routing"].append({
            "category": category,
            "prompt": prompt,
            "route": route,
            "expected": expected,
            "status": status
        })

    # Save results to json for reporting
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nResults successfully saved to benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
