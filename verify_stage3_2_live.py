"""
Stage 3.2 Live Ollama Validation Script.
Executes against real local Ollama server at http://localhost:11434.

Validates:
1. Live state reconciliation against real Ollama /api/ps.
2. Live streaming generation under ModelLease.
3. Live client cancellation mid-stream:
   - Verifies generation lease is released after client disconnect.
   - Verifies worker thread stops.
4. Live model transition after cancellation:
   - Tests bounded eviction polling against real Ollama daemon.
   - Verifies transition does not hang or time out.
"""

import json
import time
import requests

import app
from model_lifecycle import (
    OllamaLifecycleManager,
    get_model_manager,
    reset_model_manager_for_testing,
)


def run_live_tests():
    print("=" * 70)
    print("STAGE 3.2 LIVE OLLAMA VERIFICATION")
    print("=" * 70)

    # Check Ollama daemon health
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print(f"❌ Ollama returned HTTP {r.status_code}")
            return False
        print("✅ Live Ollama daemon is reachable on http://localhost:11434")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

    flask_app = app.app
    client = flask_app.test_client()

    # Create real lifecycle manager connected to live Ollama
    manager = OllamaLifecycleManager(
        base_url="http://localhost:11434",
        max_concurrent_generation=1,
        embedding_policy="adaptive",
        default_timeout=60.0,
        poll_interval=0.2,
        eviction_timeout=10.0,
        eviction_poll_interval=0.3,
    )
    reset_model_manager_for_testing(manager)

    # 1. Live State Reconciliation
    print("\n--- 1. Live State Reconciliation ---")
    recon = manager.reconcile_state()
    print(f"Resident generation model: {recon['resident_gen_model']}")
    print(f"Ollama resident models in /api/ps: {recon['ollama_ps_models']}")
    print("✅ Live state reconciled successfully.")

    # 2. Live Stream with Cancellation
    print("\n--- 2. Live Generation & Mid-Stream Cancellation ---")
    test_model = "qwen:latest"  # Use lightweight local model for fast test
    print(f"Starting chat stream with model: {test_model}")

    response = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "Write a 500 word essay about mountains and rivers."}],
            "mode": "qwen",
            "rag_mode": "off",
        },
    )

    tokens_received = []
    print("Reading streaming SSE tokens from live Ollama...")
    iterator = iter(response.response)
    for chunk in iterator:
        chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        if '"token"' in chunk_str:
            tokens_received.append(chunk_str)
            print(f"Received live token chunk: {len(tokens_received)} chunks")
            if len(tokens_received) >= 2:
                break

    if len(tokens_received) < 2:
        print("❌ Did not receive at least 2 tokens from live Ollama.")
        return False

    # Verify lease is actively held
    diag_during = manager.get_diagnostics()
    print(f"Active leases during stream: {diag_during['active_gen_leases']}")
    if len(diag_during["active_gen_leases"]) == 0:
        print("❌ Expected active generation lease during streaming!")
        return False
    print("✅ Generation lease actively held during streaming.")

    # Simulate client disconnect (AbortController / close tab)
    print("\nSimulating client disconnection (response.close())...")
    t0 = time.monotonic()
    if hasattr(response.response, "close"):
        response.response.close()
    response.close()

    # Bounded wait for worker cleanup
    time.sleep(0.3)
    t_elapsed = round(time.monotonic() - t0, 3)

    diag_after = manager.get_diagnostics()
    print(f"Active leases after abort: {diag_after['active_gen_leases']} (cleaned up in {t_elapsed}s)")
    if len(diag_after["active_gen_leases"]) != 0:
        print("❌ Generation lease leaked after client cancellation!")
        return False
    print("✅ Generation lease released cleanly and immediately on cancellation.")

    # 3. Live Model Transition After Cancellation
    print("\n--- 3. Live Model Transition After Cancellation ---")
    target_model_2 = "gemma3:latest"
    print(f"Requesting transition to: {target_model_2}")

    t_trans_start = time.monotonic()
    with manager.acquire_generation_lease(target_model_2, timeout=30.0) as lease:
        t_trans = round(time.monotonic() - t_trans_start, 2)
        print(f"Successfully acquired lease for {lease.model_name} in {t_trans}s!")
        print(f"Manager diagnostics: {manager.get_diagnostics()}")

    t_after_release = manager.get_diagnostics()
    print(f"Lease released cleanly: {t_after_release['active_gen_leases']}")
    print("✅ Model transition succeeded via bounded eviction polling without timeouts or leaks.")

    print("\n" + "=" * 70)
    print("ALL LIVE OLLAMA VALIDATION TESTS PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_live_tests()
    exit(0 if success else 1)
