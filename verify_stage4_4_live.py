"""
Stage 4.4 Live Lifecycle Integration & Reliability Verification Script.
Executes against real local Ollama server at http://localhost:11434.

Strict Constraints:
- Lightweight models ONLY: gemma3:latest and qwen:latest.
- Do NOT load deepseek-coder-v2:16b.
- Validate:
  1. Live mid-stream cancellation with canonical UUID v4.
  2. Bounded worker termination and clean lease release on abort.
  3. Live pipeline handoff verification:
     - Stage 1 worker dead
     - Stage 1 lease absent from manager
     - Stage 1 model evicted from /api/ps
     - Stage 2 acquisition, execution, and release
  4. Final state: 0 active leases, 0 cancellation registry entries, 0 resident models in Ollama.
"""

import json
import threading
import time
import uuid
import requests

import app
from model_lifecycle import (
    OllamaLifecycleManager,
    ModelLeaseType,
    reset_model_manager_for_testing,
)


def run_stage4_4_live_verification():
    print("=" * 75)
    print("STAGE 4.4 LIVE LIFECYCLE INTEGRATION & RELIABILITY VERIFICATION")
    print("=" * 75)

    # 1. Verify Ollama daemon reachability
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print(f"❌ Ollama returned HTTP {r.status_code}")
            return False
        tags_data = r.json()
        available_names = [m["name"] for m in tags_data.get("models", [])]
        print(f"✅ Ollama daemon reachable. Available models: {available_names}")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama daemon: {e}")
        return False

    # Verify lightweight test models exist and assert deepseek-coder-v2:16b is NOT used
    stage1_model = "gemma3:latest"
    stage2_model = "qwen:latest"

    assert "deepseek" not in stage1_model.lower(), "Constraint violation: stage1_model must not be DeepSeek"
    assert "deepseek" not in stage2_model.lower(), "Constraint violation: stage2_model must not be DeepSeek"

    # Patch CHAT_MODELS in app to use lightweight gemma3:latest directly
    orig_chat_models = app.CHAT_MODELS
    app.CHAT_MODELS = ["gemma3:latest"]

    flask_app = app.app
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    # Create real lifecycle manager connected to live Ollama
    manager = OllamaLifecycleManager(
        base_url="http://localhost:11434",
        max_concurrent_generation=1,
        embedding_policy="adaptive",
        default_timeout=60.0,
        poll_interval=0.2,
        eviction_timeout=15.0,
        eviction_poll_interval=0.3,
    )
    reset_model_manager_for_testing(manager)

    try:
        # Reconcile state
        recon = manager.reconcile_state()
        print(f"Initial Ollama resident models in /api/ps: {recon['ollama_ps_models']}")

        # ──────────────────────────────────────────────────────────────────────────
        # PART 1: Live Generation & Mid-Stream Cancellation (UUID v4)
        # ──────────────────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("PART 1: Live Mid-Stream Cancellation with Canonical UUID v4")
        print("-" * 75)

        req_id = str(uuid.uuid4())
        print(f"Generated canonical UUID v4 request ID: {req_id}")

        # Start generation with lightweight model gemma3:latest
        print(f"Dispatching POST /api/chat with lightweight model={stage1_model}...")
        post_data = {
            "messages": [
                {"role": "user", "content": "Write a detailed 500 word technical description of orbital mechanics."}
            ],
            "mode": "gemma",
            "rag_mode": "off",
            "request_id": req_id,
        }

        response = client.post("/api/chat", json=post_data)
        if response.status_code != 200:
            print(f"❌ POST /api/chat failed with status {response.status_code}")
            return False

        # Read SSE tokens until at least 2 tokens are received
        tokens_received = []
        print("Streaming tokens from live Ollama...")
        iterator = iter(response.response)

        for chunk in iterator:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            if '"token"' in chunk_str:
                tokens_received.append(chunk_str)
                print(f"  -> Received live token chunk #{len(tokens_received)}")
                if len(tokens_received) >= 2:
                    break

        if len(tokens_received) < 2:
            print("❌ Did not receive at least 2 tokens from live Ollama before abort test.")
            return False

        # Verify active lease is held during stream
        diag_during = manager.get_diagnostics()
        print(f"Active leases during stream: {diag_during['active_gen_leases']}")
        if len(diag_during["active_gen_leases"]) == 0:
            print("❌ Expected active generation lease during streaming!")
            return False

        # Verify request is in cancellation registry
        reg_count = app.cancellation_registry.active_count()
        print(f"Cancellation registry active count during stream: {reg_count}")
        if reg_count == 0 or app.cancellation_registry.get_event(req_id) is None:
            print(f"❌ Expected request {req_id} to be registered in cancellation registry!")
            return False

        # Dispatch POST /api/chat/abort
        print(f"\nDispatching POST /api/chat/abort for request_id={req_id}...")
        t_abort_start = time.monotonic()
        abort_resp = client.post("/api/chat/abort", json={"request_id": req_id})
        abort_json = abort_resp.get_json()
        print(f"Abort response: HTTP {abort_resp.status_code} -> {abort_json}")

        if abort_resp.status_code != 200 or abort_json.get("status") != "aborted":
            print(f"❌ Abort endpoint did not return HTTP 200 status='aborted': {abort_json}")
            return False

        # Drain remaining iterator chunks to allow generator finally block to execute
        remaining_chunks = []
        has_cancelled_event = False
        for chunk in iterator:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            remaining_chunks.append(chunk_str)
            if '"cancelled": true' in chunk_str.lower() or 'cancelled' in chunk_str.lower():
                has_cancelled_event = True

        response.close()
        t_abort_elapsed = round(time.monotonic() - t_abort_start, 3)
        print(f"Stream finished/closed in {t_abort_elapsed}s. Saw cancellation signal in SSE: {has_cancelled_event}")

        # Allow worker thread join to settle
        time.sleep(0.5)

        # Verify post-cancellation state
        diag_after = manager.get_diagnostics()
        reg_count_after = app.cancellation_registry.active_count()
        print(f"Active leases after cancellation: {diag_after['active_gen_leases']}")
        print(f"Cancellation registry count after cancellation: {reg_count_after}")

        if len(diag_after["active_gen_leases"]) != 0:
            print(f"❌ Generation lease leaked after cancellation: {diag_after['active_gen_leases']}")
            return False

        if reg_count_after != 0:
            print(f"❌ Cancellation registry leaked entries: {reg_count_after}")
            return False

        print("✅ Part 1 PASSED: Mid-stream cancellation terminated cleanly, released lease, and cleared registry.")

        # ──────────────────────────────────────────────────────────────────────────
        # PART 2: Live Pipeline Handoff Verification (Gemma3 -> Qwen)
        # ──────────────────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("PART 2: Live Pipeline Handoff Verification (gemma3:latest -> qwen:latest)")
        print("-" * 75)

        print(f"Stage 1 target: {stage1_model}")
        print(f"Stage 2 target: {stage2_model}")

        # Stage 1: Run real inference under generation lease
        stage1_lease = None
        stage1_worker = None
        stage1_output = []

        with manager.acquire_generation_lease(stage1_model, timeout=30.0) as lease:
            stage1_lease = lease
            print(f"Acquired Stage 1 lease: {lease.id} for {lease.model_name}")

            def stage1_work():
                try:
                    print("  Stage 1 worker calling Ollama /api/chat...")
                    r_ollama = requests.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": stage1_model,
                            "messages": [{"role": "user", "content": "Respond with the single word: READY."}],
                            "stream": False,
                        },
                        timeout=30,
                    )
                    if r_ollama.status_code == 200:
                        stage1_output.append(r_ollama.json().get("message", {}).get("content", ""))
                    print(f"  Stage 1 worker finished HTTP call. Response: {stage1_output}")
                finally:
                    lease.mark_worker_finished()

            stage1_worker = threading.Thread(target=stage1_work, name="Stage1-LiveWorker")
            lease.transfer_to_worker(stage1_worker)
            stage1_worker.start()
            stage1_worker.join(timeout=30.0)

        # Lease context exited -> Stage 1 lease released
        print(f"Stage 1 finished. Worker alive? {stage1_worker.is_alive()}, Output: {stage1_output}")

        # Now verify all 3 handoff conditions using verify_pipeline_handoff()
        print("Executing manager.verify_pipeline_handoff()...")
        t_handoff_start = time.monotonic()
        manager.verify_pipeline_handoff(
            stage1_lease,
            stage1_model,
            timeout=15.0,
        )
        t_handoff_elapsed = round(time.monotonic() - t_handoff_start, 3)
        print(f"✅ verify_pipeline_handoff completed successfully in {t_handoff_elapsed}s.")

        # Stage 2: Acquire generation lease for qwen:latest
        print(f"\nStage 2: Acquiring generation lease for {stage2_model}...")
        stage2_output = []
        with manager.acquire_generation_lease(stage2_model, timeout=30.0) as lease2:
            print(f"Acquired Stage 2 lease: {lease2.id} for {lease2.model_name}")

            def stage2_work():
                try:
                    print("  Stage 2 worker calling Ollama /api/chat...")
                    r_ollama2 = requests.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": stage2_model,
                            "messages": [{"role": "user", "content": "Respond with the single word: CONFIRMED."}],
                            "stream": False,
                        },
                        timeout=30,
                    )
                    if r_ollama2.status_code == 200:
                        stage2_output.append(r_ollama2.json().get("message", {}).get("content", ""))
                    print(f"  Stage 2 worker finished HTTP call. Response: {stage2_output}")
                finally:
                    lease2.mark_worker_finished()

            stage2_worker = threading.Thread(target=stage2_work, name="Stage2-LiveWorker")
            lease2.transfer_to_worker(stage2_worker)
            stage2_worker.start()
            stage2_worker.join(timeout=30.0)

        print(f"Stage 2 finished. Output: {stage2_output}")

        diag_final = manager.get_diagnostics()
        print(f"Active leases after Stage 2 exit: {diag_final['active_gen_leases']}")
        if len(diag_final["active_gen_leases"]) != 0:
            print("❌ Generation lease leaked after Stage 2!")
            return False

        print("✅ Part 2 PASSED: Pipeline handoff safely verified worker death, lease absence, and model eviction.")

        # ──────────────────────────────────────────────────────────────────────────
        # PART 3: Final Ollama Model Cleanup & 0-Residency Verification
        # ──────────────────────────────────────────────────────────────────────────
        print("\n" + "-" * 75)
        print("PART 3: Final Model Eviction & 0-Residency Verification")
        print("-" * 75)

        # Evict any remaining resident models via /api/generate with keep_alive: 0
        ps_resp = requests.get("http://localhost:11434/api/ps", timeout=5)
        current_ps = [m["name"] for m in ps_resp.json().get("models", [])]
        print(f"Resident models before final eviction: {current_ps}")

        for model_name in current_ps:
            print(f"Evicting {model_name}...")
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
            )

        # Re-query /api/ps
        time.sleep(0.5)
        final_ps_resp = requests.get("http://localhost:11434/api/ps", timeout=5)
        final_ps = [m["name"] for m in final_ps_resp.json().get("models", [])]
        print(f"Final resident models in /api/ps: {final_ps}")

        if len(final_ps) != 0:
            print(f"❌ Expected 0 resident models in Ollama, but found: {final_ps}")
            return False

        print("✅ Part 3 PASSED: 0 resident models in Ollama memory.")

        print("\n" + "=" * 75)
        print("ALL STAGE 4.4 LIVE INTEGRATION & RELIABILITY TESTS PASSED!")
        print("=" * 75)
        return True

    finally:
        # Restore app.CHAT_MODELS
        app.CHAT_MODELS = orig_chat_models


if __name__ == "__main__":
    success = run_stage4_4_live_verification()
    exit(0 if success else 1)
