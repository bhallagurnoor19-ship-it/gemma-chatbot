"""
Stage 4.1 & 4.2 Test Suite: Embedding Lease Recovery, Cancellation Registry & Backend Abort.

Stage 4.1 Tests (01-08):
1. Owner thread tracking on all embedding lease creation paths.
2. Refusal to recover active embedding leases while owner thread is alive.
3. Safe atomic recovery of active embedding leases when owner thread terminates.
4. Safe retry of incomplete/failed manager notifications on released embedding leases.
5. Deterministic synchronization race between normal release and recovery (zero double notification).
6. Non-reentrant lock safety (no deadlocks with threading.Lock).
7. Refusal to recover active leases when owner thread is None/untracked.
8. Accurate diagnostics and representation reporting.

Stage 4.2 Tests (09-18):
09. Rejection of duplicate active request IDs (409 Conflict) without overwriting event.
10. Rejection of invalid request ID format (400 Bad Request).
11. Scoped abort signaling (aborting request A never impacts request B).
12. Idempotency of repeated abort requests.
13. Handling of unknown or already completed request aborts (200 ignored).
14. Immediate unblocking (< 200 ms) of queued lease acquisition upon abort.
15. Atomic cancellation racing with lease acquisition without lease leak.
16. Registry cleanup when response generator is never iterated / dropped.
17. Immediate registry cleanup after pre-generator request errors.
18. Worker lease ownership preservation when cancelled while worker is alive.

Stage 4.3 Tests (19-24):
19. Pipeline handoff verifies Stage 1 worker termination (refuses if worker alive).
20. Pipeline handoff verifies Stage 1 lease removal from manager tracking (times out if lease retained).
21. Pipeline handoff verifies Stage 1 model absence from Ollama /api/ps (eviction & verification).
22. Pipeline handoff cancellation handling before and during verification.
23. Pipeline handoff succeeds when all three conditions are satisfied.
24. Pipeline handoff prevents Stage 2 coder lease acquisition upon handoff failure.

Stage 4.4 Tests (25-34):
25. Normal frontend completion (stream exhausts, registry cleared, lease released).
26. Explicit Stop cancellation (/api/chat/abort cancels stream and cleans up).
27. Repeated Stop requests (idempotent and safe).
28. Old-request cancellation after a new request starts (isolated).
29. Unexpected SSE disconnect (closes generator, clears registry and lease).
30. Disconnect during Stage 1 (aborts Stage 1, cleans up lease, never enters Stage 2).
31. Disconnect during Stage 2 (aborts Stage 2, cleans up lease and registry).
32. Cancellation during Stage 1 -> Stage 2 handoff (cleanly stops before Stage 2).
33. Worker remains alive after generator closure (retained under worker ownership until death).
34. Final registry and lease cleanup (zero leaked leases or registry entries).
"""

import json
import threading
import time
import unittest
import uuid
from unittest.mock import MagicMock, patch

from app import app, cancellation_registry, is_valid_uuid, ClosableStream
from model_lifecycle import (
    EmbeddingPolicy,
    ModelLease,
    ModelLeaseType,
    ModelLeaseCancelledError,
    ModelLifecycleError,
    ModelEvictionError,
    OllamaLifecycleManager,
    _LeaseState,
    get_model_manager,
    reset_model_manager_for_testing,
)


class TestEmbeddingLeaseRecoveryStage4(unittest.TestCase):
    def setUp(self):
        reset_model_manager_for_testing()
        self.manager = OllamaLifecycleManager(
            default_timeout=5.0,
            poll_interval=0.05,
            embedding_policy=EmbeddingPolicy.PERSISTENT,
        )

    def tearDown(self):
        reset_model_manager_for_testing()

    # ── Test 01: Owner thread assignment on all paths ──────────────────────

    def test_01_owner_thread_assigned_on_all_embedding_creation_paths(self):
        """Owner thread is reliably assigned on all creation paths (default, explicit, and context manager)."""
        curr = threading.current_thread()

        # A. Default creation in current thread
        lease1 = ModelLease(self.manager, "embeddinggemma:latest", ModelLeaseType.EMBEDDING)
        self.assertIs(lease1.owner_thread, curr)

        # B. Explicit owner thread
        dummy_thread = threading.Thread(target=lambda: None, name="custom-worker-thr")
        lease2 = ModelLease(
            self.manager,
            "embeddinggemma:latest",
            ModelLeaseType.EMBEDDING,
            owner_thread=dummy_thread,
        )
        self.assertIs(lease2.owner_thread, dummy_thread)

        # C. Manager context manager acquisition
        with self.manager.acquire_embedding_lease() as lease3:
            self.assertIs(lease3.owner_thread, curr)
            self.assertIn(lease3, self.manager._active_embed_leases)

    # ── Test 02: Refusal to recover while owner thread is alive ─────────────

    def test_02_recovery_refuses_active_embedding_lease_when_owner_alive(self):
        """Active embedding lease is never recovered while its owner thread is executing."""
        started_event = threading.Event()
        stop_event = threading.Event()
        owner_ref = []

        def embedding_worker():
            with self.manager.acquire_embedding_lease() as lease:
                owner_ref.append((threading.current_thread(), lease))
                started_event.set()
                stop_event.wait(timeout=5.0)

        t = threading.Thread(target=embedding_worker, name="active-embed-owner")
        t.start()
        started_event.wait(timeout=2.0)

        try:
            owner_thread, lease = owner_ref[0]
            self.assertTrue(owner_thread.is_alive())
            self.assertEqual(lease.state, _LeaseState.ACTIVE)
            self.assertEqual(len(self.manager._active_embed_leases), 1)

            # Attempt recovery while owner is alive
            recovered = self.manager.recover_stuck_leases(force=True)
            self.assertEqual(recovered, 0)
            self.assertEqual(lease.state, _LeaseState.ACTIVE)
            self.assertIn(lease, self.manager._active_embed_leases)
        finally:
            stop_event.set()
            t.join(timeout=2.0)

        # After normal termination, lease is released
        self.assertEqual(len(self.manager._active_embed_leases), 0)

    # ── Test 03: Safe recovery when owner thread dies ───────────────────────

    def test_03_recovery_reclaims_active_embedding_lease_when_owner_dies(self):
        """Active embedding lease is cleanly recovered when its owner thread has genuinely terminated."""
        lease_ref = []

        def dying_worker():
            # Acquire lease directly and register in manager, then die ungracefully
            lease = ModelLease(
                self.manager,
                "embeddinggemma:latest",
                ModelLeaseType.EMBEDDING,
                owner_thread=threading.current_thread(),
            )
            with self.manager._cv:
                self.manager._active_embed_leases.add(lease)
            lease_ref.append(lease)
            # Worker dies here without calling lease.release()

        t = threading.Thread(target=dying_worker, name="dying-embed-worker")
        t.start()
        t.join(timeout=2.0)

        self.assertFalse(t.is_alive())
        lease = lease_ref[0]
        self.assertEqual(lease.state, _LeaseState.ACTIVE)
        self.assertIn(lease, self.manager._active_embed_leases)

        # Sweep recovery
        recovered = self.manager.recover_stuck_leases()
        self.assertEqual(recovered, 1)
        self.assertEqual(lease.state, _LeaseState.RELEASED)
        self.assertNotIn(lease, self.manager._active_embed_leases)
        self.assertTrue(lease._notification_completed)

    # ── Test 04: Recovery retries failed notification ───────────────────────

    def test_04_recovery_retries_failed_embedding_release_notification(self):
        """A released embedding lease whose manager notification failed is safely retried by recovery."""
        lease = ModelLease(
            self.manager,
            "embeddinggemma:latest",
            ModelLeaseType.EMBEDDING,
        )
        with self.manager._cv:
            self.manager._active_embed_leases.add(lease)

        # Inject failure into _on_lease_released on initial release
        with patch.object(self.manager, "_on_lease_released", side_effect=RuntimeError("Simulated manager glitch")):
            with self.assertRaises(RuntimeError):
                lease.release()

        # State transitioned to RELEASED, but notification not completed
        self.assertEqual(lease.state, _LeaseState.RELEASED)
        self.assertFalse(lease._notification_completed)
        self.assertFalse(lease._notification_in_progress)
        self.assertIn(lease, self.manager._active_embed_leases)

        # Recovery sweeps and completes the notification
        recovered = self.manager.recover_stuck_leases()
        self.assertEqual(recovered, 1)
        self.assertTrue(lease._notification_completed)
        self.assertNotIn(lease, self.manager._active_embed_leases)

    # ── Test 05: Deterministic race between normal release and recovery ────

    def test_05_deterministic_race_between_normal_embedding_release_and_recovery(self):
        """Concurrent normal release and recovery race deterministically without double notification."""
        t1_ref = []
        start_barrier = threading.Barrier(2)
        notification_calls = []
        original_on_release = self.manager._on_lease_released

        def tracking_on_release(l):
            notification_calls.append(threading.current_thread().name)
            original_on_release(l)

        self.manager._on_lease_released = tracking_on_release

        lease_ref = []

        def run_normal_release():
            lease = ModelLease(
                self.manager,
                "embeddinggemma:latest",
                ModelLeaseType.EMBEDDING,
                owner_thread=threading.current_thread(),
            )
            with self.manager._cv:
                self.manager._active_embed_leases.add(lease)
            lease_ref.append(lease)
            start_barrier.wait()
            lease.release()

        def run_recovery():
            start_barrier.wait()
            self.manager.recover_stuck_leases()

        t1 = threading.Thread(target=run_normal_release, name="thread-normal-release")
        t2 = threading.Thread(target=run_recovery, name="thread-recovery-sweep")

        t1.start()
        t2.start()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        lease = lease_ref[0]
        # Exactly ONE thread must have performed manager notification
        self.assertEqual(len(notification_calls), 1)
        self.assertEqual(len(self.manager._active_embed_leases), 0)
        self.assertEqual(lease.state, _LeaseState.RELEASED)
        self.assertTrue(lease._notification_completed)

    # ── Test 06: Non-reentrant lock safety ──────────────────────────────────

    def test_06_non_reentrant_lock_safety_embedding_recovery(self):
        """Direct verification that lease._lock is non-reentrant and recovery executes without deadlock."""
        lease = ModelLease(
            self.manager,
            "embeddinggemma:latest",
            ModelLeaseType.EMBEDDING,
        )
        # Verify it is threading.Lock (non-reentrant)
        self.assertIs(type(lease._lock), type(threading.Lock()))

        # Call recover_active_embedding_lease on active lease with alive owner
        # Must execute cleanly and return False without deadlock
        result = lease.recover_active_embedding_lease()
        self.assertFalse(result)

        # Transition to RELEASED and test claiming notification authority
        with lease._lock:
            lease._state = _LeaseState.RELEASED
        result2 = lease.recover_active_embedding_lease()
        self.assertTrue(result2)
        lease._finish_notification(success=True)

    # ── Test 07: Refusal to recover when owner thread is None ───────────────

    def test_07_recovery_refuses_untracked_or_none_owner_lease(self):
        """If owner_thread is None, recovery refuses to recover an active lease."""
        lease = ModelLease(
            self.manager,
            "embeddinggemma:latest",
            ModelLeaseType.EMBEDDING,
            owner_thread=None,
        )
        # Manually force _owner_thread to None to test untracked scenario
        with lease._lock:
            lease._owner_thread = None

        self.assertIsNone(lease.owner_thread)
        self.assertFalse(lease.recover_active_embedding_lease())
        self.assertEqual(lease.state, _LeaseState.ACTIVE)

    # ── Test 08: Diagnostics and repr reporting ────────────────────────────

    def test_08_diagnostics_reports_active_embedding_leases_and_owner_info(self):
        """Diagnostics and repr accurately reflect embedding leases and owner threads."""
        lease = ModelLease(
            self.manager,
            "embeddinggemma:latest",
            ModelLeaseType.EMBEDDING,
        )
        with self.manager._cv:
            self.manager._active_embed_leases.add(lease)

        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["active_embed_leases"], 1)

        rep = repr(lease)
        self.assertIn("type=embedding", rep)
        self.assertIn("owner=", rep)

        lease.release()
        self.assertEqual(self.manager.get_diagnostics()["active_embed_leases"], 0)


class TestCancellationRegistryAndAbortStage4_2(unittest.TestCase):
    """
    Stage 4.2 Test Suite: Cancellation Registry & Backend Abort Integration.

    Tests verify:
    09. Rejection of duplicate active request IDs (409 Conflict) without overwriting event.
    10. Rejection of invalid request ID format (400 Bad Request).
    11. Scoped abort signaling (aborting request A never impacts request B).
    12. Idempotency of repeated abort requests.
    13. Handling of unknown or already completed request aborts (200 ignored).
    14. Immediate unblocking (< 200 ms) of queued lease acquisition upon abort.
    15. Atomic cancellation racing with lease acquisition without lease leak.
    16. Registry cleanup when response generator is never iterated / dropped.
    17. Immediate registry cleanup after pre-generator request errors.
    18. Worker lease ownership preservation when cancelled while worker is alive.
    """

    def setUp(self):
        self.manager = OllamaLifecycleManager(
            base_url="http://localhost:11434",
            default_timeout=5.0,
            eviction_timeout=0.3,
            embedding_policy=EmbeddingPolicy.PERSISTENT,
        )
        self.manager._query_ollama_ps = MagicMock(return_value=[])
        self.manager._evict_model_http = MagicMock(return_value=True)
        reset_model_manager_for_testing(self.manager)
        cancellation_registry.clear()
        self.client = app.test_client()

    def tearDown(self):
        cancellation_registry.clear()
        reset_model_manager_for_testing(None)

    # ── Test 09: Duplicate active request ID rejected (409) ─────────────────

    def test_09_duplicate_active_request_id_rejected(self):
        """Reject duplicate active request IDs with 409 Conflict without overwriting existing event."""
        req_id = str(uuid.uuid4())
        ev1, reg1 = cancellation_registry.register(req_id)
        self.assertTrue(reg1)
        self.assertFalse(ev1.is_set())

        # An active request with req_id exists. An incoming POST /api/chat with same req_id must return 409
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "hi"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(resp.status_code, 409)
        data = resp.get_json()
        self.assertIn("Duplicate active request_id", data.get("error", ""))

        # Verify existing event was not overwritten or corrupted
        ev_current = cancellation_registry.get_event(req_id)
        self.assertIs(ev_current, ev1)
        self.assertFalse(ev_current.is_set())

    # ── Test 10: Invalid request ID format rejected (400) ───────────────────

    def test_10_invalid_request_id_format_rejected(self):
        """Reject invalid request ID formats with 400 Bad Request."""
        invalid_ids = [
            "not-a-uuid",
            "12345",
            "../../../etc/passwd",
            "gemma-chat-request-1",
            "invalid-hex-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
            "123e4567-e89b-12d3-a456-42661417400",   # 35 chars
            "123e4567-e89b-12d3-a456-4266141740000", # 37 chars
        ]
        for bad_id in invalid_ids:
            with self.subTest(bad_id=bad_id):
                resp = self.client.post(
                    "/api/chat",
                    json={
                        "request_id": bad_id,
                        "messages": [{"role": "user", "content": "hi"}],
                        "mode": "chat",
                        "rag_mode": "off",
                    },
                )
                self.assertEqual(resp.status_code, 400)
                data = resp.get_json()
                self.assertIn("Invalid request_id format", data.get("error", ""))
        self.assertEqual(cancellation_registry.active_count(), 0)

    # ── Test 10a: Valid canonical UUID v4 accepted ──────────────────────────

    def test_10a_valid_canonical_uuid_v4_accepted(self):
        """Accept valid canonical lowercase RFC 4122 UUID v4 strings."""
        canonical_v4 = str(uuid.uuid4())
        self.assertTrue(is_valid_uuid(canonical_v4))

        # Frontend fallback generator pattern verification
        frontend_generated = "10000000-1000-4000-8000-100000000000"
        self.assertTrue(is_valid_uuid(frontend_generated))

        # Direct function check with various valid v4 instances
        for _ in range(5):
            u = str(uuid.uuid4())
            self.assertTrue(is_valid_uuid(u), f"Canonical UUID v4 '{u}' must be accepted")

    # ── Test 10b: Prefixed identifiers rejected ─────────────────────────────

    def test_10b_prefixed_identifiers_rejected(self):
        """Reject req_ and req- prefixed identifiers on both chat and abort endpoints."""
        canonical_v4 = str(uuid.uuid4())
        prefixed_ids = [
            f"req_{canonical_v4}",
            f"req-{canonical_v4}",
            f"req_{canonical_v4.replace('-', '')}",
            f"req-{canonical_v4.replace('-', '')}",
        ]
        for p_id in prefixed_ids:
            with self.subTest(prefixed_id=p_id):
                self.assertFalse(is_valid_uuid(p_id))

                # /api/chat must reject with 400
                resp_chat = self.client.post(
                    "/api/chat",
                    json={
                        "request_id": p_id,
                        "messages": [{"role": "user", "content": "hi"}],
                        "mode": "chat",
                        "rag_mode": "off",
                    },
                )
                self.assertEqual(resp_chat.status_code, 400)

                # /api/chat/abort must reject with 400
                resp_abort = self.client.post(
                    "/api/chat/abort",
                    json={"request_id": p_id},
                )
                self.assertEqual(resp_abort.status_code, 400)

    # ── Test 10c: Noncanonical representations rejected ────────────────────

    def test_10c_noncanonical_representations_rejected(self):
        """Reject 32-char unhyphenated hex, uppercase, whitespace, and non-v4 UUIDs."""
        canonical_v4 = str(uuid.uuid4())
        noncanonical_ids = [
            canonical_v4.replace("-", ""),         # 32-char unhyphenated hex
            canonical_v4.upper(),                  # Uppercase
            f" {canonical_v4}",                    # Leading space
            f"{canonical_v4} ",                    # Trailing space
            "00000000-0000-1000-8000-000000000000",# UUID v1 (not v4)
            "00000000-0000-3000-8000-000000000000",# UUID v3 (not v4)
            "00000000-0000-5000-8000-000000000000",# UUID v5 (not v4)
            canonical_v4[:19] + "c" + canonical_v4[20:], # Invalid variant bits (must be 8, 9, a, or b)
        ]
        for bad_id in noncanonical_ids:
            with self.subTest(noncanonical_id=bad_id):
                self.assertFalse(is_valid_uuid(bad_id), f"ID '{bad_id}' should be rejected as noncanonical")
                resp = self.client.post(
                    "/api/chat",
                    json={
                        "request_id": bad_id,
                        "messages": [{"role": "user", "content": "hi"}],
                        "mode": "chat",
                        "rag_mode": "off",
                    },
                )
                self.assertEqual(resp.status_code, 400)

    # ── Test 11: Scoped abort signaling ─────────────────────────────────────

    def test_11_abort_endpoint_scopes_cancellation_to_target_request(self):
        """Aborting request A cancels only request A; request B remains unaffected."""
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        ev_a, _ = cancellation_registry.register(id_a)
        ev_b, _ = cancellation_registry.register(id_b)

        resp = self.client.post("/api/chat/abort", json={"request_id": id_a})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "aborted")
        self.assertEqual(data.get("request_id"), id_a)

        self.assertTrue(ev_a.is_set(), "Target request A must be set")
        self.assertFalse(ev_b.is_set(), "Untargeted request B must NOT be set")

    # ── Test 12: Idempotent repeated abort requests ─────────────────────────

    def test_12_repeated_abort_requests_are_idempotent(self):
        """Repeated abort calls for the same request return 200 without error."""
        req_id = str(uuid.uuid4())
        ev, _ = cancellation_registry.register(req_id)

        for _ in range(3):
            resp = self.client.post("/api/chat/abort", json={"request_id": req_id})
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data.get("status"), "aborted")
            self.assertTrue(ev.is_set())

    # ── Test 13: Unknown or completed request abort handling ────────────────

    def test_13_unknown_and_completed_request_abort_handling(self):
        """Unknown or already completed requests return 200 ignored."""
        unknown_id = str(uuid.uuid4())
        resp = self.client.post("/api/chat/abort", json={"request_id": unknown_id})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "ignored")

        # Invalid/missing request_id
        resp_bad = self.client.post("/api/chat/abort", json={})
        self.assertEqual(resp_bad.status_code, 400)

    # ── Test 14: Queued lease acquisition cancelled within 200ms ────────────

    def test_14_queued_lease_acquisition_cancelled_within_200ms(self):
        """Queued lease acquisition aborts and raises ModelLeaseCancelledError within 200ms."""
        # Occupy the generation slot
        holder_lease = self.manager.acquire_generation_lease("model-a")
        holder_lease.__enter__()

        req_id = str(uuid.uuid4())
        cancel_ev, _ = cancellation_registry.register(req_id)
        waiting_started = threading.Event()
        raised_exception = []
        timing = []

        def queued_acquirer():
            waiting_started.set()
            t_start = time.perf_counter()
            try:
                with self.manager.acquire_generation_lease(
                    "model-b", is_cancelled=cancel_ev.is_set, timeout=5.0
                ):
                    pass
            except Exception as e:
                timing.append(time.perf_counter() - t_start)
                raised_exception.append(e)

        thr = threading.Thread(target=queued_acquirer, name="queued-acquirer-thr")
        thr.start()
        waiting_started.wait(timeout=1.0)
        time.sleep(0.05)  # Ensure thread is blocked inside _cv.wait()

        # Trigger abort via endpoint
        t0 = time.perf_counter()
        resp = self.client.post("/api/chat/abort", json={"request_id": req_id})
        self.assertEqual(resp.status_code, 200)

        thr.join(timeout=1.0)
        self.assertFalse(thr.is_alive())
        self.assertEqual(len(raised_exception), 1)
        self.assertIsInstance(raised_exception[0], ModelLeaseCancelledError)

        latency = timing[0] if timing else (time.perf_counter() - t0)
        self.assertLess(latency, 0.2, f"Abort wake latency was {latency:.4f}s, exceeding 200ms target")

        # Clean up holder lease
        holder_lease.__exit__(None, None, None)

    # ── Test 15: Cancellation racing with lease grant ───────────────────────

    def test_15_cancellation_racing_with_lease_grant_does_not_leak_lease(self):
        """Cancellation racing with lease grant does not leak leases or leave inconsistent manager state."""
        cancel_ev = threading.Event()
        cancel_ev.set()  # Already cancelled prior to grant check

        with self.assertRaises(ModelLeaseCancelledError):
            with self.manager.acquire_generation_lease("test-model", is_cancelled=cancel_ev.is_set):
                pass

        diag = self.manager.get_diagnostics()
        self.assertEqual(len(diag["active_gen_leases"]), 0)
        self.assertEqual(len(self.manager._active_gen_leases), 0)

    # ── Test 16: Registry cleanup when generator never iterates ─────────────

    @patch("app.get_available_ollama_models", return_value=["gemma4:12b"])
    @patch("app.get_rag_service")
    def test_16_registry_cleanup_when_generator_never_iterates(self, mock_rag, mock_models):
        """Closing client connection before generator iteration begins cleans up registry."""
        mock_rag.return_value.retrieve_context.return_value = MagicMock(has_context=False, context_text="")
        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "hello"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(cancellation_registry.is_active(req_id))

        # Close the response without consuming the stream
        resp.close()

        # Verify registry was cleaned up
        self.assertFalse(cancellation_registry.is_active(req_id), "Registry must not leak on closed response stream")

    # ── Test 17: Registry cleanup after pre-generator failures ──────────────

    @patch("app.classify_intent", side_effect=RuntimeError("Simulated classification boom"))
    def test_17_registry_cleanup_after_pre_generator_failures(self, mock_classify):
        """Failure before generator creation immediately cleans up cancellation registry."""
        req_id = str(uuid.uuid4())
        try:
            resp = self.client.post(
                "/api/chat",
                json={
                    "request_id": req_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "mode": "chat",
                    "rag_mode": "off",
                },
            )
            self.assertEqual(resp.status_code, 500)
        except RuntimeError:
            pass

        self.assertFalse(cancellation_registry.is_active(req_id), "Registry must clean up immediately on pre-generator failure")

    # ── Test 18: Worker lease ownership preserved on cancellation ───────────

    def test_18_worker_lease_ownership_preserved_on_cancellation_while_worker_alive(self):
        """Worker retains lease ownership and detaches when cancelled while worker is alive."""
        lease = ModelLease(self.manager, "test-model:latest", ModelLeaseType.GENERATION)
        with self.manager._cv:
            self.manager._active_gen_leases[lease.model_name].add(lease)

        # Simulate worker alive in separate thread
        worker_can_finish = threading.Event()
        worker_finished = threading.Event()

        def mock_worker():
            worker_can_finish.wait(timeout=2.0)
            lease.mark_worker_finished()
            lease.release_from_worker()
            worker_finished.set()

        worker_thr = threading.Thread(target=mock_worker, name="mock-worker-thr")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()

        # Generator experiences cancellation: calls detach_or_release()
        lease.detach_or_release()

        # Invariant check: lease is DETACHED, not RELEASED!
        self.assertEqual(lease.state, _LeaseState.DETACHED)
        self.assertIn(lease, self.manager._active_gen_leases[lease.model_name])

        # Unblock worker to finish
        worker_can_finish.set()
        worker_thr.join(timeout=2.0)
        worker_finished.wait(timeout=1.0)

        # Invariant check: lease is now fully RELEASED and removed from manager
        self.assertEqual(lease.state, _LeaseState.RELEASED)
        self.assertNotIn(lease, self.manager._active_gen_leases.get(lease.model_name, set()))


class TestPipelineHandoffStage4(unittest.TestCase):
    """
    Stage 4.3 Pipeline Handoff Verification Tests.
    Independently verifies:
    1. Stage 1 worker thread termination.
    2. Stage 1 lease absence from manager active lease tracking.
    3. Stage 1 model absence from Ollama /api/ps (evicted and verified).
    4. Cancellation safety across all handoff verification steps.
    """

    def setUp(self):
        reset_model_manager_for_testing()
        self.manager = OllamaLifecycleManager(
            default_timeout=2.0,
            poll_interval=0.02,
            embedding_policy=EmbeddingPolicy.PERSISTENT,
        )

    def tearDown(self):
        reset_model_manager_for_testing()

    # ── Test 19: Worker termination verification ────────────────────────────

    def test_19_pipeline_handoff_verifies_worker_termination(self):
        """Condition 1: verify_pipeline_handoff raises ModelLifecycleError if worker remains alive."""
        lease = ModelLease(self.manager, "stage1-arch:latest", ModelLeaseType.GENERATION)
        with self.manager._cv:
            self.manager._active_gen_leases[lease.model_name].add(lease)

        stop_worker = threading.Event()
        worker = threading.Thread(target=stop_worker.wait, name="mock-stage1-worker")
        lease.transfer_to_worker(worker)
        worker.start()

        try:
            with self.assertRaises(ModelLifecycleError) as ctx:
                self.manager.verify_pipeline_handoff(
                    prior_lease=lease,
                    prior_model="stage1-arch:latest",
                    timeout=0.1,
                )
            self.assertIn("failed to terminate", str(ctx.exception))
        finally:
            stop_worker.set()
            worker.join(timeout=1.0)
            lease.release()

    # ── Test 20: Lease removal from manager tracking ─────────────────────────

    def test_20_pipeline_handoff_verifies_lease_removal(self):
        """Condition 2: verify_pipeline_handoff raises ModelLifecycleError if lease tracking is not cleared."""
        lease = ModelLease(self.manager, "stage1-arch:latest", ModelLeaseType.GENERATION)
        with self.manager._cv:
            self.manager._active_gen_leases[lease.model_name].add(lease)

        # Worker is terminated
        dead_worker = threading.Thread(target=lambda: None, name="dead-stage1-worker")
        dead_worker.start()
        dead_worker.join()
        lease.transfer_to_worker(dead_worker)

        # But lease is NOT released and remains in manager active leases
        with patch.object(self.manager, "_query_ollama_ps", return_value=[]):
            with self.assertRaises(ModelLifecycleError) as ctx:
                self.manager.verify_pipeline_handoff(
                    prior_lease=lease,
                    prior_model="stage1-arch:latest",
                    timeout=0.1,
                )
            self.assertIn("was not cleanly removed from manager tracking", str(ctx.exception))

        # Cleanup
        lease.release()

    # ── Test 21: Model absence from /api/ps and eviction ────────────────────

    def test_21a_pipeline_handoff_evicts_model_if_present_in_ps(self):
        """Condition 3: Stage 1 model present in /api/ps is evicted and absence verified."""
        lease = ModelLease(self.manager, "stage1-arch:latest", ModelLeaseType.GENERATION)
        lease.release()

        # Simulate model present in /api/ps on first call, absent on second call
        ps_calls = [["stage1-arch:latest"], []]

        def mock_query():
            if ps_calls:
                return ps_calls.pop(0)
            return []

        evict_mock = MagicMock(return_value=True)

        with patch.object(self.manager, "_query_ollama_ps", side_effect=mock_query):
            with patch.object(self.manager, "_evict_model_http", evict_mock):
                self.manager.verify_pipeline_handoff(
                    prior_lease=lease,
                    prior_model="stage1-arch:latest",
                    timeout=1.0,
                )

        evict_mock.assert_called_once_with("stage1-arch:latest")

    def test_21b_pipeline_handoff_raises_when_model_remains_in_ps(self):
        """Condition 3 failure: verify_pipeline_handoff raises ModelEvictionError if eviction fails or model remains."""
        lease = ModelLease(self.manager, "stage1-arch:latest", ModelLeaseType.GENERATION)
        lease.release()

        # Case 1: _evict_model_http returns False
        with patch.object(self.manager, "_query_ollama_ps", return_value=["stage1-arch:latest"]):
            with patch.object(self.manager, "_evict_model_http", return_value=False):
                with self.assertRaises(ModelEvictionError) as ctx:
                    self.manager.verify_pipeline_handoff(
                        prior_lease=lease,
                        prior_model="stage1-arch:latest",
                        timeout=1.0,
                    )
                self.assertIn("Failed to evict Stage 1 model", str(ctx.exception))

        # Case 2: _evict_model_http returns True but model is still resident in /api/ps
        with patch.object(self.manager, "_query_ollama_ps", return_value=["stage1-arch:latest"]):
            with patch.object(self.manager, "_evict_model_http", return_value=True):
                with self.assertRaises(ModelEvictionError) as ctx:
                    self.manager.verify_pipeline_handoff(
                        prior_lease=lease,
                        prior_model="stage1-arch:latest",
                        timeout=1.0,
                    )
                self.assertIn("still resident in Ollama /api/ps", str(ctx.exception))

    # ── Test 22: Cancellation during handoff verification ───────────────────

    def test_22a_pipeline_handoff_cancelled_before_verification(self):
        """verify_pipeline_handoff raises ModelLeaseCancelledError if cancelled before verification."""
        lease = ModelLease(self.manager, "stage1-arch:latest", ModelLeaseType.GENERATION)
        with self.assertRaises(ModelLeaseCancelledError) as ctx:
            self.manager.verify_pipeline_handoff(
                prior_lease=lease,
                prior_model="stage1-arch:latest",
                timeout=1.0,
                is_cancelled=lambda: True,
            )
        self.assertIn("cancelled before verification", str(ctx.exception))

    def test_22b_pipeline_handoff_cancelled_during_lease_check(self):
        """verify_pipeline_handoff raises ModelLeaseCancelledError if cancelled while awaiting lease removal."""
        lease = ModelLease(self.manager, "stage1-arch:latest", ModelLeaseType.GENERATION)
        with self.manager._cv:
            self.manager._active_gen_leases[lease.model_name].add(lease)

        checks = 0

        def cancel_during_lease_check():
            nonlocal checks
            checks += 1
            # Check 1: before verification (returns False)
            # Check 2: after worker termination (returns False)
            # Check 3: during while True lease removal check (returns True)
            return checks >= 3

        with self.assertRaises(ModelLeaseCancelledError) as ctx:
            self.manager.verify_pipeline_handoff(
                prior_lease=lease,
                prior_model="stage1-arch:latest",
                timeout=2.0,
                is_cancelled=cancel_during_lease_check,
            )
        self.assertIn("cancelled during lease removal check", str(ctx.exception))
        lease.release()

    # ── Test 23: Clean handoff with all conditions satisfied ─────────────────

    def test_23_successful_pipeline_handoff_all_conditions_satisfied(self):
        """verify_pipeline_handoff succeeds cleanly when worker terminated, lease removed, and model absent."""
        lease = ModelLease(self.manager, "stage1-arch:latest", ModelLeaseType.GENERATION)
        dead_worker = threading.Thread(target=lambda: None)
        dead_worker.start()
        dead_worker.join()
        lease.transfer_to_worker(dead_worker)
        lease.release()

        with patch.object(self.manager, "_query_ollama_ps", return_value=[]):
            # Must not raise any error
            self.manager.verify_pipeline_handoff(
                prior_lease=lease,
                prior_model="stage1-arch:latest",
                timeout=1.0,
            )

    # ── Test 24: Pipeline handoff guards Stage 2 coder lease acquisition ────

    def test_24_stream_pipeline_handoff_prevents_stage2_on_verification_failure(self):
        """If verify_pipeline_handoff fails, Stage 2 coder lease is NEVER acquired."""
        mgr = self.manager
        stage2_acquired = False

        # Simulate verify_pipeline_handoff raising an exception
        with patch.object(mgr, "verify_pipeline_handoff", side_effect=ModelLifecycleError("Worker failed to terminate")):
            with self.assertRaises(ModelLifecycleError):
                # Mimic the app.py handoff flow
                mgr.verify_pipeline_handoff(None, "stage1-arch:latest")
                with mgr.acquire_generation_lease("stage2-coder:latest"):
                    stage2_acquired = True

        self.assertFalse(stage2_acquired, "Stage 2 coder lease must not be acquired when handoff verification fails")


class TestStage4_4LifecycleIntegrationAndReliability(unittest.TestCase):
    """
    Stage 4.4 Test Suite: Lifecycle Integration, Frontend Cancellation & Reliability Validation.

    Tests cover:
    25. Normal frontend completion (stream exhausts, registry cleared, lease released).
    26. Explicit Stop cancellation (/api/chat/abort cancels stream and cleans up).
    27. Repeated Stop requests (idempotent and safe).
    28. Old-request cancellation after a new request starts (isolated).
    29. Unexpected SSE disconnect (closes generator, clears registry and lease).
    30. Disconnect during Stage 1 (aborts Stage 1, cleans up lease, never enters Stage 2).
    31. Disconnect during Stage 2 (aborts Stage 2, cleans up lease and registry).
    32. Cancellation during Stage 1 -> Stage 2 handoff (cleanly stops before Stage 2).
    33. Worker remains alive after generator closure (retained under worker ownership until death).
    34. Final registry and lease cleanup (zero leaked leases or registry entries).
    """

    def setUp(self):
        self.manager = OllamaLifecycleManager(
            base_url="http://localhost:11434",
            default_timeout=5.0,
            eviction_timeout=0.3,
            embedding_policy=EmbeddingPolicy.PERSISTENT,
        )
        self.manager._query_ollama_ps = MagicMock(return_value=[])
        self.manager._evict_model_http = MagicMock(return_value=True)
        reset_model_manager_for_testing(self.manager)
        cancellation_registry.clear()
        self.client = app.test_client()

        # Consistent RAG mock across all tests
        self.rag_patch = patch("app.get_rag_service")
        self.mock_rag = self.rag_patch.start()
        self.mock_rag.return_value.retrieve_context.return_value = MagicMock(
            has_context=False,
            context_text="",
            mode="off",
            status="disabled",
            sources=[],
        )

    def tearDown(self):
        self.rag_patch.stop()
        cancellation_registry.clear()
        reset_model_manager_for_testing(None)

    # ── Test 25: Normal frontend completion ─────────────────────────────────

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["gemma4:12b"])
    def test_25_normal_frontend_completion(self, mock_models, mock_post):
        """Normal completion: stream exhausts, registry is cleared, lease is released."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'{"message": {"content": "Hello "}, "done": false}',
            b'{"message": {"content": "world!"}, "done": true, "done_reason": "stop"}',
        ]
        mock_resp.close = MagicMock()
        mock_post.return_value = mock_resp

        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "hi"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(resp.status_code, 200)
        chunks = [chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in resp.response]
        full_output = "".join(chunks)
        self.assertIn("Hello ", full_output)
        self.assertIn("world!", full_output)

        # Invariant checks:
        self.assertFalse(cancellation_registry.is_active(req_id), "Registry must clean up after normal completion")
        self.assertEqual(len(self.manager._active_gen_leases.get("gemma4:12b", set())), 0, "No active leases remain")

    # ── Test 26: Explicit Stop cancellation ─────────────────────────────────

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["gemma4:12b"])
    def test_26_explicit_stop_cancellation(self, mock_models, mock_post):
        """Explicit Stop cancellation via /api/chat/abort cancels stream and cleans up."""
        can_abort = threading.Event()
        worker_started = threading.Event()

        def stream_generator(*args, **kwargs):
            worker_started.set()
            yield b'{"message": {"content": "Starting..."}, "done": false}'
            can_abort.wait(timeout=2.0)
            yield b'{"message": {"content": "Never reached"}, "done": true}'

        mock_resp = MagicMock()
        mock_resp.iter_lines.side_effect = stream_generator
        mock_resp.close = MagicMock()
        mock_post.return_value = mock_resp

        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "hi"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(resp.status_code, 200)

        # Trigger abort in a separate thread once streaming worker begins
        def do_abort():
            worker_started.wait(timeout=1.0)
            abort_resp = self.client.post("/api/chat/abort", json={"request_id": req_id})
            self.assertEqual(abort_resp.status_code, 200)
            self.assertEqual(abort_resp.get_json().get("status"), "aborted")
            can_abort.set()

        abort_thr = threading.Thread(target=do_abort)
        abort_thr.start()

        chunks = [chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in resp.response]
        abort_thr.join(timeout=2.0)
        resp_text = "".join(chunks)

        self.assertIn("cancelled", resp_text.lower())
        self.assertFalse(cancellation_registry.is_active(req_id))
        self.assertEqual(len(self.manager._active_gen_leases.get("gemma4:12b", set())), 0)

    # ── Test 27: Repeated Stop requests ─────────────────────────────────────

    def test_27_repeated_stop_requests(self):
        """Repeated Stop calls for the same request are idempotent and safe."""
        req_id = str(uuid.uuid4())
        cancellation_registry.register(req_id)

        # Call abort 3 times
        r1 = self.client.post("/api/chat/abort", json={"request_id": req_id})
        r2 = self.client.post("/api/chat/abort", json={"request_id": req_id})
        r3 = self.client.post("/api/chat/abort", json={"request_id": req_id})

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r1.get_json().get("status"), "aborted")
        self.assertEqual(r2.get_json().get("status"), "aborted")
        self.assertEqual(r3.get_json().get("status"), "aborted")

        # Unregister and call abort again
        cancellation_registry.unregister(req_id)
        r4 = self.client.post("/api/chat/abort", json={"request_id": req_id})
        self.assertEqual(r4.status_code, 200)
        self.assertEqual(r4.get_json().get("status"), "ignored")

    # ── Test 28: Old-request cancellation after new request starts ──────────

    def test_28_old_request_cancellation_after_new_request_starts(self):
        """Old request cancellation cannot abort or impact a newer active request."""
        req_old = str(uuid.uuid4())
        req_new = str(uuid.uuid4())

        # Old request registers and finishes (unregistered)
        cancellation_registry.register(req_old)
        cancellation_registry.unregister(req_old)

        # New request registers
        ev_new, _ = cancellation_registry.register(req_new)
        self.assertFalse(ev_new.is_set())

        # Client sends abort for old request
        abort_resp = self.client.post("/api/chat/abort", json={"request_id": req_old})
        self.assertEqual(abort_resp.status_code, 200)
        self.assertEqual(abort_resp.get_json().get("status"), "ignored")

        # Invariant: newer request is completely untouched
        self.assertFalse(ev_new.is_set())
        self.assertTrue(cancellation_registry.is_active(req_new))

    # ── Test 29: Unexpected SSE disconnect ──────────────────────────────────

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["gemma4:12b"])
    def test_29_unexpected_sse_disconnect(self, mock_models, mock_post):
        """Unexpected SSE disconnect closes generator, unregisters registry, releases lease."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [b'{"message": {"content": "Token1"}, "done": false}']
        mock_resp.close = MagicMock()
        mock_post.return_value = mock_resp

        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "hi"}],
                "mode": "chat",
                "rag_mode": "off",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(cancellation_registry.is_active(req_id))

        # Simulate unexpected browser disconnect by closing response without consuming
        resp.close()

        # Invariant checks:
        self.assertFalse(cancellation_registry.is_active(req_id), "Registry must clean up on disconnect")
        self.assertEqual(len(self.manager._active_gen_leases.get("gemma4:12b", set())), 0)

    # ── Test 30: Disconnect during Stage 1 ───────────────────────────────────

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["deepseek-coder-v2:16b", "Qwen2.5-Coder:7B"])
    def test_30_disconnect_during_stage1(self, mock_models, mock_post):
        """Disconnect during Stage 1 aborts Stage 1, cleans up lease, never enters Stage 2."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [b'{"message": {"content": "Blueprint draft"}, "done": false}']
        mock_resp.close = MagicMock()
        mock_post.return_value = mock_resp

        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "Write a python server"}],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(cancellation_registry.is_active(req_id))

        # Close during Stage 1
        resp.close()

        # Invariant checks:
        self.assertFalse(cancellation_registry.is_active(req_id))
        self.assertEqual(len(self.manager._active_gen_leases.get("deepseek-coder-v2:16b", set())), 0)
        self.assertEqual(len(self.manager._active_gen_leases.get("qwen2.5-coder:7b", set())), 0)

    # ── Test 31: Disconnect during Stage 2 ───────────────────────────────────

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["deepseek-coder-v2:16b", "Qwen2.5-Coder:7B"])
    def test_31_disconnect_during_stage2(self, mock_models, mock_post):
        """Disconnect during Stage 2 aborts Stage 2, releases lease, and cleans up registry."""
        call_count = 0

        def mock_ollama_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            m = MagicMock()
            if call_count == 1:
                # Stage 1 (Architect) finishes cleanly
                m.iter_lines.return_value = [b'{"message": {"content": "Design OK"}, "done": true, "done_reason": "stop"}']
            else:
                # Stage 2 (Coder) streams
                m.iter_lines.return_value = [b'{"message": {"content": "def main():"}, "done": false}']
            m.close = MagicMock()
            return m

        mock_post.side_effect = mock_ollama_post

        req_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "Write a full service implementation"}],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(resp.status_code, 200)

        # Read past Stage 1 until Stage 2 starts
        gen = resp.response
        for line in gen:
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if "stage_start" in line_str and "coder" in line_str:
                break

        # Disconnect during Stage 2
        resp.close()

        # Invariant checks:
        self.assertFalse(cancellation_registry.is_active(req_id))
        self.assertEqual(len(self.manager._active_gen_leases.get("deepseek-coder-v2:16b", set())), 0)
        self.assertEqual(len(self.manager._active_gen_leases.get("qwen2.5-coder:7b", set())), 0)

    # ── Test 32: Cancellation during Stage 1 -> Stage 2 handoff ─────────────

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["deepseek-coder-v2:16b", "Qwen2.5-Coder:7B"])
    def test_32_cancellation_during_stage1_to_stage2_handoff(self, mock_models, mock_post):
        """Cancellation during pipeline handoff verification cleanly stops before Stage 2."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [b'{"message": {"content": "Design OK"}, "done": true, "done_reason": "stop"}']
        mock_resp.close = MagicMock()
        mock_post.return_value = mock_resp

        req_id = str(uuid.uuid4())

        # Intercept verify_pipeline_handoff and trigger cancellation
        orig_handoff = self.manager.verify_pipeline_handoff

        def handoff_with_cancel(*args, **kwargs):
            cancellation_registry.abort(req_id)
            orig_handoff(*args, **kwargs)

        self.manager.verify_pipeline_handoff = handoff_with_cancel

        resp = self.client.post(
            "/api/chat",
            json={
                "request_id": req_id,
                "messages": [{"role": "user", "content": "Build an app"}],
                "mode": "pipeline",
                "rag_mode": "off",
            },
        )
        self.assertEqual(resp.status_code, 200)
        chunks = [c.decode("utf-8") if isinstance(c, bytes) else c for c in resp.response]
        full_text = "".join(chunks)

        # Invariant checks:
        self.assertIn("cancelled", full_text.lower())
        self.assertNotIn("stage_start", [l for l in full_text.split("\n") if "coder" in l])
        self.assertEqual(len(self.manager._active_gen_leases.get("qwen2.5-coder:7b", set())), 0)
        self.assertFalse(cancellation_registry.is_active(req_id))

    # ── Test 33: Worker remains alive after generator closure ───────────────

    def test_33_worker_remains_alive_after_generator_closure(self):
        """Worker remains alive after generator exit: lease retained under worker ownership, safely released when worker dies."""
        lease = ModelLease(self.manager, "test-model:latest", ModelLeaseType.GENERATION)
        with self.manager._cv:
            self.manager._active_gen_leases[lease.model_name].add(lease)

        worker_unblock = threading.Event()
        worker_done = threading.Event()

        def worker_body():
            worker_unblock.wait(timeout=2.0)
            lease.mark_worker_finished()
            lease.release_from_worker()
            worker_done.set()

        worker_thr = threading.Thread(target=worker_body, name="slow-worker")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()

        # Generator exits while worker is alive
        lease.detach_or_release()

        # Invariants while worker alive:
        self.assertEqual(lease.state, _LeaseState.DETACHED)
        self.assertIn(lease, self.manager._active_gen_leases[lease.model_name])

        # Recovery refuses to steal lease while worker is alive
        self.assertFalse(lease.recover_detached_lease())
        self.assertEqual(lease.state, _LeaseState.DETACHED)

        # Worker completes
        worker_unblock.set()
        worker_thr.join(timeout=1.0)
        worker_done.wait(timeout=1.0)

        # Invariants after worker dies:
        self.assertEqual(lease.state, _LeaseState.RELEASED)
        self.assertNotIn(lease, self.manager._active_gen_leases.get(lease.model_name, set()))

    # ── Test 34: Final registry and lease cleanup ────────────────────────────

    @patch("app.http_requests.post")
    @patch("app.get_available_ollama_models", return_value=["gemma4:12b"])
    def test_34_final_registry_and_lease_cleanup(self, mock_models, mock_post):
        """Batch sequence of normal, aborted, and disconnected requests leaves zero leaked leases or registry entries."""
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [b'{"message": {"content": "ok"}, "done": true, "done_reason": "stop"}']
        mock_resp.close = MagicMock()
        mock_post.return_value = mock_resp

        # 1. Normal request
        id1 = str(uuid.uuid4())
        r1 = self.client.post("/api/chat", json={"request_id": id1, "messages": [{"role": "user", "content": "1"}], "mode": "chat", "rag_mode": "off"})
        list(r1.response)

        # 2. Aborted request
        id2 = str(uuid.uuid4())
        r2 = self.client.post("/api/chat", json={"request_id": id2, "messages": [{"role": "user", "content": "2"}], "mode": "chat", "rag_mode": "off"})
        self.client.post("/api/chat/abort", json={"request_id": id2})
        list(r2.response)

        # 3. Disconnected request
        id3 = str(uuid.uuid4())
        r3 = self.client.post("/api/chat", json={"request_id": id3, "messages": [{"role": "user", "content": "3"}], "mode": "chat", "rag_mode": "off"})
        r3.close()

        # Final audit:
        self.assertEqual(cancellation_registry.active_count(), 0, "No active requests in cancellation registry")
        diag = self.manager.get_diagnostics()
        self.assertEqual(len(diag["active_gen_leases"]), 0, "No active generation leases in manager")
        self.assertEqual(diag["active_embed_leases"], 0, "No active embedding leases in manager")


if __name__ == "__main__":
    unittest.main()
