"""
Stage 3.2.3 — Atomic Worker Completion and Lease Release Tests

Verifies the _LeaseState state-machine eliminates the TOCTOU race between
mark_worker_finished() (worker thread) and detach_or_release() (generator
thread), guaranteeing exactly-once lease release under all orderings.

Each test uses explicit threading.Event barriers to make the race
deterministic rather than relying on arbitrary sleeps.
"""

import json
import threading
import time
import unittest
from typing import List, Optional
from unittest.mock import MagicMock, patch

from app import app
from model_lifecycle import (
    ModelLease,
    ModelLeaseType,
    ModelLeaseTimeoutError,
    OllamaLifecycleManager,
    _LeaseState,
    get_model_manager,
    reset_model_manager_for_testing,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_manager(**kwargs) -> OllamaLifecycleManager:
    """Create an isolated OllamaLifecycleManager with /api/ps mocked to empty."""
    mgr = OllamaLifecycleManager(
        base_url="http://localhost:11434",
        default_timeout=5.0,
        eviction_timeout=0.3,
        **kwargs,
    )
    mgr._query_ollama_ps = MagicMock(return_value=[])
    mgr._evict_model_http = MagicMock(return_value=True)
    return mgr


def _make_lease(mgr: OllamaLifecycleManager, model: str = "qwen:latest") -> ModelLease:
    """Acquire a generation lease synchronously (fast-path: model already resident)."""
    with mgr._cv:
        mgr._resident_gen_model = model
        lease = ModelLease(mgr, model, ModelLeaseType.GENERATION)
        mgr._active_gen_leases[lease.model_name].add(lease)
    return lease


def _release_tracker(mgr: OllamaLifecycleManager) -> List[str]:
    """Wraps _on_lease_released to record each notification."""
    notifications: List[str] = []
    original = mgr._on_lease_released

    def _spy(lease):
        notifications.append(lease.id)
        original(lease)

    mgr._on_lease_released = _spy
    return notifications


# ─────────────────────────────────────────────────────────────────────────────
# Test class
# ─────────────────────────────────────────────────────────────────────────────

class TestAtomicLeaseStateTransitions(unittest.TestCase):
    """Unit tests for the _LeaseState atomic state machine on ModelLease."""

    def setUp(self):
        self.mgr = _make_manager()
        reset_model_manager_for_testing(self.mgr)

    def tearDown(self):
        reset_model_manager_for_testing(None)

    # ── 1. Worker finishes BEFORE generator detaches (Case A) ─────────────

    def test_01_worker_done_before_generator_detaches(self):
        """Case A: mark_worker_finished() first → returns False; detach_or_release()
        subsequently returns True (generator releases).  Exactly one notification."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        worker_done_res = []
        def _worker():
            res = lease.mark_worker_finished()
            worker_done_res.append(res)

        worker_thr = threading.Thread(target=_worker, name="fake-worker")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join()

        self.assertEqual(worker_done_res, [False], "Worker beat generator; generator should release")
        self.assertEqual(lease._state, _LeaseState.WORKER_DONE)

        # Generator then detaches
        should_release = lease.detach_or_release()
        self.assertTrue(should_release, "Generator must claim release when worker already done")
        self.assertEqual(lease._state, _LeaseState.RELEASED)
        lease._manager._on_lease_released(lease)   # simulate actual release

        self.assertEqual(len(notifications), 1, "Exactly one manager notification")

    # ── 2. Generator detaches BEFORE worker finishes (Case B) ────────────

    def test_02_generator_detaches_before_worker_done(self):
        """Case B: detach_or_release() first → returns False (worker will release);
        mark_worker_finished() subsequently returns True (worker releases).
        Exactly one notification.
        Worker completion code runs from the actual worker thread (matches production)."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        # Worker stays alive until signalled, then atomically completes
        worker_do_finish = threading.Event()
        worker_finished_ev = threading.Event()

        def _worker():
            worker_do_finish.wait(timeout=5.0)
            should_release = lease.mark_worker_finished()
            if should_release:
                lease.release_from_worker()
            worker_finished_ev.set()

        worker_thr = threading.Thread(target=_worker, name="fake-worker-slow")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        # Generator exits first while worker still running
        should_release = lease.detach_or_release()
        self.assertFalse(should_release, "Generator should delegate release to worker")
        self.assertEqual(lease._state, _LeaseState.DETACHED)

        # Signal worker to finish
        worker_do_finish.set()
        worker_finished_ev.wait(timeout=3.0)
        worker_thr.join(timeout=3.0)

        self.assertFalse(worker_thr.is_alive())
        self.assertEqual(lease._state, _LeaseState.RELEASED)
        self.assertEqual(len(notifications), 1, "Exactly one manager notification")

    # ── 3. Concurrent detachment and worker completion (Case C) ──────────

    def test_03_concurrent_detach_and_worker_completion(self):
        """Case C: Both operations race; exactly one side releases."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        start_barrier = threading.Barrier(2)

        def run_generator():
            start_barrier.wait()
            should_release = lease.detach_or_release()
            if should_release:
                lease._manager._on_lease_released(lease)

        def run_worker():
            start_barrier.wait()
            should_release = lease.mark_worker_finished()
            if should_release:
                lease.release_from_worker()

        g_thread = threading.Thread(target=run_generator, name="gen-thread")
        w_thread = threading.Thread(target=run_worker, name="worker-thread")
        lease.transfer_to_worker(w_thread)

        g_thread.start()
        w_thread.start()
        g_thread.join(timeout=3.0)
        w_thread.join(timeout=3.0)

        self.assertFalse(g_thread.is_alive())
        self.assertFalse(w_thread.is_alive())
        self.assertEqual(lease._state, _LeaseState.RELEASED)
        self.assertEqual(len(notifications), 1,
                         f"Exactly one manager notification; got {len(notifications)}")

    # ── 4. Repeated concurrent completion notifications ───────────────────

    def test_04_repeated_concurrent_worker_finished_calls(self):
        """mark_worker_finished() called multiple times by worker thread → idempotent."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        results = []
        def _worker():
            for _ in range(5):
                results.append(lease.mark_worker_finished())

        worker_thr = threading.Thread(target=_worker, name="fake-worker")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join(timeout=3.0)

        # Worker marked WORKER_DONE; all calls return False (generator will release)
        true_count = sum(1 for r in results if r)
        self.assertEqual(true_count, 0, "Worker marked WORKER_DONE; generator releases — no True expected")
        self.assertEqual(lease._state, _LeaseState.WORKER_DONE)

        # Now generator detaches — it should claim release
        should_release = lease.detach_or_release()
        self.assertTrue(should_release)
        lease._manager._on_lease_released(lease)
        self.assertEqual(len(notifications), 1)

    # ── 5. Exactly-once manager release notification ──────────────────────

    def test_05_exactly_once_release_notification_case_a(self):
        """Calling detach_or_release() twice after worker_finished does not double-notify."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        def _worker():
            lease.mark_worker_finished()

        worker_thr = threading.Thread(target=_worker, name="fake-w")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join()

        # First detach: should release
        should_release = lease.detach_or_release()
        self.assertTrue(should_release)
        lease._manager._on_lease_released(lease)

        # Second detach: idempotent
        should_release2 = lease.detach_or_release()
        self.assertFalse(should_release2)
        self.assertEqual(len(notifications), 1)

    def test_06_exactly_once_release_notification_case_b(self):
        """Calling mark_worker_finished() twice after detachment does not double-notify.
        Both calls run from the actual worker thread to satisfy ownership check."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        do_finish = threading.Event()
        finished_ev = threading.Event()

        def _worker():
            do_finish.wait(timeout=5.0)
            # First call: should claim release
            should_release = lease.mark_worker_finished()
            if should_release:
                lease.release_from_worker()
            # Second call from same worker thread: idempotent
            should_release2 = lease.mark_worker_finished()
            self.assertFalse(should_release2, "Second mark_worker_finished must be no-op")
            finished_ev.set()

        worker_thr = threading.Thread(target=_worker, name="fake-ww")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        lease.detach_or_release()   # → DETACHED; signal worker to release

        do_finish.set()
        finished_ev.wait(timeout=3.0)
        worker_thr.join(timeout=3.0)

        self.assertFalse(worker_thr.is_alive())
        self.assertEqual(lease._state, _LeaseState.RELEASED)
        self.assertEqual(len(notifications), 1)

    # ── 6. No lease leak after either ordering ────────────────────────────

    def test_07_no_lease_leak_case_a(self):
        """After worker_done then detach, lease removed from manager tracking."""
        lease = _make_lease(self.mgr, "qwen:latest")
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        def _worker():
            lease.mark_worker_finished()

        worker_thr = threading.Thread(target=_worker, name="w")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join()

        should_release = lease.detach_or_release()
        self.assertTrue(should_release)
        lease._manager._on_lease_released(lease)

        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

    def test_08_no_lease_leak_case_b(self):
        """After detach then worker_done, lease removed from manager tracking.
        Worker completion runs from the actual worker thread."""
        lease = _make_lease(self.mgr, "qwen:latest")
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        do_finish = threading.Event()
        finished_ev = threading.Event()

        def _worker():
            do_finish.wait(timeout=5.0)
            should_release = lease.mark_worker_finished()
            if should_release:
                lease.release_from_worker()
            finished_ev.set()

        worker_thr = threading.Thread(target=_worker, name="ww")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        lease.detach_or_release()       # → DETACHED

        do_finish.set()
        finished_ev.wait(timeout=3.0)
        worker_thr.join(timeout=3.0)

        self.assertFalse(worker_thr.is_alive())
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

    # ── 7. No premature release while worker active ───────────────────────

    def test_09_no_premature_release_while_worker_active(self):
        """detach_or_release() while worker running must NOT release (returns False)."""
        lease = _make_lease(self.mgr, "qwen:latest")
        blocking = threading.Event()

        def _worker():
            blocking.wait()   # stay alive

        worker_thr = threading.Thread(target=_worker, name="blocking-worker")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        # Generator exits while worker is alive
        should_release = lease.detach_or_release()
        self.assertFalse(should_release, "Must not release while worker is alive")
        self.assertEqual(lease._state, _LeaseState.DETACHED)
        # Lease is still in manager tracking
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        blocking.set()   # let worker finish
        worker_thr.join(timeout=2.0)

    # ── 8. Worker release from wrong thread is rejected ───────────────────

    def test_10_worker_release_from_wrong_thread_rejected(self):
        """release_from_worker() from a non-worker thread must not release the lease."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()
        worker_thr = threading.Thread(target=lambda: blocking.wait(), name="ollama-worker-test")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        # Transition to DETACHED so release_from_worker() would otherwise proceed
        lease.detach_or_release()
        lease.mark_worker_finished()   # → RELEASED (but we haven't called release_from_worker yet)
        # Revert state to DETACHED to simulate a non-worker trying to release
        with lease._lock:
            lease._state = _LeaseState.DETACHED

        # Non-worker thread attempt
        called_from_main = threading.current_thread()
        self.assertIsNot(called_from_main, worker_thr)
        # release_from_worker checks caller == _worker_thread; this should log warning and no-op
        with self.assertLogs("model_lifecycle", level="WARNING") as cm:
            lease.release_from_worker()
        self.assertTrue(any("non-worker" in line.lower() or "ignoring" in line.lower()
                            for line in cm.output))
        # Lease was not removed from manager
        self.assertEqual(len(notifications), 0)

        blocking.set()

    # ── 9. Recovery races with worker completion ──────────────────────────

    def test_11_recovery_does_not_double_release(self):
        """recover_stuck_leases() racing with worker completion → exactly one release."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()
        worker_started = threading.Event()

        def _worker():
            worker_started.set()
            blocking.wait()
            # Simulate worker completing: transitions DETACHED → RELEASED
            should_release = lease.mark_worker_finished()
            if should_release:
                lease.release_from_worker()

        worker_thr = threading.Thread(target=_worker, name="ollama-worker-race")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)
        worker_started.wait(timeout=2.0)

        # Generator detaches before worker finishes
        lease.detach_or_release()   # → DETACHED

        # Barrier: recovery and worker completion race
        race_barrier = threading.Barrier(2)

        def _recovery():
            race_barrier.wait()
            self.mgr.recover_stuck_leases()

        def _unblock_worker():
            race_barrier.wait()
            blocking.set()

        t_rec = threading.Thread(target=_recovery)
        t_unblock = threading.Thread(target=_unblock_worker)
        t_rec.start()
        t_unblock.start()
        t_rec.join(timeout=3.0)
        t_unblock.join(timeout=3.0)
        worker_thr.join(timeout=3.0)

        self.assertEqual(len(notifications), 1,
                         f"Exactly one manager notification; got {len(notifications)}")
        self.assertEqual(lease._state, _LeaseState.RELEASED)

    # ── 10. Pipeline handoff after detached architect worker completes ────

    def test_12_pipeline_handoff_after_detached_architect_worker(self):
        """After architect's detached worker finishes, coder lease can be acquired."""
        # Simulate: architect worker finishes → coder should unblock
        arch_lease = _make_lease(self.mgr, "deepseek-coder-v2:16b")
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()

        def _arch_worker():
            blocking.wait()
            # Worker completion → DETACHED transition releases
            should_release = arch_lease.mark_worker_finished()
            if should_release:
                arch_lease.release_from_worker()

        arch_worker_thr = threading.Thread(target=_arch_worker, name="ollama-worker-arch")
        arch_worker_thr.daemon = True
        arch_worker_thr.start()
        arch_lease.transfer_to_worker(arch_worker_thr)

        # Generator exits while worker alive
        arch_lease.detach_or_release()   # → DETACHED
        self.assertEqual(arch_lease._state, _LeaseState.DETACHED)

        # Coder tries to acquire; must wait for arch worker to finish
        coder_acquired = threading.Event()
        coder_error: list = []

        def _acquire_coder():
            try:
                with self.mgr._cv:
                    self.mgr._resident_gen_model = None   # force transition
                with self.mgr.acquire_generation_lease("qwen2.5-coder:7b", timeout=5.0):
                    coder_acquired.set()
            except Exception as e:
                coder_error.append(e)

        coder_thr = threading.Thread(target=_acquire_coder, name="coder-requester")
        coder_thr.start()

        # Unblock architect worker after a short delay
        time.sleep(0.1)
        blocking.set()
        arch_worker_thr.join(timeout=3.0)
        coder_thr.join(timeout=10.0)

        self.assertFalse(coder_thr.is_alive(), "Coder thread should have finished")
        if coder_error:
            raise coder_error[0]
        self.assertTrue(coder_acquired.is_set(), "Coder should have acquired lease")

        # Arch lease was fully released
        self.assertEqual(arch_lease._state, _LeaseState.RELEASED)
        self.assertNotIn(arch_lease, self.mgr._active_gen_leases.get("deepseek-coder-v2:16b", set()))


class TestAtomicLeaseIntegration(unittest.TestCase):
    """Integration tests: app.py stream_ollama_stage uses new atomic API."""

    def setUp(self):
        self.manager = _make_manager(embedding_policy="persistent")
        reset_model_manager_for_testing(self.manager)
        self.client = app.test_client()

    def tearDown(self):
        reset_model_manager_for_testing(None)

    def _make_mock_post(self, tokens=None, delay=0.0):
        tokens = tokens or [{"message": {"content": "hello"}, "done": False},
                            {"message": {"content": ""}, "done": True, "done_reason": "stop"}]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        def _lines():
            for t in tokens:
                if delay:
                    time.sleep(delay)
                yield json.dumps(t)

        mock_resp.iter_lines.return_value = _lines()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    # ── Normal completion releases lease cleanly ──────────────────────────

    @patch("app.http_requests.post")
    def test_13_normal_completion_lease_released(self, mock_post):
        """After normal inference, generation lease is fully released."""
        mock_post.return_value = self._make_mock_post()

        resp = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}],
                  "mode": "chat", "rag_mode": "off"},
        )
        resp.get_data()

        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"], {},
                         f"Active leases should be empty after normal completion: {diag}")
        self.assertEqual(diag["stuck_workers"], [])

    # ── Client disconnect leaves no stuck lease ───────────────────────────

    @patch("app.http_requests.post")
    def test_14_disconnect_no_lease_leak(self, mock_post):
        """After generator aborts mid-stream, no lease leaks after worker finishes."""
        post_entered = threading.Event()
        worker_unblock = threading.Event()

        def _slow_post(*args, **kwargs):
            post_entered.set()
            worker_unblock.wait(timeout=10.0)
            mock_resp = self._make_mock_post()
            return mock_resp

        mock_post.side_effect = _slow_post

        resp = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}],
                  "mode": "chat", "rag_mode": "off"},
        )

        # Drain until worker is blocked inside post
        for chunk in resp.response:
            if post_entered.wait(timeout=0.05):
                break

        resp.response.close()   # generator abort
        worker_unblock.set()    # let worker finish

        time.sleep(1.5)   # allow worker cleanup

        diag = self.manager.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"], {},
                         f"Active leases should be empty after disconnect+worker finish: {diag}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
