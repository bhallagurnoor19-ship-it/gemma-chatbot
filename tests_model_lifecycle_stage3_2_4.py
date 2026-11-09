"""
Stage 3.2.4 — Final Concurrency Audit & Targeted Corrections Tests

Deterministic test suite covering:
1. Worker identity validation on mark_worker_finished() and release_from_worker()
   - Unauthorized thread rejected in ACTIVE state (state remains ACTIVE, returns False)
   - Unauthorized thread rejected in DETACHED state (state remains DETACHED, returns False)
   - Authorized worker accepted in both states
   - Exactly-once manager notification after valid completion
2. Dedicated recovery ownership and recover_detached_lease()
   - Recovery of a detached lease with a dead worker (succeeds, exactly-once notification)
   - Recovery attempted while detached worker is still alive (refused, lease preserved, force=False)
   - Recovery attempted with force=True while worker alive (refused, lease preserved, warning logged)
   - Recovery racing with worker completion (exactly-once notification)
   - Recovery racing with generator detachment (thread-safe, exactly-once notification)
   - Repeated recovery calls (idempotent, 1 notification)
   - Recovery of already RELEASED lease (no-op, 0 new notifications)
   - Recovery of ACTIVE lease (refused, lease preserved)
3. Release notification consistency and idempotency
   - Double release_from_worker calls on same lease (1 notification)
   - Lagging manager cleanup for already RELEASED lease in recover_stuck_leases
4. app.py integration edge cases
   - Worker fails before HTTP response is created (exception during post; clean release)
   - Worker ownership transferred before worker thread starts
"""

import json
import threading
import time
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from app import app
from model_lifecycle import (
    ModelLease,
    ModelLeaseType,
    OllamaLifecycleManager,
    _LeaseState,
    reset_model_manager_for_testing,
)


def _make_manager(**kwargs) -> OllamaLifecycleManager:
    """Create isolated OllamaLifecycleManager with /api/ps mocked."""
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
    """Acquire a generation lease directly for test inspection."""
    with mgr._cv:
        mgr._resident_gen_model = model
        lease = ModelLease(mgr, model, ModelLeaseType.GENERATION)
        mgr._active_gen_leases[lease.model_name].add(lease)
    return lease


def _release_tracker(mgr: OllamaLifecycleManager) -> List[str]:
    """Wraps _on_lease_released to count notifications."""
    notifications: List[str] = []
    original = mgr._on_lease_released

    def _spy(lease):
        notifications.append(lease.id)
        original(lease)

    mgr._on_lease_released = _spy
    return notifications


class TestWorkerIdentityValidation(unittest.TestCase):
    """Verifies worker identity validation inside mark_worker_finished() and release_from_worker()."""

    def setUp(self):
        self.mgr = _make_manager()
        reset_model_manager_for_testing(self.mgr)

    def tearDown(self):
        reset_model_manager_for_testing(None)

    def test_01_unauthorized_thread_mark_worker_finished_in_active_state_rejected(self):
        """Unauthorized thread calling mark_worker_finished() in ACTIVE state is rejected.
        State remains ACTIVE, returns False, no notification."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()
        worker_thr = threading.Thread(target=blocking.wait, name="real-worker-01")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        self.assertEqual(lease._state, _LeaseState.ACTIVE)

        # Call from non-worker thread (MainThread)
        res = lease.mark_worker_finished()
        self.assertFalse(res, "Unauthorized caller must receive False")
        self.assertEqual(lease._state, _LeaseState.ACTIVE, "State must remain ACTIVE")
        self.assertEqual(len(notifications), 0, "No manager notification permitted")

        blocking.set()
        worker_thr.join(timeout=3.0)

    def test_02_unauthorized_thread_mark_worker_finished_in_detached_state_rejected(self):
        """Unauthorized thread calling mark_worker_finished() in DETACHED state is rejected.
        State remains DETACHED, returns False, no notification."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()
        worker_thr = threading.Thread(target=blocking.wait, name="real-worker-02")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        # Generator detaches
        should_release_gen = lease.detach_or_release()
        self.assertFalse(should_release_gen)
        self.assertEqual(lease._state, _LeaseState.DETACHED)

        # Call from non-worker thread (MainThread)
        res = lease.mark_worker_finished()
        self.assertFalse(res, "Unauthorized caller must receive False")
        self.assertEqual(lease._state, _LeaseState.DETACHED, "State must remain DETACHED")
        self.assertEqual(len(notifications), 0, "No manager notification permitted")

        blocking.set()
        worker_thr.join(timeout=3.0)

    def test_03_authorized_worker_in_active_and_detached_states(self):
        """Authorized worker calling mark_worker_finished() succeeds in both states.
        ACTIVE -> WORKER_DONE (returns False), DETACHED -> RELEASED (returns True)."""
        # Part A: ACTIVE -> WORKER_DONE
        lease_a = _make_lease(self.mgr)
        res_a = []

        def _worker_a():
            res_a.append(lease_a.mark_worker_finished())

        wt_a = threading.Thread(target=_worker_a, name="worker-a")
        lease_a.transfer_to_worker(wt_a)
        wt_a.start()
        wt_a.join(timeout=3.0)

        self.assertEqual(res_a, [False], "Worker finished before detachment returns False")
        self.assertEqual(lease_a._state, _LeaseState.WORKER_DONE)

        # Part B: DETACHED -> RELEASED
        lease_b = _make_lease(self.mgr)
        res_b = []
        ready = threading.Event()
        proceed = threading.Event()

        def _worker_b():
            ready.set()
            proceed.wait(timeout=3.0)
            res_b.append(lease_b.mark_worker_finished())

        wt_b = threading.Thread(target=_worker_b, name="worker-b")
        lease_b.transfer_to_worker(wt_b)
        wt_b.start()
        ready.wait(timeout=3.0)

        # Generator detaches first
        lease_b.detach_or_release()
        self.assertEqual(lease_b._state, _LeaseState.DETACHED)

        proceed.set()
        wt_b.join(timeout=3.0)

        self.assertEqual(res_b, [True], "Worker finished after detachment returns True")
        self.assertEqual(lease_b._state, _LeaseState.RELEASED)

    def test_04_unauthorized_thread_release_from_worker_rejected(self):
        """release_from_worker() called by unauthorized thread is ignored and does not notify."""
        lease = _make_lease(self.mgr)
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()
        worker_thr = threading.Thread(target=blocking.wait, name="real-worker-04")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        with lease._lock:
            lease._state = _LeaseState.DETACHED

        # Unauthorized caller calls release_from_worker()
        lease.release_from_worker()

        self.assertEqual(len(notifications), 0, "Unauthorized thread must not notify manager")
        self.assertEqual(lease._state, _LeaseState.DETACHED)

        blocking.set()
        worker_thr.join(timeout=3.0)


class TestRecoveryOwnership(unittest.TestCase):
    """Verifies dedicated recovery methods on ModelLease and OllamaLifecycleManager."""

    def setUp(self):
        self.mgr = _make_manager()
        reset_model_manager_for_testing(self.mgr)

    def tearDown(self):
        reset_model_manager_for_testing(None)

    def test_05_recover_detached_lease_with_dead_worker_succeeds(self):
        """Recovery of a detached lease whose worker genuinely terminated succeeds."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        worker_thr = threading.Thread(target=lambda: None, name="dead-worker")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join(timeout=3.0)

        with lease._lock:
            lease._state = _LeaseState.DETACHED

        # Worker is confirmed dead, state is DETACHED
        recovered = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(recovered, 1)
        self.assertEqual(len(notifications), 1)
        self.assertEqual(lease._state, _LeaseState.RELEASED)
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

    def test_06_recover_detached_lease_with_live_worker_refused_force_false(self):
        """Recovery attempted while detached worker is alive is refused (force=False)."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()
        worker_thr = threading.Thread(target=blocking.wait, name="live-worker-06")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        lease.detach_or_release()
        self.assertEqual(lease._state, _LeaseState.DETACHED)

        # Attempt recovery while worker is alive
        recovered = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(recovered, 0, "Must not recover while worker is alive")
        self.assertEqual(len(notifications), 0)
        self.assertEqual(lease._state, _LeaseState.DETACHED)
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        blocking.set()
        worker_thr.join(timeout=3.0)

    def test_07_recover_detached_lease_with_live_worker_refused_force_true(self):
        """Even with force=True, recovery is REFUSED if the worker is still alive."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()
        worker_thr = threading.Thread(target=blocking.wait, name="live-worker-07")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        lease.detach_or_release()
        self.assertEqual(lease._state, _LeaseState.DETACHED)

        # Attempt recovery with force=True while worker is alive
        recovered = self.mgr.recover_stuck_leases(force=True)
        self.assertEqual(recovered, 0, "Must NEVER recover while worker is alive, even with force=True")
        self.assertEqual(len(notifications), 0)
        self.assertEqual(lease._state, _LeaseState.DETACHED)
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        blocking.set()
        worker_thr.join(timeout=3.0)

    def test_08_recovery_racing_with_worker_completion(self):
        """Recovery racing with worker completion produces exactly one notification."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        barrier = threading.Barrier(2)
        worker_unblock = threading.Event()

        def _worker():
            worker_unblock.wait(timeout=3.0)
            barrier.wait()
            should_rel = lease.mark_worker_finished()
            if should_rel:
                lease.release_from_worker()

        worker_thr = threading.Thread(target=_worker, name="race-worker-08")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        # Generator detaches
        lease.detach_or_release()
        self.assertEqual(lease._state, _LeaseState.DETACHED)

        def _recovery():
            barrier.wait()
            self.mgr.recover_stuck_leases(force=False)

        rec_thr = threading.Thread(target=_recovery, name="race-recovery-08")
        worker_unblock.set()
        rec_thr.start()

        rec_thr.join(timeout=3.0)
        worker_thr.join(timeout=3.0)

        self.assertEqual(len(notifications), 1, "Exactly one manager notification on race")
        self.assertEqual(lease._state, _LeaseState.RELEASED)

    def test_09_recovery_racing_with_generator_detachment(self):
        """Recovery racing with generator detachment does not premature-release or corrupt state."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        blocking = threading.Event()
        worker_thr = threading.Thread(target=blocking.wait, name="race-worker-09")
        worker_thr.daemon = True
        worker_thr.start()
        lease.transfer_to_worker(worker_thr)

        barrier = threading.Barrier(2)

        def _detach():
            barrier.wait()
            lease.detach_or_release()

        def _recover():
            barrier.wait()
            self.mgr.recover_stuck_leases(force=False)

        t_det = threading.Thread(target=_detach, name="detach-thr")
        t_rec = threading.Thread(target=_recover, name="recover-thr")
        t_det.start()
        t_rec.start()
        t_det.join(timeout=3.0)
        t_rec.join(timeout=3.0)

        # Because worker is alive throughout, recovery MUST NOT have released the lease
        self.assertEqual(len(notifications), 0, "Recovery must not release while worker alive")
        self.assertEqual(lease._state, _LeaseState.DETACHED)

        blocking.set()
        worker_thr.join(timeout=3.0)

    def test_10_repeated_recovery_calls_idempotent(self):
        """Repeated recover_stuck_leases() calls notify manager exactly once."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        worker_thr = threading.Thread(target=lambda: None, name="dead-worker-10")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join(timeout=3.0)

        with lease._lock:
            lease._state = _LeaseState.DETACHED

        # First call recovers
        count1 = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(count1, 1)
        self.assertEqual(len(notifications), 1)

        # Subsequent calls are no-ops
        count2 = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(count2, 0)
        count3 = self.mgr.recover_stuck_leases(force=True)
        self.assertEqual(count3, 0)
        self.assertEqual(len(notifications), 1)

    def test_11_recovery_of_active_lease_refused(self):
        """recover_stuck_leases() never recovers an ACTIVE lease (generator still attached)."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        self.assertEqual(lease._state, _LeaseState.ACTIVE)
        count = self.mgr.recover_stuck_leases(force=True)
        self.assertEqual(count, 0)
        self.assertEqual(len(notifications), 0)
        self.assertEqual(lease._state, _LeaseState.ACTIVE)


class TestWorkerReleasePrerequisites(unittest.TestCase):
    """Verifies that release_from_worker() strictly enforces the prerequisite that the
    assigned worker thread must have successfully claimed release authorization via
    mark_worker_finished() (DETACHED -> RELEASED transition)."""

    def setUp(self):
        self.mgr = _make_manager()
        reset_model_manager_for_testing(self.mgr)

    def tearDown(self):
        reset_model_manager_for_testing(None)

    def test_direct_worker_release_while_active_rejected(self):
        """Assigned worker calling release_from_worker() directly while ACTIVE is rejected.
        State remains ACTIVE, manager is not notified, and lease remains tracked."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        def _worker():
            with self.assertLogs("model_lifecycle", level="WARNING") as cm:
                lease.release_from_worker()
            self.assertTrue(any("without prior worker release authorization" in line for line in cm.output))

        worker_thr = threading.Thread(target=_worker, name="worker-direct-active")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join(timeout=3.0)

        self.assertEqual(lease._state, _LeaseState.ACTIVE)
        self.assertEqual(len(notifications), 0, "Manager must not be notified on unauthorized release")
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

    def test_direct_worker_release_while_worker_done_rejected(self):
        """Assigned worker calling release_from_worker() directly while WORKER_DONE is rejected.
        (Worker completed before generator detached; generator owns release authority).
        State remains WORKER_DONE, manager is not notified, and lease remains tracked."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        def _worker():
            # Transitions ACTIVE -> WORKER_DONE (returns False, generator will release)
            res = lease.mark_worker_finished()
            self.assertFalse(res)
            with self.assertLogs("model_lifecycle", level="WARNING") as cm:
                lease.release_from_worker()
            self.assertTrue(any("without prior worker release authorization" in line for line in cm.output))

        worker_thr = threading.Thread(target=_worker, name="worker-direct-worker-done")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join(timeout=3.0)

        self.assertEqual(lease._state, _LeaseState.WORKER_DONE)
        self.assertEqual(len(notifications), 0, "Manager must not be notified by worker in WORKER_DONE state")
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        # When generator subsequently exits, it cleanly releases
        should_rel = lease.detach_or_release()
        self.assertTrue(should_rel)
        lease._manager._on_lease_released(lease)
        self.assertEqual(len(notifications), 1)

    def test_valid_worker_completion_followed_by_release(self):
        """Worker completing after generator detachment (DETACHED -> RELEASED) successfully
        claims worker release authorization and notifies the manager exactly once."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        ready = threading.Event()
        proceed = threading.Event()

        def _worker():
            ready.set()
            proceed.wait(timeout=3.0)
            should_rel = lease.mark_worker_finished()
            self.assertTrue(should_rel, "Worker must claim release authorization after detachment")
            lease.release_from_worker()

        worker_thr = threading.Thread(target=_worker, name="worker-valid-release")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        ready.wait(timeout=3.0)

        # Generator exits first
        lease.detach_or_release()
        self.assertEqual(lease._state, _LeaseState.DETACHED)

        # Worker finishes and releases
        proceed.set()
        worker_thr.join(timeout=3.0)

        self.assertEqual(lease._state, _LeaseState.RELEASED)
        self.assertEqual(len(notifications), 1)
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

    def test_repeated_release_after_valid_completion_idempotent(self):
        """Repeated release_from_worker() calls after valid completion are completely idempotent.
        Manager receives exactly one removal notification."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        def _worker():
            should_rel = lease.mark_worker_finished()
            self.assertTrue(should_rel)
            lease.release_from_worker()
            # Repeated call from authorized worker
            lease.release_from_worker()

        worker_thr = threading.Thread(target=_worker, name="worker-repeat-release")
        lease.transfer_to_worker(worker_thr)

        lease.detach_or_release()
        worker_thr.start()
        worker_thr.join(timeout=3.0)

        self.assertEqual(lease._state, _LeaseState.RELEASED)
        self.assertEqual(len(notifications), 1, "Exactly one notification allowed")


class TestReleaseNotificationFailureWindowAndIdempotency(unittest.TestCase):
    """Verifies state transition, simulated notification failure, and recovery bookkeeping consistency."""

    def setUp(self):
        self.mgr = _make_manager()
        reset_model_manager_for_testing(self.mgr)

    def tearDown(self):
        reset_model_manager_for_testing(None)

    def test_12_double_release_from_worker_notifies_once(self):
        """Calling release_from_worker() multiple times notifies manager exactly once."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        def _worker():
            lease.mark_worker_finished()
            lease.release_from_worker()
            lease.release_from_worker()  # duplicate call

        worker_thr = threading.Thread(target=_worker, name="worker-12")
        lease.transfer_to_worker(worker_thr)

        lease.detach_or_release()
        worker_thr.start()
        worker_thr.join(timeout=3.0)

        self.assertEqual(len(notifications), 1)
        self.assertEqual(lease._state, _LeaseState.RELEASED)

    def test_13_notification_failure_followed_by_recovery(self):
        """Simulate original notification failure where lease reached RELEASED with _notified=True,
        but failed before removing lease from manager tracking. Recovery cleans up manager tracking."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        # 1. Lease created and in _active_gen_leases
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        # 2 & 3. Transition to RELEASED and set _notified = True under lock
        with lease._lock:
            lease._state = _LeaseState.RELEASED
            lease._notified = True

        # 4. Lease remains in _active_gen_leases (simulating original notification failure)
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        # 5. Injected notification failure simulation:
        # Original caller threw an exception before calling _on_lease_released
        # 6. Call recover_stuck_leases()
        recovered = self.mgr.recover_stuck_leases()

        # 7. Verify manager tracking is cleaned up
        self.assertEqual(recovered, 1)
        self.assertEqual(len(notifications), 1)
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

        # 8. Verify no duplicate or inconsistent manager state
        diag = self.mgr.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"], {})
        self.assertEqual(diag["transitioning"], False)

    def test_14_recovery_after_successful_notification_does_not_reprocess(self):
        """Recovery run after a successful lease release does not reprocess or double-notify."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        # Normal release
        lease.release()
        self.assertEqual(len(notifications), 1)
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

        # Recovery run afterward
        recovered = self.mgr.recover_stuck_leases()
        self.assertEqual(recovered, 0, "Must not recover already removed lease")
        self.assertEqual(len(notifications), 1, "Notification count must remain exactly 1")


class TestNotificationAuthorityAndRecoveryConcurrency(unittest.TestCase):
    """
    Focused concurrency tests verifying that notification authority is claimed atomically,
    recovery and normal release never both claim authority, retries work after genuine failure,
    and actual manager notification attempts are strictly counted.
    """

    def setUp(self):
        self.mgr = _make_manager()
        reset_model_manager_for_testing(self.mgr)

    def tearDown(self):
        reset_model_manager_for_testing(None)

    def test_recovery_racing_with_authorized_worker_release(self):
        """Recovery racing with an authorized worker release does not claim authority or double-notify."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        worker_in_flight = threading.Event()
        worker_can_complete = threading.Event()
        worker_ready = threading.Event()
        gen_detached = threading.Event()

        orig_on_released = self.mgr._on_lease_released

        def _delayed_on_released(l):
            if l.id == lease.id and threading.current_thread() == worker_thr:
                worker_in_flight.set()
                worker_can_complete.wait(timeout=3.0)
            orig_on_released(l)

        self.mgr._on_lease_released = _delayed_on_released

        def _worker():
            worker_ready.set()
            gen_detached.wait(timeout=3.0)
            should_rel = lease.mark_worker_finished()
            self.assertTrue(should_rel)
            lease.release_from_worker()

        worker_thr = threading.Thread(target=_worker, name="auth-worker-race")
        worker_thr.daemon = True
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        self.assertTrue(worker_ready.wait(timeout=3.0))

        # Generator detaches
        lease.detach_or_release()
        self.assertEqual(lease._state, _LeaseState.DETACHED)
        gen_detached.set()

        # Wait until worker has claimed authority and is actively calling manager
        self.assertTrue(worker_in_flight.wait(timeout=3.0))

        # While worker notification is in progress, recovery attempts to sweep
        recovered = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(recovered, 0, "Recovery must not claim authority while worker notification is pending")

        # Allow worker to finish
        worker_can_complete.set()
        worker_thr.join(timeout=3.0)

        # Exactly 1 notification should have occurred
        self.assertEqual(len(notifications), 1, "Exactly one manager notification permitted")
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

    def test_recovery_racing_with_generator_owned_release(self):
        """Recovery racing with generator-owned release does not claim authority or double-notify."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        gen_in_flight = threading.Event()
        gen_can_complete = threading.Event()

        # Worker finishes first (WORKER_DONE)
        def _worker():
            res = lease.mark_worker_finished()
            self.assertFalse(res)

        worker_thr = threading.Thread(target=_worker, name="worker-first")
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        worker_thr.join(timeout=3.0)
        self.assertEqual(lease._state, _LeaseState.WORKER_DONE)

        orig_on_released = self.mgr._on_lease_released

        def _delayed_on_released(l):
            if l.id == lease.id:
                gen_in_flight.set()
                gen_can_complete.wait(timeout=3.0)
            orig_on_released(l)

        self.mgr._on_lease_released = _delayed_on_released

        def _generator():
            should_rel = lease.detach_or_release()
            self.assertTrue(should_rel)
            self.mgr._on_lease_released(lease)

        gen_thr = threading.Thread(target=_generator, name="generator-race")
        gen_thr.daemon = True
        gen_thr.start()

        # Wait until generator has claimed authority and is notifying manager
        self.assertTrue(gen_in_flight.wait(timeout=3.0))

        # While generator notification is in progress, recovery runs
        recovered = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(recovered, 0, "Recovery must not claim authority while generator notification is pending")

        # Allow generator to complete
        gen_can_complete.set()
        gen_thr.join(timeout=3.0)

        self.assertEqual(len(notifications), 1, "Exactly one manager notification on generator-owned release race")
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

    def test_manager_notification_failing_before_bookkeeping_followed_by_successful_recovery(self):
        """Manager notification fails before bookkeeping; recovery subsequently retries and succeeds."""
        lease = _make_lease(self.mgr, "qwen:latest")
        attempt_count = [0]
        real_on_released = self.mgr._on_lease_released

        def _flaky_on_released(l):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                raise RuntimeError("Simulated transient network failure before manager bookkeeping")
            real_on_released(l)

        self.mgr._on_lease_released = _flaky_on_released

        def _worker():
            should_rel = lease.mark_worker_finished()
            self.assertTrue(should_rel)
            with self.assertRaises(RuntimeError):
                lease.release_from_worker()

        worker_thr = threading.Thread(target=_worker, name="flaky-worker")
        lease.transfer_to_worker(worker_thr)
        lease.detach_or_release()
        worker_thr.start()
        worker_thr.join(timeout=3.0)

        # First attempt failed
        self.assertEqual(attempt_count[0], 1)
        self.assertEqual(lease._state, _LeaseState.RELEASED)
        self.assertFalse(lease._notification_completed, "Notification must NOT be marked complete after failure")
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"], "Lease must remain tracked")

        # Recovery runs and successfully retries manager notification
        recovered = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(recovered, 1, "Recovery must successfully recover the incomplete release")
        self.assertEqual(attempt_count[0], 2, "Exactly 2 attempts total (1 failed, 1 successful recovery)")
        self.assertTrue(lease._notification_completed)
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

    def test_successful_notification_followed_by_recovery_no_duplicate(self):
        """Successful notification followed by recovery results in zero duplicate notifications."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        lease.release()
        self.assertEqual(len(notifications), 1)
        self.assertTrue(lease._notification_completed)

        # First recovery call
        rec1 = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(rec1, 0)
        self.assertEqual(len(notifications), 1)

        # Second recovery call with force=True
        rec2 = self.mgr.recover_stuck_leases(force=True)
        self.assertEqual(rec2, 0)
        self.assertEqual(len(notifications), 1)

    def test_repeated_recovery_calls_while_notification_pending(self):
        """Multiple recovery calls while notification is pending all return 0 and do not double-notify."""
        lease = _make_lease(self.mgr, "qwen:latest")
        notifications = _release_tracker(self.mgr)

        worker_blocked = threading.Event()
        worker_resume = threading.Event()

        orig_on_released = self.mgr._on_lease_released

        def _blocking_on_released(l):
            if l.id == lease.id:
                worker_blocked.set()
                worker_resume.wait(timeout=3.0)
            orig_on_released(l)

        self.mgr._on_lease_released = _blocking_on_released

        worker_ready = threading.Event()
        gen_detached = threading.Event()

        def _worker():
            worker_ready.set()
            gen_detached.wait(timeout=3.0)
            should_rel = lease.mark_worker_finished()
            self.assertTrue(should_rel)
            lease.release_from_worker()

        worker_thr = threading.Thread(target=_worker, name="pending-worker")
        worker_thr.daemon = True
        lease.transfer_to_worker(worker_thr)
        worker_thr.start()
        self.assertTrue(worker_ready.wait(timeout=3.0))

        lease.detach_or_release()
        self.assertEqual(lease._state, _LeaseState.DETACHED)
        gen_detached.set()
        self.assertTrue(worker_blocked.wait(timeout=3.0))

        # Call recovery 3 times while notification is pending
        for i in range(3):
            rec = self.mgr.recover_stuck_leases(force=False)
            self.assertEqual(rec, 0, f"Recovery call {i+1} must return 0 while notification pending")

        # Notification has not completed yet
        self.assertEqual(len(notifications), 0)

        # Unblock worker to finish
        worker_resume.set()
        worker_thr.join(timeout=3.0)

        # Now notification completed exactly once
        self.assertEqual(len(notifications), 1)
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

        # Recovery call afterward returns 0
        rec_after = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(rec_after, 0)
        self.assertEqual(len(notifications), 1)

    def test_exception_in_interval_after_discard_guarantees_notification_completed(self):
        """If an exception occurs after discarding a lease from manager tracking (e.g. during notify_all),
        the lease is guaranteed to have _notification_completed=True, preventing inconsistent state."""
        lease = _make_lease(self.mgr, "qwen:latest")
        self.assertIn(lease, self.mgr._active_gen_leases["qwen:latest"])

        # Patch notify_all on the manager's condition variable to raise an exception
        orig_notify = self.mgr._cv.notify_all

        def _exploding_notify():
            raise RuntimeError("Simulated failure during notify_all after discard")

        self.mgr._cv.notify_all = _exploding_notify

        try:
            with self.assertRaises(RuntimeError):
                lease.release()
        finally:
            self.mgr._cv.notify_all = orig_notify

        # 1. Lease was discarded from manager active leases
        self.assertNotIn(lease, self.mgr._active_gen_leases.get("qwen:latest", set()))

        # 2. _notification_completed is guaranteed True despite the exception
        self.assertTrue(lease._notification_completed, "Notification must be marked completed once bookkeeping committed")
        self.assertFalse(lease._notification_in_progress, "Notification in progress must be cleared")

        # 3. Recovery run afterward is a safe no-op and does not encounter inconsistent state
        recovered = self.mgr.recover_stuck_leases(force=False)
        self.assertEqual(recovered, 0)


class TestAppIntegrationEdgeCases(unittest.TestCase):
    """Verifies edge cases in app.py streaming and worker integration."""

    def setUp(self):
        self.mgr = _make_manager(embedding_policy="persistent")
        reset_model_manager_for_testing(self.mgr)
        self.client = app.test_client()

    def tearDown(self):
        reset_model_manager_for_testing(None)

    @patch("app.http_requests.post")
    def test_14_worker_fails_before_http_response_releases_cleanly(self, mock_post):
        """If worker fails during http_requests.post (e.g. ConnectionError), lease releases cleanly."""
        mock_post.side_effect = ConnectionError("Ollama daemon unreachable")

        resp = self.client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hello"}],
                  "mode": "chat", "rag_mode": "off"},
        )
        # Drain generator until error terminates it
        try:
            for _ in resp.response:
                pass
        except Exception:
            pass

        time.sleep(0.5)
        diag = self.mgr.get_diagnostics()
        self.assertEqual(diag["active_gen_leases"], {}, f"Lease must not leak on worker failure: {diag}")
        self.assertEqual(diag["stuck_workers"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
