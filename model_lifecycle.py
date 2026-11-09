"""
Resource-Aware Model Lifecycle Manager for Ollama
- Thread-safe, lease-based synchronization for local LLMs and embeddings
- Guarantees at most one active generation model resident on resource-constrained hosts
- Adaptive coordination between heavy generation models (DeepSeek 16B) and EmbeddingGemma
- Strict condition-variable state transitions, reconciliation against /api/ps, and bounded waits
- Explicit, thread-safe, idempotent lease cleanup without finalizers
"""

import collections
import contextlib
from enum import Enum
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set
import uuid
import requests

logger = logging.getLogger("model_lifecycle")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class ModelLeaseType(str, Enum):
    GENERATION = "generation"
    EMBEDDING = "embedding"


class EmbeddingPolicy(str, Enum):
    PERSISTENT = "persistent"
    EPHEMERAL = "ephemeral"
    ADAPTIVE = "adaptive"


class ModelLifecycleError(Exception):
    """Base exception for model lifecycle operations."""
    pass


class ModelLeaseTimeoutError(ModelLifecycleError):
    """Raised when waiting to acquire a model lease exceeds the deadline."""
    pass


class ModelLeaseCancelledError(ModelLifecycleError):
    """Raised when an in-progress wait for a lease is cancelled."""
    pass


class ModelEvictionError(ModelLifecycleError):
    """Raised when model eviction cannot be verified via /api/ps."""
    pass


class StateReconciliationError(ModelLifecycleError):
    """Raised when ground-truth residency cannot be verified from Ollama."""
    pass


def normalize_model_name(name: Optional[str]) -> str:
    """
    Normalize model identifier for consistent internal matching.
    Strips whitespace and converts to lowercase. Preserves explicit tags.
    """
    if not name:
        return ""
    n = name.strip().lower()
    if ":" not in n:
        return f"{n}:latest"
    return n


def model_names_match(a: Optional[str], b: Optional[str]) -> bool:
    """Check if two model identifiers resolve to the same Ollama model."""
    if not a or not b:
        return False
    return normalize_model_name(a) == normalize_model_name(b)


HEAVY_MODEL_PATTERNS = ["16b", "deepseek", "33b", "70b"]


def is_heavy_model(model_name: Optional[str]) -> bool:
    """Check if a model is classified as resource-heavy (e.g. DeepSeek 16B)."""
    if not model_name:
        return False
    n = model_name.lower()
    return any(p in n for p in HEAVY_MODEL_PATTERNS)


class _LeaseState(str, Enum):
    """Atomic ownership states for a ModelLease with an assigned inference worker."""
    ACTIVE = "active"           # Worker assigned; generator still present; either may release
    WORKER_DONE = "worker_done" # Worker signalled completion first; generator will release
    DETACHED = "detached"       # Generator exited first while worker alive; worker will release
    RELEASED = "released"       # Terminal state; lease has been returned to manager


class ModelLease:
    """
    An active, RAII-bound reservation granting inference rights on a resident model.

    Atomic worker-completion/detachment protocol (Stage 3.2.3):
    ─────────────────────────────────────────────────────────────
    A single _state field (protected by _lock) tracks the joint ownership
    between the inference worker thread and the Flask SSE generator:

      ACTIVE → WORKER_DONE  (worker calls mark_worker_finished() first)
                             Generator subsequently calls detach_or_release()
                             → sees WORKER_DONE → claims RELEASED → releases.

      ACTIVE → DETACHED     (generator calls detach_or_release() first)
                             Worker subsequently calls release_from_worker()
                             → sees DETACHED → claims RELEASED → releases.

    In both orderings, exactly one party releases the lease to the manager.
    No lease leak. No premature release. No duplicate manager notification.

    If no worker has been transferred (embedding leases, early non-worker release),
    the original single-step release() path is used.
    """

    def __init__(
        self,
        manager: "OllamaLifecycleManager",
        model_name: str,
        lease_type: ModelLeaseType,
        owner_thread: Optional[threading.Thread] = None,
    ):
        self._manager = manager
        self.model_name = normalize_model_name(model_name)
        self.lease_type = lease_type
        self.id = uuid.uuid4().hex[:8]
        self.acquired_at = time.monotonic()
        self._lock = threading.Lock()
        self._state: _LeaseState = _LeaseState.ACTIVE
        self._worker_thread: Optional[threading.Thread] = None
        self._owner_thread: Optional[threading.Thread] = (
            owner_thread if owner_thread is not None else threading.current_thread()
        )
        self._stuck: bool = False
        self._notified: bool = False
        self._worker_release_authorized: bool = False
        self._notification_completed: bool = False
        self._notification_in_progress: bool = False
        self._notifying_thread: Optional[threading.Thread] = None

    # ── Notification authority ─────────────────────────────────────────────

    def _claim_notification_authority(self) -> bool:
        """
        Atomically claim authority to notify the manager under self._lock.
        Returns True if authority is granted to the calling thread.
        Returns False if notification is already completed or currently in progress
        by another alive thread.
        """
        if self._notification_completed:
            return False
        curr_t = threading.current_thread()
        if self._notification_in_progress:
            if self._notifying_thread is not None and self._notifying_thread.is_alive():
                if self._notifying_thread != curr_t:
                    return False
                return True
        self._notification_in_progress = True
        self._notifying_thread = curr_t
        self._notified = True
        return True

    def _finish_notification(self, success: bool = True) -> None:
        """
        Record the completion or failure of manager notification under self._lock.
        Thread-safe.
        """
        with self._lock:
            self._notification_in_progress = False
            self._notifying_thread = None
            if success:
                self._notification_completed = True
                self._notified = True
            else:
                if not self._notification_completed:
                    self._notified = False

    # ── Worker ownership ───────────────────────────────────────────────────

    @property
    def worker_thread(self) -> Optional[threading.Thread]:
        with self._lock:
            return self._worker_thread

    @property
    def owner_thread(self) -> Optional[threading.Thread]:
        with self._lock:
            return self._owner_thread

    def transfer_to_worker(self, worker_thread: threading.Thread) -> None:
        """Transfer ownership of the lease to the active inference worker thread.
        Must be called before the worker thread starts.
        """
        with self._lock:
            if self._state != _LeaseState.ACTIVE:
                raise RuntimeError(
                    f"Cannot transfer lease {self.id} to worker: state is already {self._state.value}"
                )
            self._worker_thread = worker_thread

    def is_worker_alive(self) -> bool:
        """Check whether the assigned inference worker thread is still alive.
        Note: do NOT use this as the sole synchronization mechanism for release
        decisions. Use mark_worker_finished() / detach_or_release() instead.
        """
        with self._lock:
            return self._worker_thread is not None and self._worker_thread.is_alive()

    # ── Atomic state transitions ───────────────────────────────────────────

    def mark_worker_finished(self) -> bool:
        """
        Called by the inference worker thread when it is about to terminate.

        Atomically validates worker thread identity and transitions state:
          ACTIVE   → WORKER_DONE : generator has not detached yet;
                                   generator will release when it exits.
                                   Returns False (worker must NOT release).
          DETACHED → RELEASED    : generator already detached;
                                   worker claims release authority.
                                   Returns True (worker MUST release).
          Any other state        : no-op, returns False.

        Thread-safe and idempotent.
        Only the registered inference worker may claim completion. Calls from any
        other thread (or before a worker is assigned) are rejected without modifying _state.
        """
        with self._lock:
            curr_t = threading.current_thread()
            if self._worker_thread is None or curr_t != self._worker_thread:
                logger.warning(
                    f"mark_worker_finished() called by unauthorized thread '{curr_t.name}' "
                    f"(expected '{self._worker_thread.name if self._worker_thread else None}') "
                    f"on lease {self.id} ({self.model_name}). Ignoring."
                )
                return False

            if self._state == _LeaseState.ACTIVE:
                # Generator has not exited yet; it will release when it does.
                self._state = _LeaseState.WORKER_DONE
                return False
            if self._state == _LeaseState.DETACHED:
                # Generator already detached; worker is responsible for release.
                self._state = _LeaseState.RELEASED
                self._worker_release_authorized = True
                self._claim_notification_authority()
                return True
            # Already RELEASED or WORKER_DONE from a concurrent call; no-op.
            return False

    def detach_or_release(self) -> bool:
        """
        Called by the Flask SSE generator when it exits (normally or via exception).

        Atomically transitions state:
          ACTIVE (no worker)   → RELEASED  : no worker was assigned (e.g. error before spawn);
                                             generator claims release authority.
                                             Returns True (generator MUST release).
          ACTIVE (with worker) → DETACHED  : worker is still running;
                                             generator delegates release to worker.
                                             Returns False (generator must NOT release).
          WORKER_DONE          → RELEASED  : worker already finished;
                                             generator claims release authority.
                                             Returns True (generator MUST release).
          Any other state                  : no-op, returns False.

        Thread-safe. Idempotent for repeated calls on the same lease.
        """
        with self._lock:
            if self._notification_completed:
                return False
            if self._state == _LeaseState.ACTIVE:
                if self._worker_thread is None:
                    # No worker assigned (error before worker spawn); release directly
                    self._state = _LeaseState.RELEASED
                    return self._claim_notification_authority()
                # Worker assigned and still running. Delegate release to it.
                self._state = _LeaseState.DETACHED
                return False
            if self._state == _LeaseState.WORKER_DONE:
                # Worker already finished. Generator claims release.
                self._state = _LeaseState.RELEASED
                return self._claim_notification_authority()
            # Already RELEASED or DETACHED from a concurrent call; no-op.
            return False

    def release_from_worker(self) -> None:
        """
        Worker-safe explicit release. Verifies the calling thread is the assigned
        inference worker and has successfully completed the authorized DETACHED -> RELEASED
        transition via mark_worker_finished(). Atomically claims manager notification
        and notifies the manager.

        Only valid after mark_worker_finished() returns True.
        Thread-safe and idempotent against duplicate calls.
        """
        with self._lock:
            curr_t = threading.current_thread()
            if self._worker_thread is None or curr_t != self._worker_thread:
                logger.warning(
                    f"release_from_worker() called by unauthorized thread '{curr_t.name}' "
                    f"(expected '{self._worker_thread.name if self._worker_thread else None}') "
                    f"on lease {self.id} ({self.model_name}). Ignoring."
                )
                return
            if not self._worker_release_authorized:
                logger.warning(
                    f"release_from_worker() called on lease {self.id} ({self.model_name}) "
                    f"without prior worker release authorization (state={self._state.value}). Ignoring."
                )
                return
            if self._notification_completed:
                return
            if not self._claim_notification_authority():
                return
        try:
            self._manager._on_lease_released(self)
        except Exception:
            self._finish_notification(success=False)
            raise

    def recover_detached_lease(self, force: bool = False) -> bool:
        """
        Atomically recover a detached lease whose worker thread has genuinely terminated.

        Guarantees:
        - Only operates on leases in DETACHED state.
        - If the worker thread is still alive, recovery is refused (even with force=True),
          a clear warning is logged, and the active lease is preserved.
        - If the worker thread has genuinely terminated (or is None), atomically
          transitions DETACHED -> RELEASED, claims notification authority, and notifies manager.
        - Thread-safe and idempotent against concurrent worker completion or repeated recovery.
        """
        with self._lock:
            if self._notification_completed or self._state != _LeaseState.DETACHED:
                return False

            if self._worker_thread is not None and self._worker_thread.is_alive():
                logger.warning(
                    f"Refusing to recover detached lease {self.id} ({self.model_name}): "
                    f"inference worker '{self._worker_thread.name}' is still executing "
                    f"(force={force}). Preserving active lease."
                )
                return False

            if not self._claim_notification_authority():
                return False

            # Worker has genuinely terminated; claim recovery release atomically.
            self._state = _LeaseState.RELEASED

        try:
            self._manager._on_lease_released(self)
        except Exception:
            self._finish_notification(success=False)
            raise
        return True

    def recover_active_embedding_lease(self) -> bool:
        """
        Atomically check owner thread liveness and claim recovery transition for an embedding lease.
        Executed strictly under self._lock (non-reentrant).

        Safety guarantees:
        - Only operates on EMBEDDING leases.
        - If state is RELEASED with incomplete manager notification:
          claims notification authority and returns True.
        - If state is ACTIVE:
          verifies self._owner_thread is not None and not self._owner_thread.is_alive().
          If owner thread is still alive, recovery is refused and returns False.
          If owner thread has terminated, atomically transitions ACTIVE -> RELEASED,
          claims notification authority, and returns True.
        - If notification is already completed or currently in progress by another alive thread, returns False.
        """
        with self._lock:
            if self.lease_type != ModelLeaseType.EMBEDDING:
                return False
            if self._notification_completed:
                return False

            if self._state == _LeaseState.RELEASED:
                return self._claim_notification_authority()

            if self._state == _LeaseState.ACTIVE:
                if self._owner_thread is None:
                    logger.warning(
                        f"Refusing to recover active embedding lease {self.id} ({self.model_name}): "
                        "no owner thread recorded."
                    )
                    return False
                if self._owner_thread.is_alive():
                    logger.warning(
                        f"Refusing to recover active embedding lease {self.id} ({self.model_name}): "
                        f"owner thread '{self._owner_thread.name}' is still executing. Preserving active lease."
                    )
                    return False

                if not self._claim_notification_authority():
                    return False
                self._state = _LeaseState.RELEASED
                return True

            return False

    # ── Stuck / diagnostic helpers ─────────────────────────────────────────

    def mark_stuck(self) -> None:
        """Mark this lease as having a worker that did not join within timeout."""
        with self._lock:
            self._stuck = True

    @property
    def is_stuck(self) -> bool:
        with self._lock:
            return self._stuck and (self._worker_thread is not None and self._worker_thread.is_alive())

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._state != _LeaseState.RELEASED

    @property
    def state(self) -> _LeaseState:
        with self._lock:
            return self._state

    # ── Legacy / embedding / no-worker release path ────────────────────────

    def release(self) -> None:
        """
        General-purpose idempotent release for leases without an assigned worker
        (embedding leases, error paths before transfer_to_worker is called).

        For leases with an assigned worker, use mark_worker_finished() and
        detach_or_release() instead — they guarantee exactly-once release with
        no TOCTOU races.

        Will refuse to release if a worker is still alive and the caller is not
        that worker, logging a warning and preserving the lease.
        """
        with self._lock:
            if self._notification_completed or self._state == _LeaseState.RELEASED:
                return
            if self._worker_thread is not None and self._worker_thread.is_alive():
                curr_t = threading.current_thread()
                if curr_t != self._worker_thread:
                    logger.warning(
                        f"Non-worker thread '{curr_t.name}' attempted to release lease {self.id} "
                        f"({self.model_name}) while worker '{self._worker_thread.name}' is still alive. "
                        "Preserving lease under worker ownership."
                    )
                    return
            if not self._claim_notification_authority():
                return
            self._state = _LeaseState.RELEASED
        try:
            self._manager._on_lease_released(self)
        except Exception:
            self._finish_notification(success=False)
            raise

    def __enter__(self) -> "ModelLease":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __repr__(self) -> str:
        with self._lock:
            worker_info = f" worker={self._worker_thread.name}" if self._worker_thread else ""
            owner_info = f" owner={self._owner_thread.name}" if self._owner_thread else ""
            stuck_info = " STUCK" if self._stuck else ""
            return f"<ModelLease id={self.id} model={self.model_name} type={self.lease_type.value} state={self._state.value}{worker_info}{owner_info}{stuck_info}>"


class OllamaLifecycleManager:
    """
    Centralized coordinator for local Ollama model memory residency.
    Enforces atomic model transitions, bounded condition-variable waits,
    active lease tracking, and adaptive embedding coordination.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        max_concurrent_generation: Optional[int] = None,
        embedding_policy: Optional[str] = None,
        embedding_model_name: str = "embeddinggemma:latest",
        default_timeout: float = 60.0,
        http_timeout: float = 10.0,
        poll_interval: float = 0.2,
        eviction_timeout: Optional[float] = None,
        eviction_poll_interval: Optional[float] = None,
    ):
        self.base_url = base_url.rstrip("/")
        
        # Concurrency limit for generation models (default 1 for 16 GB laptop)
        if max_concurrent_generation is not None:
            self.max_concurrent_generation = max(1, max_concurrent_generation)
        else:
            env_val = os.environ.get("OLLAMA_MAX_CONCURRENT_GENERATION", "1")
            try:
                self.max_concurrent_generation = max(1, int(env_val))
            except ValueError:
                self.max_concurrent_generation = 1

        # Embedding policy: persistent, ephemeral, or adaptive
        if embedding_policy is not None:
            self.embedding_policy = EmbeddingPolicy(embedding_policy.lower())
        else:
            env_pol = os.environ.get("OLLAMA_EMBEDDING_POLICY", "adaptive").lower()
            try:
                self.embedding_policy = EmbeddingPolicy(env_pol)
            except ValueError:
                self.embedding_policy = EmbeddingPolicy.ADAPTIVE

        self.embedding_model_name = normalize_model_name(embedding_model_name)
        self.default_timeout = default_timeout
        self.http_timeout = http_timeout
        self.poll_interval = poll_interval

        # Eviction verification configuration
        if eviction_timeout is not None:
            self.eviction_timeout = max(0.1, float(eviction_timeout))
        else:
            env_evict_to = os.environ.get("OLLAMA_EVICTION_TIMEOUT")
            if env_evict_to:
                try:
                    self.eviction_timeout = max(0.1, float(env_evict_to))
                except ValueError:
                    self.eviction_timeout = min(5.0, self.default_timeout)
            else:
                self.eviction_timeout = min(5.0, self.default_timeout)

        if eviction_poll_interval is not None:
            self.eviction_poll_interval = max(0.01, float(eviction_poll_interval))
        else:
            env_evict_poll = os.environ.get("OLLAMA_EVICTION_POLL_INTERVAL")
            if env_evict_poll:
                try:
                    self.eviction_poll_interval = max(0.01, float(env_evict_poll))
                except ValueError:
                    self.eviction_poll_interval = self.poll_interval
            else:
                self.eviction_poll_interval = self.poll_interval

        self.session = requests.Session()

        # Synchronization primitives
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        # Active lease registries
        self._active_gen_leases: Dict[str, Set[ModelLease]] = collections.defaultdict(set)
        self._active_embed_leases: Set[ModelLease] = set()

        # Ground-truth tracked residency
        self._resident_gen_model: Optional[str] = None
        self._transitioning: bool = False

    def _query_ollama_ps(self) -> List[str]:
        """
        Query Ollama GET /api/ps to retrieve actual resident models in memory.
        Must NOT be called while holding self._lock.
        Raises StateReconciliationError if query fails (fails closed).
        """
        url = f"{self.base_url}/api/ps"
        try:
            resp = self.session.get(url, timeout=self.http_timeout)
            if resp.status_code != 200:
                raise StateReconciliationError(
                    f"Ollama /api/ps returned HTTP {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name") or m.get("model")
                if name:
                    models.append(normalize_model_name(name))
            return models
        except Exception as exc:
            if isinstance(exc, StateReconciliationError):
                raise
            raise StateReconciliationError(f"Failed to query Ollama /api/ps: {exc}") from exc

    def _evict_model_http(self, model_name: str) -> bool:
        """
        Issue explicit keep_alive: 0 to Ollama and verify removal via bounded /api/ps polling.
        Must NOT be called while holding self._lock.
        Returns True if model was successfully removed, False otherwise.
        """
        norm = normalize_model_name(model_name)
        url = f"{self.base_url}/api/generate"
        try:
            resp = self.session.post(
                url,
                json={"model": model_name, "keep_alive": 0},
                timeout=self.http_timeout,
            )
            if resp.status_code not in (200, 404):
                logger.warning(
                    f"Ollama unload returned HTTP {resp.status_code} for {model_name}: {resp.text}"
                )
        except Exception as exc:
            logger.warning(f"HTTP error attempting to evict {model_name}: {exc}")
            return False

        # Bounded polling verification via /api/ps:
        # After keep_alive: 0, Ollama may report status 'Stopping...' while deallocating memory.
        # A past expires_at or 'Stopping...' status does NOT mean the model has been unloaded;
        # the model is considered evicted ONLY when it completely disappears from /api/ps.
        deadline = time.monotonic() + self.eviction_timeout
        while True:
            try:
                ps_models = self._query_ollama_ps()
                if not any(model_names_match(m, norm) for m in ps_models):
                    logger.info(f"Eviction verified: {model_name} is no longer resident in /api/ps")
                    return True
            except Exception as exc:
                logger.error(f"Failed to query /api/ps during eviction verification of {model_name}: {exc}")
                return False

            if time.monotonic() >= deadline:
                logger.error(
                    f"Eviction verification timed out after {self.eviction_timeout}s: "
                    f"{model_name} is still present in /api/ps"
                )
                return False
            time.sleep(self.eviction_poll_interval)

    def reconcile_state(self) -> Dict[str, Any]:
        """
        Reconcile internal manager state against real Ollama /api/ps ground truth.
        Must be called on startup, after transition failures, or on unexpected errors.
        Fails closed by raising StateReconciliationError if /api/ps is unreachable.
        """
        ps_models = self._query_ollama_ps()
        with self._lock:
            # Reconcile generation model
            if self._resident_gen_model:
                if not any(model_names_match(m, self._resident_gen_model) for m in ps_models):
                    logger.info(
                        f"Reconciliation: model {self._resident_gen_model} is no longer resident in /api/ps"
                    )
                    self._resident_gen_model = None

            if self._resident_gen_model is None:
                for m in ps_models:
                    if not model_names_match(m, self.embedding_model_name):
                        self._resident_gen_model = m
                        logger.info(f"Reconciliation: detected resident generation model in /api/ps: {m}")
                        break

            return {
                "resident_gen_model": self._resident_gen_model,
                "ollama_ps_models": ps_models,
                "active_gen_leases": {k: len(v) for k, v in self._active_gen_leases.items() if len(v) > 0},
                "active_embed_leases": len(self._active_embed_leases),
                "transitioning": self._transitioning,
            }

    def verify_pipeline_handoff(
        self,
        prior_lease: Optional[ModelLease],
        prior_model: str,
        timeout: Optional[float] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Verify all prerequisites before transitioning from Stage 1 to Stage 2 in multi-stage pipelines:
        1. Stage 1 inference worker thread has terminated.
        2. Stage 1 generation lease is completely absent from manager tracking.
        3. Stage 1 model is absent from Ollama /api/ps (evicted and verified).

        Raises ModelLeaseCancelledError if cancellation is signaled.
        Raises ModelLifecycleError if worker remains alive or lease tracking is not cleared.
        Raises ModelEvictionError if prior model cannot be evicted from /api/ps.
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        deadline = time.monotonic() + effective_timeout

        if is_cancelled and is_cancelled():
            raise ModelLeaseCancelledError("Pipeline handoff cancelled before verification")

        # ── 1. Verify Stage 1 worker termination ──
        if prior_lease is not None:
            w = prior_lease.worker_thread
            if w is not None and w.is_alive():
                remaining = max(0.1, deadline - time.monotonic())
                w.join(timeout=min(remaining, 5.0))
            if prior_lease.is_worker_alive():
                thr_name = getattr(w, "name", "unknown") if w else "unknown"
                raise ModelLifecycleError(
                    f"Stage 1 inference worker '{thr_name}' failed to terminate within timeout before Stage 2 handoff"
                )

        if is_cancelled and is_cancelled():
            raise ModelLeaseCancelledError("Pipeline handoff cancelled after worker termination check")

        # ── 2. Verify Stage 1 lease removal from manager tracking ──
        norm_prior = normalize_model_name(prior_model)
        while True:
            if is_cancelled and is_cancelled():
                raise ModelLeaseCancelledError("Pipeline handoff cancelled during lease removal check")
            with self._cv:
                active_leases = self._active_gen_leases.get(norm_prior, set())
                is_absent = (prior_lease is None) or (
                    prior_lease not in active_leases and prior_lease.state == _LeaseState.RELEASED
                )
                if is_absent and len(active_leases) == 0:
                    break

            if time.monotonic() >= deadline:
                state_str = prior_lease.state.value if prior_lease else "none"
                raise ModelLifecycleError(
                    f"Stage 1 lease '{getattr(prior_lease, 'id', 'none')}' ({norm_prior}) was not cleanly removed "
                    f"from manager tracking before Stage 2 handoff (lease_state='{state_str}')"
                )
            time.sleep(self.poll_interval)

        if is_cancelled and is_cancelled():
            raise ModelLeaseCancelledError("Pipeline handoff cancelled after lease removal check")

        # ── 3. Verify Stage 1 model absence from Ollama /api/ps ──
        ps_models = self._query_ollama_ps()
        if any(model_names_match(m, norm_prior) for m in ps_models):
            logger.info(f"Pipeline handoff: evicting Stage 1 model '{prior_model}' before Stage 2 start...")
            evicted = self._evict_model_http(prior_model)
            if not evicted:
                raise ModelEvictionError(
                    f"Failed to evict Stage 1 model '{prior_model}' before Stage 2 handoff; model remains in /api/ps"
                )
            # Re-query to guarantee absence
            ps_after = self._query_ollama_ps()
            if any(model_names_match(m, norm_prior) for m in ps_after):
                raise ModelEvictionError(
                    f"Stage 1 model '{prior_model}' is still resident in Ollama /api/ps after eviction"
                )
            logger.info(f"Pipeline handoff: successfully verified absence of '{prior_model}' from /api/ps")

    @contextlib.contextmanager
    def acquire_generation_lease(
        self,
        target_model: str,
        timeout: Optional[float] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        """
        Atomically reserve a generation model for inference.
        Guarantees:
        1. target_model will NOT be evicted while this lease is active.
        2. Any different generation model is evicted before granting this lease.
        3. Under adaptive policy, active embedding leases drain and EmbeddingGemma is evicted
           if target_model is heavy (e.g. DeepSeek 16B).
        4. Bounded polling with monotonic deadline and cancellation checking.
        5. Guaranteed release on exit or exception.
        """
        norm_target = normalize_model_name(target_model)
        effective_timeout = timeout if timeout is not None else self.default_timeout
        deadline = time.monotonic() + effective_timeout

        lease = None
        blocking_condition = "initial"
        while True:
            with self._cv:
                while True:
                    now = time.monotonic()
                    remaining = deadline - now
                    if remaining <= 0:
                        elapsed = round(time.monotonic() - (deadline - effective_timeout), 1)
                        active_gen_counts = {k: len(v) for k, v in self._active_gen_leases.items() if len(v) > 0}
                        raise ModelLeaseTimeoutError(
                            f"Timed out waiting to acquire generation lease for '{norm_target}' after {elapsed}s "
                            f"[blocking_condition='{blocking_condition}', resident='{self._resident_gen_model}', "
                            f"active_gen={active_gen_counts}, active_embed={len(self._active_embed_leases)}, "
                            f"transitioning={self._transitioning}]"
                        )
                    if is_cancelled and is_cancelled():
                        raise ModelLeaseCancelledError(
                            f"Acquisition of generation lease for {norm_target} was cancelled"
                        )

                    # A. If another thread is executing a model transition, wait
                    if self._transitioning:
                        blocking_condition = "transition_in_progress"
                        self._cv.wait(timeout=min(remaining, self.poll_interval))
                        continue

                    # B. Under adaptive policy, heavy models (DeepSeek 16B) must wait for embedding leases to drain
                    if self.embedding_policy == EmbeddingPolicy.ADAPTIVE and is_heavy_model(norm_target):
                        if len(self._active_embed_leases) > 0:
                            blocking_condition = f"waiting_for_embedding_leases_to_drain(count={len(self._active_embed_leases)})"
                            self._cv.wait(timeout=min(remaining, self.poll_interval))
                            continue

                    # C. Check if target model is already resident
                    needs_adaptive_embed_check = (
                        self.embedding_policy == EmbeddingPolicy.ADAPTIVE
                        and is_heavy_model(norm_target)
                        and len(self._active_gen_leases[norm_target]) == 0
                    )

                    if self._resident_gen_model == norm_target and not needs_adaptive_embed_check:
                        current_active = len(self._active_gen_leases[norm_target])
                        if current_active < self.max_concurrent_generation:
                            # Target model is resident and has capacity -> Grant lease!
                            lease = ModelLease(self, norm_target, ModelLeaseType.GENERATION)
                            self._active_gen_leases[norm_target].add(lease)
                            break
                        else:
                            # Same model, but at concurrency limit -> Queue and wait
                            blocking_condition = f"concurrency_limit_reached(active={current_active}, max={self.max_concurrent_generation})"
                            self._cv.wait(timeout=min(remaining, self.poll_interval))
                            continue

                    # D. Target model differs from resident model (switch required)
                    if self._resident_gen_model is not None:
                        active_count = len(self._active_gen_leases[self._resident_gen_model])
                        if active_count > 0:
                            # Resident model is actively streaming in another thread -> Wait for drain
                            blocking_condition = f"active_lease_on_resident_model(resident='{self._resident_gen_model}', active={active_count})"
                            self._cv.wait(timeout=min(remaining, self.poll_interval))
                            continue

                    # No active leases on resident model. Claim transition lock!
                    self._transitioning = True
                    break

            # If lease was granted on fast path, break outer loop
            if lease is not None:
                break

            # Perform physical transition outside the lock
            try:
                # 1. Query Ollama /api/ps ground truth
                ps_models = self._query_ollama_ps()

                # 2. Evict any non-target models present in Ollama
                for resident_m in ps_models:
                    if model_names_match(resident_m, norm_target):
                        continue
                    # Check embedding model under adaptive policy
                    if model_names_match(resident_m, self.embedding_model_name):
                        if self.embedding_policy == EmbeddingPolicy.ADAPTIVE and is_heavy_model(norm_target):
                            evict_ok = self._evict_model_http(resident_m)
                            if not evict_ok:
                                raise ModelEvictionError(
                                    f"Failed to evict embedding model {resident_m} before loading heavy model {norm_target}"
                                )
                        continue

                    # Generation or unmanaged external model
                    evict_ok = self._evict_model_http(resident_m)
                    if not evict_ok:
                        raise ModelEvictionError(
                            f"Failed to evict resident model {resident_m}; cannot safely load {norm_target}"
                        )

                # 3. Verify clean state after eviction
                ps_models_after = self._query_ollama_ps()
                for resident_m in ps_models_after:
                    if not model_names_match(resident_m, norm_target):
                        if not (model_names_match(resident_m, self.embedding_model_name) and not is_heavy_model(norm_target)):
                            raise ModelEvictionError(
                                f"Post-eviction check showed {resident_m} still resident in Ollama"
                            )

                # 4. Transition verified! Re-acquire lock to commit state and grant lease
                with self._cv:
                    self._resident_gen_model = norm_target
                    lease = ModelLease(self, norm_target, ModelLeaseType.GENERATION)
                    self._active_gen_leases[norm_target].add(lease)
                    break

            finally:
                with self._cv:
                    self._transitioning = False
                    self._cv.notify_all()

        try:
            yield lease
        finally:
            if lease is not None:
                should_release = lease.detach_or_release()
                if should_release:
                    # Worker finished before we got here (or no worker assigned); release.
                    try:
                        lease._manager._on_lease_released(lease)
                    except Exception:
                        lease._finish_notification(success=False)
                        raise
                else:
                    if lease._worker_thread is not None:
                        logger.warning(
                            f"Generator exited while worker {lease._worker_thread.name} is alive. "
                            f"Retaining lease {lease.id} ({lease.model_name}) under worker ownership."
                        )

    @contextlib.contextmanager
    def acquire_embedding_lease(
        self,
        timeout: Optional[float] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        """
        Atomically reserve rights to execute an embedding operation.
        Guarantees:
        1. Under adaptive policy, waits if a heavy generation model (DeepSeek 16B) is actively leasing.
        2. Bounded polling with monotonic deadline and cancellation checking.
        3. Respects persistent, ephemeral, or adaptive lifecycle policies.
        4. Guaranteed release on exit or exception.
        """
        norm_embed = self.embedding_model_name
        effective_timeout = timeout if timeout is not None else self.default_timeout
        deadline = time.monotonic() + effective_timeout

        lease = None
        blocking_condition = "initial"
        while True:
            with self._cv:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    elapsed = round(time.monotonic() - (deadline - effective_timeout), 1)
                    active_gen_counts = {k: len(v) for k, v in self._active_gen_leases.items() if len(v) > 0}
                    raise ModelLeaseTimeoutError(
                        f"Timed out waiting to acquire embedding lease for '{norm_embed}' after {elapsed}s "
                        f"[blocking_condition='{blocking_condition}', resident='{self._resident_gen_model}', "
                        f"active_gen={active_gen_counts}, active_embed={len(self._active_embed_leases)}, "
                        f"transitioning={self._transitioning}]"
                    )
                if is_cancelled and is_cancelled():
                    raise ModelLeaseCancelledError(
                        f"Acquisition of embedding lease for {norm_embed} was cancelled"
                    )

                # A. Under adaptive policy: if a heavy generation model is actively leasing, wait
                if self.embedding_policy == EmbeddingPolicy.ADAPTIVE:
                    heavy_active = False
                    for gen_m, leases in self._active_gen_leases.items():
                        if len(leases) > 0 and is_heavy_model(gen_m):
                            heavy_active = True
                            blocking_condition = f"heavy_generation_model_active({gen_m})"
                            break
                    if heavy_active:
                        self._cv.wait(timeout=min(remaining, self.poll_interval))
                        continue

                # B. If a transition is underway targeting a heavy model, wait
                if self._transitioning:
                    blocking_condition = "transition_in_progress"
                    self._cv.wait(timeout=min(remaining, self.poll_interval))
                    continue

                # Grant embedding lease!
                lease = ModelLease(
                    self,
                    norm_embed,
                    ModelLeaseType.EMBEDDING,
                    owner_thread=threading.current_thread(),
                )
                self._active_embed_leases.add(lease)
                break

        try:
            yield lease
        finally:
            if lease is not None:
                lease.release()

    def _on_lease_released(self, lease: ModelLease) -> None:
        """Called by ModelLease release paths under thread safety."""
        should_evict_ephemeral = False
        bookkeeping_done = False
        try:
            with self._cv:
                if lease.lease_type == ModelLeaseType.GENERATION:
                    if lease.model_name in self._active_gen_leases:
                        self._active_gen_leases[lease.model_name].discard(lease)
                elif lease.lease_type == ModelLeaseType.EMBEDDING:
                    self._active_embed_leases.discard(lease)
                    if self.embedding_policy == EmbeddingPolicy.EPHEMERAL and len(self._active_embed_leases) == 0:
                        should_evict_ephemeral = True
                bookkeeping_done = True
                self._cv.notify_all()
        finally:
            if bookkeeping_done:
                lease._finish_notification(success=True)

        if should_evict_ephemeral:
            try:
                self._evict_model_http(lease.model_name)
            except Exception as e:
                logger.warning(f"Notice during ephemeral embedding eviction: {e}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return snapshot of internal state and active leases for monitoring and tests."""
        with self._lock:
            stuck_workers = []
            for model, leases in self._active_gen_leases.items():
                for lease in leases:
                    if lease.is_stuck:
                        wt = lease.worker_thread
                        stuck_workers.append({
                            "lease_id": lease.id,
                            "model": lease.model_name,
                            "worker_name": wt.name if wt else None,
                            "worker_alive": lease.is_worker_alive(),
                        })

            return {
                "resident_gen_model": self._resident_gen_model,
                "transitioning": self._transitioning,
                "active_gen_leases": {k: len(v) for k, v in self._active_gen_leases.items() if len(v) > 0},
                "active_embed_leases": len(self._active_embed_leases),
                "stuck_workers": stuck_workers,
                "max_concurrent_generation": self.max_concurrent_generation,
                "embedding_policy": self.embedding_policy.value,
            }

    def recover_stuck_leases(self, force: bool = False) -> int:
        """
        Sweep active generation and embedding leases and recover any eligible detached leases
        whose worker threads have genuinely terminated, clean up incomplete releases,
        or recover active embedding leases whose owner threads have terminated.

        Safety guarantees:
        - Only recovers leases in DETACHED state whose worker has terminated,
          leases in RELEASED state whose notification did not complete,
          or active embedding leases whose recorded owner thread has terminated.
        - Refuses recovery if the worker/owner thread is still alive, even with force=True.
          (Never silently releases a lease while inference/embedding is still executing).
        - Idempotent and thread-safe: claims notification authority atomically;
          cannot double-notify the manager or race with worker/generator/owner completion.
        - Preserves manager tracking for active leases with living workers/owners.
        """
        released_count = 0
        with self._cv:
            gen_leases_to_check = []
            for model, leases in list(self._active_gen_leases.items()):
                for lease in list(leases):
                    gen_leases_to_check.append(lease)
            embed_leases_to_check = list(self._active_embed_leases)

        for lease in gen_leases_to_check:
            with lease._lock:
                state = lease._state
            if state == _LeaseState.RELEASED:
                with lease._lock:
                    can_recover = lease._claim_notification_authority()
                if can_recover:
                    try:
                        self._on_lease_released(lease)
                        released_count += 1
                    except Exception:
                        lease._finish_notification(success=False)
                        raise
            elif lease.recover_detached_lease(force=force):
                released_count += 1

        for lease in embed_leases_to_check:
            if lease.recover_active_embedding_lease():
                try:
                    self._on_lease_released(lease)
                    released_count += 1
                except Exception:
                    lease._finish_notification(success=False)
                    raise

        return released_count


_global_model_manager: Optional[OllamaLifecycleManager] = None
_global_manager_lock = threading.Lock()


def get_model_manager() -> OllamaLifecycleManager:
    """Retrieve global singleton OllamaLifecycleManager."""
    global _global_model_manager
    if _global_model_manager is None:
        with _global_manager_lock:
            if _global_model_manager is None:
                _global_model_manager = OllamaLifecycleManager()
    return _global_model_manager


def reset_model_manager_for_testing(
    new_manager: Optional[OllamaLifecycleManager] = None,
) -> None:
    """Reset global manager instance for isolated unit tests."""
    global _global_model_manager
    with _global_manager_lock:
        _global_model_manager = new_manager
