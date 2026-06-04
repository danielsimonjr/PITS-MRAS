r"""Parallel thread architecture for deployment (IP §9.2).

Owning phase: Phase 6 (Inference Engine).

Three-thread deployment scaffold for the topology named in ARCHITECTURE.md §9.2:

* ``ControlThread``    -- nominally 1 kHz, calls ``engine.step()`` every cycle,
  publishes the latest ``u_safe`` / monitoring values into the shared
  ``ControllerState`` under a lock, and feeds the per-cycle ``(e, u_safe)`` into
  a bounded window for the adaptation thread. It never blocks on the adaptation
  thread's compute.
* ``AdaptationThread`` -- nominally 100 Hz, performs a **real** Integral-RL
  critic update (one Adam step of :class:`~pits_mras.losses.irl.IRLBellmanLoss`)
  on a ``copy.deepcopy`` of the critic, then atomically swaps it back
  (double-buffer pattern, so a control-thread read never sees a half-updated
  critic). The deepcopy + gradient step happen *off* the critic lock; only the
  pointer swap is locked.
* ``MonitorThread``    -- nominally 10 Hz, snapshots the CBF-activation rate for
  logging.

**Robustness:** each thread body is guarded -- the first exception is captured
(see :attr:`error` / :meth:`check`) and triggers a fail-fast shutdown rather than
a silently-dead daemon thread. Graceful shutdown is via a single
``threading.Event``; ``stop()`` is idempotent.

**Still scaffold (honest):** ``x_p`` / ``r`` are fixed per cycle (a real
deployment reads a live sensor / command); the scheduler is cooperative
``Event.wait(period)``, not a hard-real-time scheduler; the CBF matrix ``P`` is
the one fixed at ``setup_safety_filter`` time (the adaptation updates the value /
costate path, not the safety set).
"""

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import torch
from torch import Tensor

from pits_mras.inference.realtime import RealtimeInferenceEngine
from pits_mras.losses.irl import IRLBellmanLoss


@dataclass
class ControllerState:
    """Shared, lock-protected snapshot of the latest control cycle (§9.2)."""

    u_safe: Optional[Tensor] = None
    e_norm: float = 0.0
    v_hat: float = 0.0
    h_cbf: float = 0.0
    cbf_active: bool = False
    cbf_activations: int = 0
    step_count: int = 0
    cbf_activation_rate: float = 0.0
    adaptation_swaps: int = 0


class ParallelInferenceEngine:
    """Three-thread deployment engine around a ``RealtimeInferenceEngine``.

    Args:
        engine: the closed-loop engine the control thread drives.
        x_p: fixed plant state fed to ``engine.step`` each cycle (the scaffold
            does not model a live plant; a real deployment would read a sensor).
        r: fixed reference command fed each cycle.
        dt: integration timestep passed to ``engine.step``.
        control_hz / adapt_hz / monitor_hz: nominal thread rates. Kept modest in
            tests so the threads tick a few cycles quickly without busy-waiting.
        irl_window: number of control cycles the adaptation update fits over.
        adapt_lr: Adam learning rate for the adaptation critic step.
    """

    def __init__(
        self,
        engine: RealtimeInferenceEngine,
        x_p: Tensor,
        r: Tensor,
        dt: float = 0.01,
        control_hz: float = 1000.0,
        adapt_hz: float = 100.0,
        monitor_hz: float = 10.0,
        irl_window: int = 8,
        adapt_lr: float = 1e-3,
    ) -> None:
        self.engine = engine
        self._x_p = x_p
        self._r = r
        self._dt = dt
        self._control_period = 1.0 / control_hz
        self._adapt_period = 1.0 / adapt_hz
        self._monitor_period = 1.0 / monitor_hz
        self._adapt_lr = adapt_lr

        self.state = ControllerState()
        self._state_lock = threading.Lock()
        # Guards the critic double-buffer swap so a control-thread read never
        # observes a half-updated critic.
        self._critic_lock = threading.Lock()
        # Rolling (e, u_safe) window the control thread publishes for adaptation.
        self._window_cap = irl_window + 1
        self._window: Deque[Tuple[Tensor, Tensor]] = deque(maxlen=self._window_cap)
        self._window_lock = threading.Lock()
        self._irl = IRLBellmanLoss(engine.ref_model.Q, engine.ref_model.R)

        self._shutdown = threading.Event()
        self._threads: List[threading.Thread] = []
        self._error: Optional[BaseException] = None
        self._failed_thread: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Thread bodies.
    # ------------------------------------------------------------------ #
    def _control_loop(self) -> None:
        while not self._shutdown.is_set():
            with self._critic_lock:
                out = self.engine.step(self._x_p, self._r, self._dt)
            with self._state_lock:
                self.state.u_safe = out["u_safe"]
                self.state.e_norm = float(out["e"].norm().item())
                self.state.v_hat = float(out["v_hat"].item())
                self.state.h_cbf = float(out["h_cbf"].item())
                self.state.cbf_active = bool(out["cbf_active"])
                self.state.step_count += 1
                if out["cbf_active"]:
                    self.state.cbf_activations += 1
            with self._window_lock:
                self._window.append(
                    (out["e"].detach().reshape(-1), out["u_safe"].detach().reshape(-1))
                )
            self._shutdown.wait(self._control_period)

    def _adaptation_update(self) -> bool:
        """Run one double-buffered IRL critic update; return ``True`` if applied.

        Snapshots the rolling window; if it is not yet full, does nothing and
        returns ``False``. Otherwise fits a ``deepcopy`` of the critic with one
        IRL Bellman gradient step and atomically swaps it in (the heavy work is
        off the critic lock; only the swap is locked). Factored out so it can be
        unit-tested without the threads.
        """
        with self._window_lock:
            if len(self._window) < self._window_cap:
                return False
            e_list = [e for e, _ in self._window]
            u_list = [u for _, u in self._window]
        e_win = torch.stack(e_list).unsqueeze(0)  # [1, W+1, state_dim]
        u_win = torch.stack(u_list).unsqueeze(0)  # [1, W+1, control_dim]

        with self._critic_lock:
            # Deepcopy under the lock (the quadratic critic is tiny -> microseconds)
            # so the copy never races a concurrent control-thread forward pass; the
            # heavy gradient step below runs OFF the lock (the double-buffer point).
            critic_copy = copy.deepcopy(self.engine.controller.critic)
        optimizer = torch.optim.Adam(critic_copy.parameters(), lr=self._adapt_lr)
        optimizer.zero_grad()
        loss = self._irl(critic_copy, e_win, u_win, self._dt)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic_copy.parameters(), max_norm=1.0)
        optimizer.step()

        with self._critic_lock:
            # Atomic double-buffer swap of BOTH critic references (the costate
            # head wraps the critic, so it must point at the new copy too).
            self.engine.controller.critic = critic_copy
            self.engine.controller.costate_head.critic = critic_copy
        with self._state_lock:
            self.state.adaptation_swaps += 1
        return True

    def _adaptation_loop(self) -> None:
        while not self._shutdown.is_set():
            self._adaptation_update()
            self._shutdown.wait(self._adapt_period)

    def _monitor_loop(self) -> None:
        while not self._shutdown.is_set():
            with self._state_lock:
                if self.state.step_count > 0:
                    self.state.cbf_activation_rate = (
                        self.state.cbf_activations / self.state.step_count
                    )
            self._shutdown.wait(self._monitor_period)

    # ------------------------------------------------------------------ #
    # Lifecycle.
    # ------------------------------------------------------------------ #
    def _guard(self, name: str, loop: Callable[[], None]) -> None:
        """Run a thread loop; capture the first exception and fail fast."""
        try:
            loop()
        except BaseException as exc:  # noqa: BLE001  (capture + surface, then stop)
            with self._state_lock:
                if self._error is None:
                    self._error = exc
                    self._failed_thread = name
            self._shutdown.set()

    def start(self) -> None:
        """Launch the three daemon threads."""
        if self._threads:
            raise RuntimeError("ParallelInferenceEngine already started.")
        self._shutdown.clear()
        self._error = None
        self._failed_thread = None
        for name, target in (
            ("ControlThread", self._control_loop),
            ("AdaptationThread", self._adaptation_loop),
            ("MonitorThread", self._monitor_loop),
        ):
            thread = threading.Thread(
                target=self._guard, args=(name, target), name=name, daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def wait(self, seconds: float) -> None:
        """Block the caller for ``seconds`` while the threads run."""
        time.sleep(seconds)

    def is_alive(self) -> bool:
        """True if any worker thread is still running."""
        return any(t.is_alive() for t in self._threads)

    @property
    def error(self) -> Optional[BaseException]:
        """The first exception raised by any thread body, or ``None``."""
        return self._error

    def check(self) -> None:
        """Re-raise the first captured thread exception, if any.

        Intended for the ``stop(); check()`` pattern (most reliable once the
        threads have joined); a failure also fail-fast-stops the engine on its own.
        """
        if self._error is not None:
            raise self._error

    def stop(self, timeout: float = 5.0) -> None:
        """Signal shutdown and join all threads (idempotent)."""
        self._shutdown.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads = [t for t in self._threads if t.is_alive()]
