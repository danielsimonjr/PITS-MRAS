r"""Parallel thread architecture for deployment (IP §9.2).

Owning phase: Phase 6 (Inference Engine).

Reference skeleton for the three-thread deployment topology named in
ARCHITECTURE.md §9.2:

* ``ControlThread``    -- nominally 1 kHz, calls ``engine.step()`` every cycle
  and publishes the latest ``u_safe`` / monitoring values into the shared
  ``ControllerState`` under a lock. It NEVER blocks on the adaptation thread.
* ``AdaptationThread`` -- nominally 100 Hz, performs the slow critic update on a
  ``copy.deepcopy`` of the critic and atomically swaps it back (double-buffer
  pattern, so a control-thread read never sees a half-updated critic).
* ``MonitorThread``    -- nominally 10 Hz, snapshots :math:`\hat V`, ``||e||``,
  ``h_CBF`` and the CBF-activation rate for logging.

Graceful shutdown is via a single ``threading.Event``; ``stop()`` is idempotent.
Per the spec, "a reference skeleton is sufficient": the threads tick safely,
share state through a lock, and join cleanly -- this is a deployment scaffold,
not a tuned hard-real-time scheduler. The adaptation step here is a no-op
double-buffer swap placeholder (the real IRL update lands with the deployment
work); the structural guarantee (deepcopy -> update -> atomic swap) is wired so
the control thread is never exposed to a partially-updated critic.
"""

import copy
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor

from pits_mras.inference.realtime import RealtimeInferenceEngine


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


class ParallelInferenceEngine:
    """Three-thread deployment skeleton around a ``RealtimeInferenceEngine``.

    Args:
        engine: the closed-loop engine the control thread drives.
        x_p: fixed plant state fed to ``engine.step`` each cycle (the skeleton
            does not model a live plant; a real deployment would read a sensor).
        r: fixed reference command fed each cycle.
        dt: integration timestep passed to ``engine.step``.
        control_hz / adapt_hz / monitor_hz: nominal thread rates. Kept modest in
            tests so the skeleton ticks a few cycles quickly without busy-waiting.
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
    ) -> None:
        self.engine = engine
        self._x_p = x_p
        self._r = r
        self._dt = dt
        self._control_period = 1.0 / control_hz
        self._adapt_period = 1.0 / adapt_hz
        self._monitor_period = 1.0 / monitor_hz

        self.state = ControllerState()
        self._state_lock = threading.Lock()
        # Guards the critic double-buffer swap so a control-thread read never
        # observes a half-updated critic.
        self._critic_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._threads: List[threading.Thread] = []

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
            self._shutdown.wait(self._control_period)

    def _adaptation_loop(self) -> None:
        while not self._shutdown.is_set():
            # Double-buffer pattern: copy the critic, "update" the copy, then
            # atomically swap it back. The control thread only ever reads the
            # live critic under ``_critic_lock``.
            critic_copy = copy.deepcopy(self.engine.controller.critic)
            # (placeholder for the IRL Bellman update on ``critic_copy``)
            with self._critic_lock:
                self.engine.controller.critic = critic_copy
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
    def start(self) -> None:
        """Launch the three daemon threads."""
        if self._threads:
            raise RuntimeError("ParallelInferenceEngine already started.")
        self._shutdown.clear()
        for name, target in (
            ("ControlThread", self._control_loop),
            ("AdaptationThread", self._adaptation_loop),
            ("MonitorThread", self._monitor_loop),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def wait(self, seconds: float) -> None:
        """Block the caller for ``seconds`` while the threads run."""
        time.sleep(seconds)

    def is_alive(self) -> bool:
        """True if any worker thread is still running."""
        return any(t.is_alive() for t in self._threads)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal shutdown and join all threads (idempotent)."""
        self._shutdown.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads = [t for t in self._threads if t.is_alive()]
