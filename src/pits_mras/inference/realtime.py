r"""Real-time closed-loop inference engine (IP §9.1).

Owning phase: Phase 6 (Inference Engine).

Implements the seven-step closed loop named in ARCHITECTURE.md §9.1:

1. Measure plant state ``x_p``.
2. Lazily initialize the reference-model state and the bounded ``deque``
   history buffers on the first call.
3. PITNN forward pass -> dynamics prediction :math:`\hat f_\theta`.
4. Reference-model step -> tracking error ``e = x_p - x_m`` (canonical
   ``C_p = I`` plant output).
5. Controller forward pass -> nominal control.
6. CBF safety filter -> ``u_safe`` (replaces the heuristic :math:`\dot V < 0`
   check; active inside ``MRASController.forward`` when the filter is set up).
7. Update the bounded history with the applied ``u_safe`` and return the
   monitoring dict ``{u_safe, e, v_hat, h_cbf, f_hat, cbf_active}``.

The whole ``step`` is decorated ``@torch.no_grad()`` and guarded by a
``threading.Lock`` so a 1 kHz control thread can call it safely while a slower
adaptation thread mutates shared model parameters.

Adaptation note (real Phase 1-5 APIs vs. the §9 spec text)
----------------------------------------------------------
The implementation-plan §9.1 listing assumes a controller signature
``controller(e, x_p, x_m, r, context)`` returning ``u_safe`` / ``V_hat`` /
``cbf_slack``, and a PITNN call ``pitnn(x_hist, u_hist, x_p, u_prev, e, e_hist)``.
The *real* Phase-4 controller is ``MRASController.forward(e, r, x_plant,
apply_safety=True)`` returning ``{u_nom, u, h_cbf, slack}`` -- it neither takes a
``context`` nor computes the value function. So this engine:

* computes the value :math:`\hat V(e)` itself via ``controller.critic`` (the
  spec's ``V_hat`` -> the real lowercase ``v_hat`` key);
* reads the applied control from ``ctrl_out["u"]`` (the real key; exported as
  ``u_safe``) and the CBF activation from ``ctrl_out["slack"]``;
* feeds the PITNN the real six-argument signature
  ``(x_hist, u_hist, x_p_curr, u_curr, e_curr, e_hist)``.
"""

import threading
from collections import deque
from typing import Any, Deque, Dict, Optional

import torch
from torch import Tensor

from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.models.pitnn import PITNN


class RealtimeInferenceEngine:
    """Thread-safe real-time closed-loop control engine (IP §9.1)."""

    def __init__(
        self,
        pitnn: PITNN,
        controller: MRASController,
        ref_model: LinearReferenceModel,
        horizon: int = 50,
        device: str = "cpu",
    ) -> None:
        self.pitnn = pitnn.eval()
        self.controller = controller.eval()
        self.ref_model = ref_model
        self.horizon = horizon
        self.device = torch.device(device)

        self._x_hist: Deque[Tensor] = deque(maxlen=horizon)
        self._u_hist: Deque[Tensor] = deque(maxlen=horizon)
        self._e_hist: Deque[Tensor] = deque(maxlen=horizon)
        # Reference-model state, lazily initialized on the first ``step``.
        self._x_m: Optional[Tensor] = None
        self._lock = threading.Lock()

        self._n_state = int(ref_model.A_m.shape[0])
        self._n_ctrl = int(ref_model.B_m.shape[1])

    @torch.no_grad()
    def step(self, x_p: Tensor, r: Tensor, dt: float = 0.01) -> Dict[str, Any]:
        r"""Execute one control cycle.

        Args:
            x_p: ``[state_dim]`` current plant state (no batch dim).
            r: ``[ref_dim]`` current reference command.
            dt: integration timestep.

        Returns:
            dict with ``u_safe`` ``[control_dim]``, ``e`` ``[state_dim]``,
            ``v_hat`` (scalar), ``h_cbf`` (scalar), ``f_hat`` ``[output_dim]``,
            and ``cbf_active`` (``bool``).
        """
        x_p = x_p.to(self.device)
        r = r.to(self.device)

        with self._lock:
            if self._x_m is None:
                self._x_m = x_p.clone()
                # Seed the history buffers so the first PITNN forward has a
                # full window of the requested horizon length.
                zero_u = torch.zeros(self._n_ctrl, device=self.device)
                zero_e = torch.zeros_like(x_p)
                for _ in range(self.horizon):
                    self._x_hist.append(x_p.clone())
                    self._u_hist.append(zero_u.clone())
                    self._e_hist.append(zero_e.clone())

            # Build batched history tensors: [1, T, dim].
            x_hist = torch.stack(list(self._x_hist)).unsqueeze(0)
            u_hist = torch.stack(list(self._u_hist)).unsqueeze(0)
            e_hist = torch.stack(list(self._e_hist)).unsqueeze(0)

            # Reference-model step (batched) + tracking error (C_p = I).
            self._x_m = self.ref_model.step(
                self._x_m.unsqueeze(0), r.unsqueeze(0), dt
            ).squeeze(0)
            e = x_p - self._x_m  # [state_dim]

            # PITNN forward (real six-arg signature). The port-Hamiltonian
            # decoder computes its dynamics via an *internal*
            # ``torch.autograd.grad`` (dH/dq, dH/dp); its ``qp.requires_grad_``
            # is a no-op under ``torch.no_grad()``, so re-enable grad just for
            # this forward pass, then ``detach`` the output -- no training graph
            # escapes the step.
            u_prev = self._u_hist[-1]
            with torch.enable_grad():
                pitnn_out = self.pitnn(
                    x_hist,
                    u_hist,
                    x_p.unsqueeze(0),
                    u_prev.unsqueeze(0),
                    e.unsqueeze(0),
                    e_hist,
                )
            f_hat = pitnn_out["f_hat"].detach().squeeze(0)  # [output_dim]

            # Controller forward (real signature: e, r, x_plant).
            ctrl_out = self.controller(e.unsqueeze(0), r.unsqueeze(0), x_p.unsqueeze(0))
            u_safe = ctrl_out["u"].squeeze(0)  # [control_dim]

            # CBF activation: the safety filter populates "slack" only when the
            # filter is set up; treat any non-trivial correction as active.
            if "slack" in ctrl_out:
                cbf_active = bool((ctrl_out["slack"] > 1e-4).any().item())
                h_cbf = ctrl_out["h_cbf"].squeeze()
            else:
                cbf_active = False
                h_cbf = torch.zeros((), device=self.device)

            # Value function V_hat(e) computed here (controller does not).
            v_hat = self.controller.critic(e.unsqueeze(0)).squeeze()

            # Update bounded history with the applied control.
            self._x_hist.append(x_p.clone())
            self._u_hist.append(u_safe.detach().clone())
            self._e_hist.append(e.detach().clone())

        return {
            "u_safe": u_safe,
            "e": e,
            "v_hat": v_hat,
            "h_cbf": h_cbf,
            "f_hat": f_hat,
            "cbf_active": cbf_active,
        }
