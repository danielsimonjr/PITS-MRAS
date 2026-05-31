r"""Port-Hamiltonian utilities (IP §4.4).

Owning phase: Phase 1 (Foundation Layer).

Connection 2 foundation (storage = value). Four functions (IP §4.4):
``make_skew_symmetric`` (:math:`J=-J^\top`), ``make_positive_definite``
(:math:`R_\theta=L^\top L\succeq 0`), ``port_hamiltonian_energy_loss``,
``hamiltonian_positivity_loss``.

A port-Hamiltonian system satisfies::

    ẋ = (J(q) − R(q)) ∇H(q,p) + g(x) u
    y = g(x)ᵀ ∇H(q,p)

with ``J = −Jᵀ`` (skew-symmetric, energy-conserving) and ``R = Lᵀ L ≥ 0``
(dissipative). The storage function ``H`` satisfies
``Ḣ = −∇Hᵀ R ∇H + yᵀ u ≤ yᵀ u`` (passivity inequality), the continuous-time
analog of the RL Bellman inequality.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def make_skew_symmetric(raw: Tensor) -> Tensor:
    r"""Convert a ``[..., n, n]`` raw tensor to a skew-symmetric matrix.

    Uses :math:`J = (raw - raw^\top) / 2`, guaranteeing :math:`J = -J^\top`.
    """
    return (raw - raw.transpose(-1, -2)) / 2.0


def make_positive_definite(L: Tensor, epsilon: float = 1e-6) -> Tensor:
    r"""Build :math:`R = L^\top L + \varepsilon I \succ 0` from ``L``.

    Given ``L`` of shape ``[batch, n, n]`` (lower-triangular network output),
    computes :math:`R = L^\top L + \varepsilon I`. This guarantees positive
    definiteness for any ``L``.
    """
    LtL = torch.bmm(L.transpose(-1, -2), L)
    eye = epsilon * torch.eye(L.shape[-1], device=L.device, dtype=L.dtype)
    return LtL + eye.unsqueeze(0)


def port_hamiltonian_energy_loss(
    H_pred: Tensor,  # predicted Hamiltonian values [batch] (kept for API parity)
    dH_dt: Tensor,  # time derivative of H [batch]
    P_control: Tensor,  # control power yᵀu [batch]
    P_diss: Tensor,  # dissipated power ∇Hᵀ R ∇H ≥ 0 [batch]
) -> Tensor:
    r"""Enforce the port-Hamiltonian dissipation inequality.

    Balance law :math:`dH/dt = P_{control} - P_{dissipation}`, applied as a
    loss :math:`\| dH/dt - P_{control} + P_{diss} \|^2`. ``H_pred`` is part of
    the signature for API parity with the spec but does not enter the residual.
    """
    del H_pred  # not used in the residual; present for API parity (IP §4.4)
    residual = dH_dt - P_control + P_diss
    return (residual ** 2).mean()


def hamiltonian_positivity_loss(H: Tensor) -> Tensor:
    r"""Enforce :math:`H > 0` everywhere (energy must be non-negative).

    Loss :math:`= \mathrm{mean}(\mathrm{ReLU}(-H))`.
    """
    return F.relu(-H).mean()
