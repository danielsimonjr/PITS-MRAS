"""Safety test (IP §11.3): CLF-CBF-QP filter / forward invariance (Identity 3).

Targets ``pits_mras.controllers.safety`` (``CLFCBFSafetyFilter``).
Owning phase: Phase 4 per ROADMAP.md (authored alongside its target phase;
§11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.3):
``test_cbf_projects_unsafe_control``, ``test_cbf_identity_when_safe``,
``test_cbf_forward_invariance`` (100-step sim stays in the safe set).
"""

import numpy as np
import torch

from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.controllers.safety import CLFCBFSafetyFilter


def _make_filter(safety_margin: float = 5.0, decay_rate: float = 1.0):
    """Build a filter from a small Hurwitz reference model (P from Lyapunov)."""
    A_m = np.array([[-1.0, 0.5], [0.0, -2.0]])
    B_m = np.array([[1.0, 0.0], [0.0, 1.0]])
    C_m = np.eye(2)
    rm = LinearReferenceModel(A_m, B_m, C_m, np.eye(2), np.eye(2))
    filt = CLFCBFSafetyFilter(
        P=rm.P,
        A_m=rm.A_m,
        B_ctrl=rm.B_m,
        safety_margin=safety_margin,
        decay_rate=decay_rate,
    )
    return filt, rm


def _safety_index(filt, e, u):
    """a = L_f h + L_g h . u + gamma h(e), per §3.4. >= 0 means safe."""
    P = filt.P
    A_m = filt.A_m
    B = filt.B_ctrl
    ePe = (e @ P * e).sum(dim=-1)
    h_e = filt.safety_margin - ePe
    Pe = e @ P
    L_f_h = -2.0 * (Pe * (e @ A_m.T)).sum(dim=-1)
    L_g_h = -2.0 * (Pe @ B)
    L_g_h_u = (L_g_h * u).sum(dim=-1)
    return L_f_h + L_g_h_u + filt.decay_rate * h_e


def test_cbf_projects_unsafe_control() -> None:
    """An unsafe nominal control is projected onto the safe half-space."""
    filt, _ = _make_filter(safety_margin=5.0, decay_rate=1.0)
    # Pick e near the ellipsoid boundary and a u_nom that pushes outward.
    e = torch.tensor([[1.8, 1.4]])
    # Drive e outward: choose u_nom aligned to increase e^T P e.
    u_nom = torch.tensor([[20.0, 20.0]])

    a_before = _safety_index(filt, e, u_nom)
    assert a_before.item() < 0, "test setup: u_nom must violate the constraint"

    u_safe, h_e, slack = filt(e, u_nom)
    # The control was actually modified.
    assert not torch.allclose(u_safe, u_nom)
    assert slack.item() > 0

    # After projection the constraint is satisfied (a >= 0 up to tol).
    a_after = _safety_index(filt, e, u_safe)
    assert a_after.item() >= -1e-4


def test_cbf_identity_when_safe() -> None:
    """A safe nominal control passes through the filter unchanged."""
    filt, _ = _make_filter(safety_margin=10.0, decay_rate=1.0)
    e = torch.tensor([[0.1, 0.1]])  # deep interior, h(e) large
    u_nom = torch.tensor([[0.05, -0.05]])  # mild control

    a_before = _safety_index(filt, e, u_nom)
    assert a_before.item() >= 0, "test setup: u_nom must already be safe"

    u_safe, h_e, slack = filt(e, u_nom)
    assert torch.allclose(u_safe, u_nom, atol=1e-7)
    assert slack.item() < 1e-7


def test_cbf_forward_invariance() -> None:
    """A 100-step closed-loop sim stays inside the safe error ellipsoid."""
    margin = 5.0
    filt, rm = _make_filter(safety_margin=margin, decay_rate=1.0)
    P = filt.P
    A_m = filt.A_m
    B = filt.B_ctrl

    # Start from an interior point: e^T P e < c.
    e = torch.tensor([[1.5, 1.2]])
    ePe0 = (e @ P * e).sum(dim=-1).item()
    assert ePe0 < margin, "test setup: must start in the interior"

    dt = 0.01
    # A nominal control that, unfiltered, would drive e out of the set.
    rng = torch.Generator().manual_seed(0)
    max_ePe = ePe0
    for _ in range(100):
        # Aggressive destabilizing nominal control (random + outward push).
        u_nom = 30.0 * torch.randn(1, 2, generator=rng)
        u_safe, _, _ = filt(e, u_nom)
        # Error dynamics: e_dot = A_m e + B u_safe (Euler).
        e = e + (e @ A_m.T + u_safe @ B.T) * dt
        ePe = (e @ P * e).sum(dim=-1).item()
        max_ePe = max(max_ePe, ePe)
        # Forward invariance: never leave the safe set (small numeric tol).
        assert ePe <= margin + 1e-2, f"left safe set: e^T P e = {ePe} > {margin}"
