r"""Sequence-PITNN -> one-step ``f(x, u)`` adapter for the H-infinity min-max loop.

Targets :func:`pits_mras.training.hinf_minmax.pitnn_one_step` and the thin
wrapper :func:`pits_mras.training.hinf_minmax.hinf_minmax_from_pitnn` (the last
open FEATURE: wiring the full sequence model into the neural adversarial min-max
game via a one-step dynamics adapter).

The crux design decision (documented in ``pitnn_one_step``): a PITNN consumes a
HISTORY WINDOW, but the min-max loop / :func:`linearize_dynamics` need a one-step
``f(x, u) -> xdot`` on a SINGLE state. The adapter holds a FIXED operating-point
context (a history window, default zero history) and varies only the CURRENT
state ``x`` (the canonical ``[q, p]`` block, dim ``2*n_q == output_dim``) and
control ``u``, returning the decoder's instantaneous derivative ``f_hat``.

There is NO oracle-recovery assertion here: a learned (untrained, random-init)
PITNN is not a clean linear system, so recovering the analytic GARE ``P*`` is not
expected and asserting it would be dishonest. The honest verification is
finiteness + shape + differentiability + end-to-end runnability.
"""

import numpy as np
import torch

from pits_mras.config import NetworkConfig, PhysicsConfig
from pits_mras.models.pitnn import PITNN
from pits_mras.training.hinf_minmax import hinf_minmax_from_pitnn, pitnn_one_step
from pits_mras.utils.linearization import linearize_dynamics


# --------------------------------------------------------------------------- #
# Tiny PITNN fixture (mirrors tests/test_models.py).
# --------------------------------------------------------------------------- #
def _make_pitnn() -> tuple[PITNN, int, int, int]:
    """Build a small PITNN; return (model, state_dim=2*n_q, control_dim, input_dim)."""
    torch.manual_seed(0)
    net_cfg = NetworkConfig(
        input_dim=6,
        hidden_dim=16,
        output_dim=4,  # == 2 * n_q
        lstm_layers=1,
        attention_heads=2,
        embedding_dim=8,
    )
    phys_cfg = PhysicsConfig(
        n_generalized_coords=2,  # n_q -> state_dim = output_dim = 4
        hamiltonian_hidden=16,
        dissipation_hidden=8,
    )
    model = PITNN(net_cfg, phys_cfg)
    state_dim = net_cfg.output_dim  # 2 * n_q
    control_dim = phys_cfg.n_generalized_coords  # wired control dim
    return model, state_dim, control_dim, net_cfg.input_dim


# --------------------------------------------------------------------------- #
# pitnn_one_step: returns a callable with the documented one-step contract.
# --------------------------------------------------------------------------- #
def test_pitnn_one_step_returns_callable_with_correct_shape() -> None:
    """f = pitnn_one_step(pitnn); f(x, u) -> xdot of shape [state_dim], finite."""
    pitnn, state_dim, control_dim, _ = _make_pitnn()
    f = pitnn_one_step(pitnn)
    assert callable(f)

    x = torch.randn(state_dim)
    u = torch.randn(control_dim)
    xdot = f(x, u)
    assert xdot.shape == (state_dim,)
    assert torch.isfinite(xdot).all()


def test_pitnn_one_step_differentiable_in_x_and_u() -> None:
    """Grads flow through f w.r.t. both x and u (finite Jacobians).

    The PITNN decoder differentiates a learned Hamiltonian via
    ``torch.autograd.grad`` inside its forward pass, which the ``torch.func``
    (functorch) transforms used by jacrev FORBID. Classic double-backward
    (:func:`torch.autograd.functional.jacobian`) composes with that inner
    autograd, so it is the correct differentiation engine for a PITNN adapter --
    this is the documented ``backend="autograd"`` path of
    :func:`linearize_dynamics`. We verify grads flow through both x and u here.
    """
    pitnn, state_dim, control_dim, _ = _make_pitnn()
    f = pitnn_one_step(pitnn)

    x = torch.randn(state_dim)
    u = torch.randn(control_dim)

    A, B = torch.autograd.functional.jacobian(f, (x, u))
    assert A.shape == (state_dim, state_dim)
    assert B.shape == (state_dim, control_dim)
    assert torch.isfinite(A).all()
    assert torch.isfinite(B).all()


def test_pitnn_one_step_accepts_explicit_history() -> None:
    """A user-supplied [T, dim] history window is accepted and gives finite output."""
    pitnn, state_dim, control_dim, input_dim = _make_pitnn()
    T = 5
    history = torch.randn(T, input_dim)
    f = pitnn_one_step(pitnn, history=history)

    x = torch.randn(state_dim)
    u = torch.randn(control_dim)
    xdot = f(x, u)
    assert xdot.shape == (state_dim,)
    assert torch.isfinite(xdot).all()


# --------------------------------------------------------------------------- #
# Integration: linearize_dynamics on the adapter returns finite (A, B).
# --------------------------------------------------------------------------- #
def test_linearize_dynamics_on_pitnn_one_step() -> None:
    """linearize_dynamics(pitnn_one_step(pitnn), x0, u0, backend='autograd') -> finite (A, B).

    The ``autograd`` backend is required because the PITNN decoder uses
    ``torch.autograd.grad`` internally (jacrev/functorch forbids that).
    """
    pitnn, state_dim, control_dim, _ = _make_pitnn()
    f = pitnn_one_step(pitnn)
    x0 = torch.zeros(state_dim)
    u0 = torch.zeros(control_dim)

    A, B = linearize_dynamics(f, x0, u0, backend="autograd")
    assert A.shape == (state_dim, state_dim)
    assert B.shape == (state_dim, control_dim)
    assert torch.isfinite(A).all()
    assert torch.isfinite(B).all()


# --------------------------------------------------------------------------- #
# End-to-end: hinf_minmax_from_pitnn runs and returns finite metrics + (A, B).
# --------------------------------------------------------------------------- #
def test_hinf_minmax_from_pitnn_runs_end_to_end() -> None:
    """The full sequence-PITNN -> min-max wiring runs and returns finite metrics.

    NO oracle-recovery assertion: an untrained PITNN is not a clean linear
    system, so recovering the analytic GARE P* is not expected; finiteness +
    shapes + end-to-end runnability is the honest verification.
    """
    pitnn, state_dim, control_dim, _ = _make_pitnn()
    x0 = torch.zeros(state_dim)
    u0 = torch.zeros(control_dim)
    Q = np.eye(state_dim)
    R = np.eye(control_dim)
    gamma = 50.0  # large -> near-LQR, keeps the GARE feasible for a random (A, B)

    out = hinf_minmax_from_pitnn(pitnn, x0, u0, Q, R, gamma, n_iters=20, batch_size=32, seed=0)

    # Extracted linear model exposed for transparency.
    assert out["A"].shape == (state_dim, state_dim)
    assert out["B"].shape == (state_dim, control_dim)
    assert np.isfinite(out["A"]).all()
    assert np.isfinite(out["B"]).all()

    # Finite training metrics.
    assert np.isfinite(out["P_hat"]).all()
    assert np.isfinite(out["K_hat"]).all()
    assert all(np.isfinite(out["residual"]))
    assert len(out["residual"]) == 20
