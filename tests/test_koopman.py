"""Deep Koopman lifting model tests (ROADMAP proposal #2).

Targets ``pits_mras.models.koopman``. ADDITIVE capability: a learnable encoder
lifting the nonlinear state into a latent space with *exactly linear* dynamics
``(A_z, B_z)``, plus the three Lusch-et-al.-style training losses
(reconstruction, latent linearity, state prediction). NOT wired into the
control loop -- this file exercises the model + losses in isolation.

Design choices that drive the deterministic asserts:
- ``include_state=True`` makes the lift ``z = concat([x, psi(x)])`` and the
  decoder the exact slice ``z[:, :state_dim]``, so reconstruction is exact by
  construction (loss ~ 0).
- ``latent_step`` is literally ``z @ A_z^T + u @ B_z^T`` (no bias), so it is
  exactly linear and testable against the raw parameters from
  ``latent_matrices()``.
"""

import numpy as np
import pytest
import torch

from pits_mras.models.koopman import KoopmanLiftingModel, koopman_loss


# --------------------------------------------------------------------------- #
# Construction / validation
# --------------------------------------------------------------------------- #
def test_dim_validation_positive() -> None:
    """Non-positive dims must raise."""
    with pytest.raises(ValueError):
        KoopmanLiftingModel(state_dim=0, control_dim=1, latent_dim=4)
    with pytest.raises(ValueError):
        KoopmanLiftingModel(state_dim=2, control_dim=0, latent_dim=4)
    with pytest.raises(ValueError):
        KoopmanLiftingModel(state_dim=2, control_dim=1, latent_dim=0)


def test_latent_dim_must_cover_state_when_include_state() -> None:
    """With include_state=True the lift contains the state, so latent_dim >= state_dim."""
    with pytest.raises(ValueError):
        KoopmanLiftingModel(state_dim=4, control_dim=1, latent_dim=3, include_state=True)
    # Equality is allowed (zero extra coords).
    KoopmanLiftingModel(state_dim=4, control_dim=1, latent_dim=4, include_state=True)
    # include_state=False has no such constraint.
    KoopmanLiftingModel(state_dim=4, control_dim=1, latent_dim=2, include_state=False)


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("include_state", [True, False])
def test_shapes(include_state: bool) -> None:
    """encode/latent_step/decode/forward produce the documented [batch, dim] shapes."""
    b, sd, cd, ld = 5, 3, 2, 6
    model = KoopmanLiftingModel(sd, cd, ld, include_state=include_state)
    x = torch.randn(b, sd)
    u = torch.randn(b, cd)
    z = model.encode(x)
    assert z.shape == (b, ld)
    z_next = model.latent_step(z, u)
    assert z_next.shape == (b, ld)
    x_rec = model.decode(z)
    assert x_rec.shape == (b, sd)
    x_next = model.forward(x, u)
    assert x_next.shape == (b, sd)


# --------------------------------------------------------------------------- #
# latent_step is exactly linear
# --------------------------------------------------------------------------- #
def test_latent_step_matches_matrices() -> None:
    """latent_step(z, u) == z @ A_z^T + u @ B_z^T using params from latent_matrices()."""
    model = KoopmanLiftingModel(state_dim=3, control_dim=2, latent_dim=6)
    A_z, B_z = model.latent_matrices()
    z = torch.randn(7, 6)
    u = torch.randn(7, 2)
    expected = z @ A_z.T + u @ B_z.T
    got = model.latent_step(z, u)
    assert torch.allclose(got, expected, atol=1e-6)


def test_latent_step_linearity() -> None:
    """latent_step is linear: f(a*z1+b*z2, a*u1+b*u2) = a*f(z1,u1)+b*f(z2,u2)."""
    model = KoopmanLiftingModel(state_dim=2, control_dim=1, latent_dim=5)
    z1, z2 = torch.randn(4, 5), torch.randn(4, 5)
    u1, u2 = torch.randn(4, 1), torch.randn(4, 1)
    a, b = 1.7, -0.4
    lhs = model.latent_step(a * z1 + b * z2, a * u1 + b * u2)
    rhs = a * model.latent_step(z1, u1) + b * model.latent_step(z2, u2)
    assert torch.allclose(lhs, rhs, atol=1e-5)


# --------------------------------------------------------------------------- #
# include_state=True: exact state pass-through
# --------------------------------------------------------------------------- #
def test_include_state_exact_reconstruction() -> None:
    """decode(encode(x)) recovers x exactly when include_state=True."""
    model = KoopmanLiftingModel(state_dim=4, control_dim=2, latent_dim=8, include_state=True)
    x = torch.randn(6, 4)
    x_rec = model.decode(model.encode(x))
    assert torch.allclose(x_rec, x, atol=1e-6)


def test_include_state_recon_loss_zero() -> None:
    """Reconstruction loss term is ~0 by construction with include_state=True."""
    model = KoopmanLiftingModel(state_dim=3, control_dim=1, latent_dim=5, include_state=True)
    x = torch.randn(8, 3)
    u = torch.randn(8, 1)
    x_next = torch.randn(8, 3)
    terms = koopman_loss(model, x, u, x_next)
    assert terms["recon"].item() < 1e-10


def test_include_state_lift_contains_state() -> None:
    """The first state_dim coords of the lift equal the state itself."""
    model = KoopmanLiftingModel(state_dim=3, control_dim=1, latent_dim=7, include_state=True)
    x = torch.randn(5, 3)
    z = model.encode(x)
    assert torch.allclose(z[:, :3], x, atol=1e-6)


# --------------------------------------------------------------------------- #
# koopman_loss correctness
# --------------------------------------------------------------------------- #
def test_loss_keys_and_weighting() -> None:
    """Loss dict exposes the three terms and a weighted total."""
    model = KoopmanLiftingModel(state_dim=2, control_dim=1, latent_dim=4)
    x, u, x_next = torch.randn(6, 2), torch.randn(6, 1), torch.randn(6, 2)
    terms = koopman_loss(model, x, u, x_next, w_recon=2.0, w_pred=3.0, w_lin=0.5)
    for k in ("recon", "lin", "pred", "loss"):
        assert k in terms
    expected_total = 2.0 * terms["recon"] + 0.5 * terms["lin"] + 3.0 * terms["pred"]
    assert torch.allclose(terms["loss"], expected_total, atol=1e-6)


def test_loss_zero_on_self_consistent_data() -> None:
    """All three terms ~0 when x_next is the model's own (exact-linear) rollout.

    With include_state=True the lift contains the state, so feeding back the
    decoded one-step prediction as the target makes every term vanish: recon by
    construction, pred = ||x_next - forward(x,u)||^2 = 0, and lin holds because
    encode(decode(z_next)) reproduces the latent for the state coords and the
    extra coords are a deterministic function psi of those state coords.
    """
    torch.manual_seed(0)
    model = KoopmanLiftingModel(state_dim=2, control_dim=1, latent_dim=4, include_state=True)
    x = torch.randn(10, 2)
    u = torch.randn(10, 1)
    # Self-consistent next state: exactly what the model predicts.
    with torch.no_grad():
        x_next = model.forward(x, u)
    terms = koopman_loss(model, x, u, x_next)
    assert terms["recon"].item() < 1e-10
    assert terms["pred"].item() < 1e-10
    assert terms["lin"].item() < 1e-8
    assert terms["loss"].item() < 1e-8


def test_loss_positive_on_inconsistent_data() -> None:
    """Loss is strictly positive for data the model does not predict."""
    torch.manual_seed(1)
    model = KoopmanLiftingModel(state_dim=2, control_dim=1, latent_dim=4, include_state=True)
    x = torch.randn(10, 2)
    u = torch.randn(10, 1)
    x_next = model.forward(x, u).detach() + 5.0  # deliberately wrong target
    terms = koopman_loss(model, x, u, x_next)
    assert terms["loss"].item() > 1e-3


def test_gradients_flow() -> None:
    """loss.backward() yields finite, non-None grads on encoder, A_z, B_z, decoder."""
    model = KoopmanLiftingModel(state_dim=3, control_dim=2, latent_dim=6, include_state=False)
    x, u, x_next = torch.randn(8, 3), torch.randn(8, 2), torch.randn(8, 3)
    terms = koopman_loss(model, x, u, x_next)
    terms["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    for g in grads:
        assert g is not None
        assert torch.isfinite(g).all()


# --------------------------------------------------------------------------- #
# Bridge sanity to the linear core (shape/dtype smoke only)
# --------------------------------------------------------------------------- #
def test_latent_matrices_shapes_and_numpy_bridge() -> None:
    """A_z [ld, ld], B_z [ld, cd]; convertible to numpy for solve_care shape-wise."""
    ld, cd = 5, 2
    model = KoopmanLiftingModel(state_dim=2, control_dim=cd, latent_dim=ld)
    A_z, B_z = model.latent_matrices()
    assert A_z.shape == (ld, ld)
    assert B_z.shape == (ld, cd)
    A_np = A_z.detach().cpu().numpy()
    B_np = B_z.detach().cpu().numpy()
    # Shapes solve_care would accept: A [n,n], B [n,m], Q [n,n], R [m,m].
    Q = np.eye(ld, dtype=A_np.dtype)
    R = np.eye(cd, dtype=B_np.dtype)
    assert A_np.shape == (ld, ld)
    assert B_np.shape == (ld, cd)
    assert Q.shape == (ld, ld)
    assert R.shape == (cd, cd)


# --------------------------------------------------------------------------- #
# Optional loose training sanity
# --------------------------------------------------------------------------- #
def test_adam_reduces_loss_on_linear_system() -> None:
    """A few Adam steps reduce koopman_loss on a small linear-in-state system."""
    torch.manual_seed(2)
    sd, cd, ld = 2, 1, 4
    # Ground-truth linear system in the state itself.
    A_true = torch.tensor([[0.9, 0.1], [-0.05, 0.95]])
    B_true = torch.tensor([[0.2], [0.1]])
    x = torch.randn(64, sd)
    u = torch.randn(64, cd)
    x_next = x @ A_true.T + u @ B_true.T

    model = KoopmanLiftingModel(sd, cd, ld, include_state=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss0 = koopman_loss(model, x, u, x_next)["loss"].item()
    for _ in range(50):
        opt.zero_grad()
        loss = koopman_loss(model, x, u, x_next)["loss"]
        loss.backward()
        opt.step()
    loss1 = koopman_loss(model, x, u, x_next)["loss"].item()
    assert loss1 < loss0
