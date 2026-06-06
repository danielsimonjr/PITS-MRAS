r"""Tests for the GENERIC/GFINN thermodynamic decoder (``models.generic``).

The structural guarantees ARE the spec. A faithful GENERIC construction must
satisfy, *by construction* (to floating-point tolerance, not via penalty):

- ``L`` skew-symmetric:  :math:`L^\top = -L`.
- ``M`` symmetric PSD:   :math:`M = M^\top \succeq 0`.
- degeneracy:            :math:`L(z)\,\nabla S(z) = 0` and
                         :math:`M(z)\,\nabla E(z) = 0`.
- first law (energy conservation):  :math:`dE/dt = \nabla E^\top \dot z = 0`.
- second law (entropy production):   :math:`dS/dt = \nabla S^\top \dot z \ge 0`.

Tests use float64 for tight degeneracy/conservation tolerances.
"""

import torch

from pits_mras.models import GFINNDecoder

_TOL = 1e-5
_TIGHT = 1e-6


def _make(dim: int = 4, seed: int = 0) -> GFINNDecoder:
    torch.manual_seed(seed)
    model = GFINNDecoder(dim=dim, n_skew=3, n_friction=3).double()
    return model


def _z(batch: int = 5, dim: int = 4, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, dim, dtype=torch.float64, requires_grad=True)


def test_potentials_and_field_shapes() -> None:
    dim, batch = 4, 6
    model = _make(dim)
    z = _z(batch, dim)
    assert model.energy(z).shape == (batch, 1)
    assert model.entropy(z).shape == (batch, 1)
    assert model.grad_E(z).shape == (batch, dim)
    assert model.grad_S(z).shape == (batch, dim)
    assert model.L(z).shape == (batch, dim, dim)
    assert model.M(z).shape == (batch, dim, dim)
    assert model.forward(z).shape == (batch, dim)


def test_L_skew_symmetric() -> None:
    model = _make()
    z = _z()
    L = model.L(z)
    resid = (L + L.transpose(-1, -2)).abs().max().item()
    assert resid < _TIGHT, f"L not skew-symmetric: {resid}"


def test_M_symmetric_psd() -> None:
    model = _make()
    z = _z()
    M = model.M(z)
    sym_resid = (M - M.transpose(-1, -2)).abs().max().item()
    assert sym_resid < _TIGHT, f"M not symmetric: {sym_resid}"
    eig_min = torch.linalg.eigvalsh(M).min().item()
    assert eig_min >= -_TIGHT, f"M not PSD: min eig {eig_min}"


def test_degeneracy_L_grad_S() -> None:
    """``L(z) ∇S(z) = 0`` by construction."""
    model = _make()
    z = _z()
    L = model.L(z)
    gS = model.grad_S(z)
    resid = torch.bmm(L, gS.unsqueeze(-1)).squeeze(-1).norm(dim=-1).max().item()
    assert resid < _TOL, f"L ∇S != 0: {resid}"


def test_degeneracy_M_grad_E() -> None:
    """``M(z) ∇E(z) = 0`` by construction."""
    model = _make()
    z = _z()
    M = model.M(z)
    gE = model.grad_E(z)
    resid = torch.bmm(M, gE.unsqueeze(-1)).squeeze(-1).norm(dim=-1).max().item()
    assert resid < _TOL, f"M ∇E != 0: {resid}"


def test_first_law_energy_conserved() -> None:
    """``dE/dt = ∇Eᵀ ż ≈ 0`` for random z (energy conservation)."""
    model = _make()
    z = _z()
    gE = model.grad_E(z)
    dz = model.forward(z)
    dE_dt = (gE * dz).sum(dim=-1)
    assert dE_dt.abs().max().item() < _TOL, f"dE/dt != 0: {dE_dt.abs().max().item()}"


def test_second_law_entropy_nondecreasing() -> None:
    """``dS/dt = ∇Sᵀ ż ≥ 0`` for random z (entropy production)."""
    model = _make()
    z = _z()
    gS = model.grad_S(z)
    dz = model.forward(z)
    dS_dt = (gS * dz).sum(dim=-1)
    assert dS_dt.min().item() >= -1e-7, f"dS/dt < 0: {dS_dt.min().item()}"


def test_forward_differentiable_grads_flow() -> None:
    """forward() is differentiable and grads flow to E, S, L, M params."""
    torch.manual_seed(0)
    model = GFINNDecoder(dim=4, n_skew=2, n_friction=2)  # float32 default
    z = torch.randn(5, 4, requires_grad=True)
    loss = model.forward(z).pow(2).sum()
    loss.backward()
    param_groups = {
        "energy": model.energy_net,
        "entropy": model.entropy_net,
        "skew": model.skew_net,
        "friction": model.friction_net,
    }
    for name, sub in param_groups.items():
        grads = [p.grad for p in sub.parameters() if p.grad is not None]
        assert grads, f"no grad reached {name} params"
        assert any(g.abs().sum().item() > 0 for g in grads), f"zero grad in {name}"


def test_second_law_strictly_positive_generic_case() -> None:
    """With random init the entropy production is generically strictly positive."""
    model = _make(seed=3)
    z = _z(batch=8, seed=4)
    gS = model.grad_S(z)
    dz = model.forward(z)
    dS_dt = (gS * dz).sum(dim=-1)
    # At least one sample should show genuine production (not all clamped to ~0).
    assert dS_dt.max().item() > 1e-6
