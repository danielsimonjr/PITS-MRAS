"""Models test (IP §11.4): port-Hamiltonian structure.

Targets ``pits_mras.models.decoders`` (and ``models.pitnn``).
Owning phase: Phase 2 per ROADMAP.md (authored alongside its target phase;
§11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.4):
``test_dissipation_matrix_psd``, ``test_J_skew_symmetric``,
``test_hamiltonian_positive`` -- the Phase-2 acceptance gate. Additional
shape/backprop/critic tests are added alongside them per the Phase-2 TDD brief.
"""

import torch

from pits_mras.config import NetworkConfig, PhysicsConfig
from pits_mras.models.attention import PhysicsInformedAttention
from pits_mras.models.critic import CostateHead, QuadraticCritic
from pits_mras.models.decoders import (
    DissipationNet,
    HamiltonianNet,
    PortHamiltonianDecoder,
)
from pits_mras.models.pitnn import PITNN


# --------------------------------------------------------------------------- #
# Acceptance gate (IP §11.4 / §13): three mandated structural tests.
# --------------------------------------------------------------------------- #
def test_dissipation_matrix_psd() -> None:
    """R_theta = L^T L (+ epsilon I) is symmetric positive semidefinite."""
    torch.manual_seed(0)
    n_q = 3
    net = DissipationNet(n_q=n_q, hidden_dim=16)
    q = torch.randn(8, n_q)
    R = net(q)  # [batch, n_q, n_q]
    assert R.shape == (8, n_q, n_q)
    # Symmetry.
    assert torch.allclose(R, R.transpose(-1, -2), atol=1e-5)
    # PSD: all eigenvalues >= 0 (with the epsilon ridge, strictly > 0).
    eigvals = torch.linalg.eigvalsh(R)
    assert (eigvals >= -1e-5).all()


def test_J_skew_symmetric() -> None:
    """The decoder's interconnection matrix J satisfies J = -J^T."""
    torch.manual_seed(0)
    n_q = 2
    # Canonical (constant) J.
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q)
    J = dec.get_J(torch.randn(5, n_q))  # [batch, 2n_q, 2n_q]
    assert J.shape == (5, 2 * n_q, 2 * n_q)
    assert torch.allclose(J, -J.transpose(-1, -2), atol=1e-6)
    # Position-dependent (learned) J must also be skew-symmetric.
    dec_pd = PortHamiltonianDecoder(
        n_q=n_q, context_dim=8, output_dim=2 * n_q, use_position_dependent_J=True
    )
    J_pd = dec_pd.get_J(torch.randn(5, n_q))
    assert torch.allclose(J_pd, -J_pd.transpose(-1, -2), atol=1e-6)


def test_hamiltonian_positive() -> None:
    """The learned Hamiltonian H_theta(q, p) is strictly positive."""
    torch.manual_seed(0)
    n_q = 3
    net = HamiltonianNet(n_q=n_q, hidden_dim=16)
    q = torch.randn(32, n_q)
    p = torch.randn(32, n_q)
    H = net(q, p)
    assert H.shape == (32, 1)
    assert (H > 0).all()


# --------------------------------------------------------------------------- #
# Attention module.
# --------------------------------------------------------------------------- #
def test_attention_shapes_and_alpha_sums_to_one() -> None:
    """context is [batch, d_k]; alpha is [batch, T] and sums to 1 per row."""
    torch.manual_seed(0)
    batch, T, d_k, e_dim, n_state, c_dim = 4, 7, 16, 3, 5, 2
    attn = PhysicsInformedAttention(
        d_k=d_k, e_dim=e_dim, n_state=n_state, control_dim=c_dim
    )
    H_enc = torch.randn(batch, T, d_k)
    e_hist = torch.randn(batch, T, e_dim)
    x_p = torch.randn(batch, n_state)
    e_curr = torch.randn(batch, e_dim)
    x_p_dot = torch.randn(batch, n_state)
    u_curr = torch.randn(batch, c_dim)
    context, alpha = attn(H_enc, e_hist, x_p, e_curr, x_p_dot, u_curr)
    assert context.shape == (batch, d_k)
    assert alpha.shape == (batch, T)
    # Convex combination of three softmax distributions -> rows sum to 1.
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(batch), atol=1e-5)
    assert (alpha >= 0).all()


def test_attention_regularization_loss_is_scalar() -> None:
    torch.manual_seed(0)
    attn = PhysicsInformedAttention(d_k=8, e_dim=2, n_state=3)
    alpha = torch.softmax(torch.randn(4, 6), dim=-1)
    loss = attn.attention_regularization_loss(alpha)
    assert loss.shape == ()


# --------------------------------------------------------------------------- #
# Decoder: shapes + out-of-place backprop (the in-place autograd fix).
# --------------------------------------------------------------------------- #
def test_decoder_forward_shapes() -> None:
    torch.manual_seed(0)
    n_q, c_dim = 2, 2
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q)
    batch = 6
    q = torch.randn(batch, n_q)
    p = torch.randn(batch, n_q)
    q_dot = torch.randn(batch, n_q)
    u = torch.randn(batch, c_dim)
    c_t = torch.randn(batch, 8)
    f_hat, H_val, P_diss, energy_loss = dec(q, p, q_dot, u, c_t)
    assert f_hat.shape == (batch, 2 * n_q)
    assert H_val.shape == (batch, 1)
    assert P_diss.shape == (batch,)
    assert energy_loss.shape == ()


def test_decoder_backward_runs() -> None:
    """loss.backward() must run without an in-place-modification autograd error."""
    torch.manual_seed(0)
    n_q, c_dim = 2, 2
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q)
    batch = 6
    q = torch.randn(batch, n_q)
    p = torch.randn(batch, n_q)
    q_dot = torch.randn(batch, n_q)
    u = torch.randn(batch, c_dim)
    c_t = torch.randn(batch, 8)
    f_hat, _, _, energy_loss = dec(q, p, q_dot, u, c_t)
    loss = f_hat.pow(2).mean() + energy_loss
    loss.backward()  # would raise if a grad-requiring tensor were modified in place
    grads = [pm.grad for pm in dec.parameters() if pm.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------- #
# Critic.
# --------------------------------------------------------------------------- #
def test_critic_value_nonnegative_and_quadratic() -> None:
    """V(e) >= 0 near init (P ~ I) and equals e^T P e with extract_P."""
    torch.manual_seed(0)
    n = 4
    critic = QuadraticCritic(state_dim=n)
    e = torch.randn(10, n)
    V = critic(e)
    assert V.shape == (10,)
    assert (V >= 0).all()
    # Quadratic identity: V(e) == e^T P e.
    P = critic.extract_P()
    quad = torch.einsum("bi,ij,bj->b", e, P, e)
    assert torch.allclose(V, quad, atol=1e-5)
    # P symmetric.
    assert torch.allclose(P, P.T, atol=1e-6)


def test_critic_costate_equals_two_P_e_and_autograd() -> None:
    """costate lambda = grad V = 2 P e (Identity 2 by construction)."""
    torch.manual_seed(0)
    n = 3
    critic = QuadraticCritic(state_dim=n)
    e = torch.randn(7, n)
    lam = critic.gradient(e)
    assert lam.shape == (7, n)
    P = critic.extract_P()
    expected = 2.0 * (e @ P.T)
    assert torch.allclose(lam, expected, atol=1e-5)


def test_critic_positivity_loss_zero_at_init() -> None:
    critic = QuadraticCritic(state_dim=4)
    loss = critic.positivity_loss()
    assert loss.shape == ()
    assert loss.item() <= 1e-5  # P ~ I at init -> positive definite -> ~0


def test_costate_head_optimal_control_shape() -> None:
    torch.manual_seed(0)
    n, m = 4, 2
    critic = QuadraticCritic(state_dim=n)
    R_inv = torch.eye(m)
    B = torch.randn(n, m)
    head = CostateHead(critic, R_inv, B)
    e = torch.randn(5, n)
    lam, u = head(e)
    assert lam.shape == (5, n)
    assert u.shape == (5, m)
    # u* = -R^-1 B^T lambda.
    expected = -(lam @ B) @ R_inv.T
    assert torch.allclose(u, expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# PITNN end-to-end forward.
# --------------------------------------------------------------------------- #
def test_pitnn_forward_returns_dict_with_shapes() -> None:
    torch.manual_seed(0)
    net_cfg = NetworkConfig(
        input_dim=6, hidden_dim=16, output_dim=4, lstm_layers=1,
        attention_heads=2, embedding_dim=8,
    )
    phys_cfg = PhysicsConfig(
        n_generalized_coords=2, hamiltonian_hidden=16, dissipation_hidden=8,
    )
    model = PITNN(net_cfg, phys_cfg)
    batch, T = 3, 5
    x_hist = torch.randn(batch, T, net_cfg.input_dim)
    u_hist = torch.randn(batch, T, net_cfg.input_dim)
    x_p_curr = torch.randn(batch, net_cfg.input_dim)
    u_curr = torch.randn(batch, 2)
    e_curr = torch.randn(batch, net_cfg.output_dim)
    e_hist = torch.randn(batch, T, net_cfg.output_dim)
    out = model(x_hist, u_hist, x_p_curr, u_curr, e_curr, e_hist)
    for key in ("f", "H", "context", "alpha", "h_enc"):
        assert key in out, f"missing key {key}"
    assert out["f"].shape == (batch, net_cfg.output_dim)
    assert out["H"].shape == (batch, 1)
    assert out["context"].shape == (batch, net_cfg.hidden_dim)
    assert out["alpha"].shape == (batch, T)
    assert out["h_enc"].shape == (batch, T, net_cfg.hidden_dim)
