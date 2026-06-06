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
    attn = PhysicsInformedAttention(d_k=d_k, e_dim=e_dim, n_state=n_state, control_dim=c_dim)
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
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q, control_dim=c_dim)
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


def test_port_hamiltonian_energy_residual_vanishes() -> None:
    """The pH dissipation balance dH/dt = P_control - P_diss holds by construction.

    Regression for the dissipation-channel fix (audit #4). With no control
    (u=0) and no temporal correction (W_corr=0), the only contributions to
    dH/dt are conservative (grad_H . J grad_H = 0 by skew-symmetry) and
    dissipative. Because dissipation now acts on the momentum (p) block via the
    velocity dH/dp and P_diss = (dH/dp)^T R (dH/dp), the dissipative part of
    dH/dt is exactly -P_diss, so the energy residual ||dH/dt - P_control +
    P_diss||^2 vanishes. Previously f_diss used the finite-difference q_dot
    while P_diss used grad_H_q, so the residual could not vanish.
    """
    torch.manual_seed(0)
    n_q = 2
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q, control_dim=n_q)
    with torch.no_grad():
        dec.W_corr.weight.zero_()
        dec.W_corr.bias.zero_()
    batch = 16
    q = torch.randn(batch, n_q)
    p = torch.randn(batch, n_q)
    q_dot = torch.randn(batch, n_q)  # irrelevant to the pH-consistent dissipation
    u = torch.zeros(batch, n_q)
    c_t = torch.randn(batch, 8)
    _, _, _, energy_loss = dec(q, p, q_dot, u, c_t)
    assert energy_loss.item() < 1e-6


def test_decoder_backward_runs() -> None:
    """loss.backward() must run without an in-place-modification autograd error."""
    torch.manual_seed(0)
    n_q, c_dim = 2, 2
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q, control_dim=c_dim)
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
# G8: generalized MIMO control input  f_ctrl = B(x_p) @ u.
# --------------------------------------------------------------------------- #
def test_decoder_control_dim1_matches_old_scalar_formula() -> None:
    """Characterization: at control_dim=1 the generalized B(x_p) @ u reduces
    EXACTLY to the pre-G8 single-input form B_val * u.sum(dim=-1, keepdim=True).

    At control_dim=1 the B_net head width is 2*n_q (== old layout), so for a
    fixed seed B_net produces the identical [batch, 2*n_q] output. The old code
    multiplied that by u.sum() (a no-op for a 1-column u). We recompute that
    reference inline from B_net and require bit-tight equality with the f_ctrl
    embedded in f_hat (isolated by zeroing every other dynamics channel).
    """
    torch.manual_seed(0)
    n_q = 3
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q, control_dim=1)
    # Head width must be unchanged from the old single-input layout.
    assert dec.B_net[-1].out_features == 2 * n_q
    batch = 7
    q = torch.randn(batch, n_q)
    p = torch.randn(batch, n_q)
    q_dot = torch.randn(batch, n_q)
    u = torch.randn(batch, 1)
    c_t = torch.randn(batch, 8)

    # Reference: OLD formula f_ctrl = B_val * u.sum(dim=-1, keepdim=True).
    with torch.no_grad():
        B_val = dec.B_net(torch.cat([q, p], dim=-1))  # [batch, 2*n_q]
        f_ctrl_old = B_val * u.sum(dim=-1, keepdim=True)  # [batch, 2*n_q]

    # Isolate f_ctrl inside the model: zero the H, dissipation, and correction
    # channels so f_hat == f_ctrl. With H == 0, grad_H == 0 -> f_cons, f_diss = 0.
    with torch.no_grad():
        for layer in dec.H_net.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        dec.W_corr.weight.zero_()
        dec.W_corr.bias.zero_()
    f_hat, _, _, _ = dec(q, p, q_dot, u, c_t)
    assert torch.allclose(f_hat, f_ctrl_old, atol=1e-6)


def test_decoder_mimo_control_equals_bmm() -> None:
    """MIMO: f_ctrl == bmm(B_mat, u) exactly, read from the model's own B_net."""
    torch.manual_seed(1)
    n_q, c_dim = 2, 3
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q, control_dim=c_dim)
    assert dec.B_net[-1].out_features == 2 * n_q * c_dim
    batch = 5
    q = torch.randn(batch, n_q)
    p = torch.randn(batch, n_q)
    q_dot = torch.randn(batch, n_q)
    u = torch.randn(batch, c_dim)
    c_t = torch.randn(batch, 8)

    # Zero everything but the control channel so f_hat == f_ctrl.
    with torch.no_grad():
        for layer in dec.H_net.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        dec.W_corr.weight.zero_()
        dec.W_corr.bias.zero_()
        B_mat = dec.B_net(torch.cat([q, p], dim=-1)).view(batch, 2 * n_q, c_dim)
        f_ctrl_ref = torch.bmm(B_mat, u.unsqueeze(-1)).squeeze(-1)
    f_hat, _, _, _ = dec(q, p, q_dot, u, c_t)
    assert torch.allclose(f_hat, f_ctrl_ref, atol=1e-6)


def test_decoder_mimo_columns_act_independently() -> None:
    """The columns of B act independently: u=[1,0] vs u=[0,1] give DIFFERENT,
    column-specific f_ctrl (not the scalar-sum collapse, which would be equal).
    """
    torch.manual_seed(2)
    n_q, c_dim = 2, 2
    dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q, control_dim=c_dim)
    batch = 4
    q = torch.randn(batch, n_q)
    p = torch.randn(batch, n_q)
    q_dot = torch.randn(batch, n_q)
    c_t = torch.randn(batch, 8)
    with torch.no_grad():
        for layer in dec.H_net.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        dec.W_corr.weight.zero_()
        dec.W_corr.bias.zero_()
        B_mat = dec.B_net(torch.cat([q, p], dim=-1)).view(batch, 2 * n_q, c_dim)

    u_e0 = torch.tensor([[1.0, 0.0]]).expand(batch, c_dim)
    u_e1 = torch.tensor([[0.0, 1.0]]).expand(batch, c_dim)
    f0, _, _, _ = dec(q, p, q_dot, u_e0, c_t)
    f1, _, _, _ = dec(q, p, q_dot, u_e1, c_t)
    # Each selects the corresponding column of B.
    assert torch.allclose(f0, B_mat[:, :, 0], atol=1e-6)
    assert torch.allclose(f1, B_mat[:, :, 1], atol=1e-6)
    # The columns are genuinely distinct -> the scalar-sum collapse (which would
    # make f0 == f1 for any unit input) is gone.
    assert not torch.allclose(f0, f1, atol=1e-4)


def test_decoder_control_dim_shapes() -> None:
    """f_ctrl/f_hat shape is [batch, 2*n_q] for control_dim in {1, 2, 3}."""
    n_q = 2
    batch = 6
    for c_dim in (1, 2, 3):
        torch.manual_seed(c_dim)
        dec = PortHamiltonianDecoder(n_q=n_q, context_dim=8, output_dim=2 * n_q, control_dim=c_dim)
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


def test_critic_positivity_loss_is_differentiable_and_trainable() -> None:
    """positivity_loss must contribute a REAL gradient (regression for the
    detached no-op): a seeded indefinite P gives a positive, differentiable loss
    whose gradient descent restores positive-definiteness."""
    torch.manual_seed(0)
    critic = QuadraticCritic(state_dim=2)
    critic.set_P(torch.tensor([[1.0, 0.0], [0.0, -2.0]]))  # indefinite (min eig -2)
    lp = critic.positivity_loss()
    assert lp.requires_grad and lp.grad_fn is not None
    assert lp.item() > 0.0
    # A real, non-zero gradient flows to W_c while P is indefinite (the no-op bug
    # produced None / zero here).
    lp.backward()
    assert critic.W_c.weight.grad is not None
    assert critic.W_c.weight.grad.abs().sum() > 0.0
    # Gradient descent on the term restores positive-definiteness (loss -> ~0,
    # so the gradient legitimately vanishes once P is PD).
    opt = torch.optim.Adam(critic.parameters(), lr=0.1)
    first = critic.positivity_loss().item()
    for _ in range(150):
        opt.zero_grad()
        loss = critic.positivity_loss()
        loss.backward()
        opt.step()
    assert critic.positivity_loss().item() < first
    assert critic.positivity_loss().item() < 1e-4  # P is positive-definite now


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
    # u* = -½ R^-1 B^T lambda (half_grad=True default; V=eᵀPe -> u*=-Ke).
    expected = -0.5 * (lam @ B) @ R_inv.T
    assert torch.allclose(u, expected, atol=1e-5)
    # half_grad=False recovers the literal un-halved form.
    head_full = CostateHead(critic, R_inv, B, half_grad=False)
    _, u_full = head_full(e)
    assert torch.allclose(u_full, -(lam @ B) @ R_inv.T, atol=1e-5)


# --------------------------------------------------------------------------- #
# PITNN end-to-end forward.
# --------------------------------------------------------------------------- #
def test_pitnn_forward_returns_dict_with_shapes() -> None:
    torch.manual_seed(0)
    net_cfg = NetworkConfig(
        input_dim=6,
        hidden_dim=16,
        output_dim=4,
        lstm_layers=1,
        attention_heads=2,
        embedding_dim=8,
    )
    phys_cfg = PhysicsConfig(
        n_generalized_coords=2,
        hamiltonian_hidden=16,
        dissipation_hidden=8,
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
    for key in ("f_hat", "H_val", "context", "alpha", "h_enc"):
        assert key in out, f"missing key {key}"
    assert out["f_hat"].shape == (batch, net_cfg.output_dim)
    assert out["H_val"].shape == (batch, 1)
    assert out["context"].shape == (batch, net_cfg.hidden_dim)
    assert out["alpha"].shape == (batch, T)
    assert out["h_enc"].shape == (batch, T, net_cfg.hidden_dim)
    # The redundant brief-key aliases (== f_hat / H_val) were removed.
    assert "f" not in out and "H" not in out
