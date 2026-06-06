r"""Port-Hamiltonian decoders (IP §5.2).

Owning phase: Phase 2 (Neural Network Models).

Implements Connection 2 (port-Hamiltonian storage = value). Three modules:

- ``HamiltonianNet``: MLP with a ``Softplus`` head so :math:`H_\theta > 0`.
- ``DissipationNet``: emits a lower-triangular Cholesky factor ``L`` and returns
  :math:`R_\theta = L^\top L + \varepsilon I \succeq 0` via
  ``utils.hamiltonian.make_positive_definite``.
- ``PortHamiltonianDecoder``: the full decoder enforcing §3.1::

      f_hat = J(q) grad_H - [0; R_theta(q) (dH/dp)] + B(x_p) u + W_corr c_t + b_corr

  with :math:`J = -J^\top` (skew-symmetric), :math:`R_\theta \succeq 0`,
  :math:`H_\theta > 0`. The Hamiltonian gradient is taken with autograd and
  ``create_graph=True`` so the decoder is differentiable end-to-end.

  Dissipation is pH-consistent (audit #4): damping enters the momentum
  (:math:`\dot p`) equation acting against the Hamiltonian velocity
  :math:`\partial H/\partial p`, so :math:`f_{diss} = [0;\, -R_\theta\,
  \partial H/\partial p]` and the dissipated power
  :math:`P_{diss} = (\partial H/\partial p)^\top R_\theta (\partial H/\partial p)`
  equals :math:`-\nabla H\cdot f_{diss}` exactly. The port-Hamiltonian energy
  residual :math:`\|dH/dt - P_{control} + P_{diss}\|^2` therefore vanishes by
  construction for the conservative + dissipative + control terms (only the
  learned temporal correction ``f_corr`` contributes), which is the soft-loss
  weakness the PCML hard-constraint layer is designed to supersede.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pits_mras.utils.hamiltonian import (
    hamiltonian_positivity_loss,
    make_positive_definite,
    make_skew_symmetric,
    port_hamiltonian_energy_loss,
)


class HamiltonianNet(nn.Module):
    """Learns the scalar Hamiltonian :math:`H_\\theta(q, p) > 0` (total energy).

    Architecture: a 2-hidden-layer ``Tanh`` MLP with a ``Softplus`` output so
    the energy is strictly positive.
    """

    def __init__(self, n_q: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_q, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # guarantees H > 0
        )

    def forward(self, q: Tensor, p: Tensor) -> Tensor:
        """Return :math:`H_\\theta(q, p)` of shape ``[batch, 1]``."""
        return self.net(torch.cat([q, p], dim=-1))


class DissipationNet(nn.Module):
    r"""Learns the dissipation matrix :math:`R_\theta(q) = L^\top L \succeq 0`.

    The network emits the entries of ``L``; we keep its lower-triangular part,
    softplus the diagonal (proper Cholesky factor), then return
    :math:`L^\top L + \varepsilon I` via ``make_positive_definite``.
    """

    def __init__(self, n_q: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.n_q = n_q
        self.net = nn.Sequential(
            nn.Linear(n_q, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_q * n_q),  # reshaped to [n_q, n_q]
        )

    def forward(self, q: Tensor) -> Tensor:
        """Return :math:`R_\\theta(q)` of shape ``[batch, n_q, n_q]``."""
        batch = q.shape[0]
        raw = self.net(q).view(batch, self.n_q, self.n_q)  # [batch, n_q, n_q]
        # Proper Cholesky factor: strictly-lower part + softplus(diagonal) > 0.
        L = torch.tril(raw, diagonal=-1)
        diag = F.softplus(torch.diagonal(raw, dim1=-2, dim2=-1))  # [batch, n_q]
        L = L + torch.diag_embed(diag)
        return make_positive_definite(L)  # [batch, n_q, n_q]


class PortHamiltonianDecoder(nn.Module):
    r"""Full port-Hamiltonian decoder implementing §3.1.

    Given a context vector ``c_t`` and the current state ``x_p = [q; p]``::

        f_hat = J(q) grad_H_theta - [0; R_theta(q) (dH/dp)] + B(x_p) u + W_corr c_t

    Dissipation enters the momentum (p) block via the velocity ``dH/dp`` (see
    the module docstring), so it is pH-consistent and the energy residual
    vanishes by construction.
    """

    J_canonical: Tensor

    def __init__(
        self,
        n_q: int,  # generalized coordinate dimension
        context_dim: int,  # dimension of attention context c_t
        output_dim: int,  # full output dimension (== 2 * n_q for [q, p])
        hamiltonian_hidden: int = 64,
        dissipation_hidden: int = 32,
        use_position_dependent_J: bool = False,
        control_dim: int = 1,  # dimension of the control input u (MIMO, G8)
    ) -> None:
        super().__init__()
        self.n_q = n_q
        self.control_dim = control_dim
        self.use_position_dependent_J = use_position_dependent_J
        self.H_net = HamiltonianNet(n_q, hamiltonian_hidden)
        self.L_net = DissipationNet(n_q, dissipation_hidden)
        if use_position_dependent_J:
            # Learned J(q) for nonholonomic constraints (skew-symmetrized below).
            self.J_net = nn.Sequential(
                nn.Linear(n_q, 32),
                nn.Tanh(),
                nn.Linear(32, (2 * n_q) ** 2),
            )
        else:
            # Constant canonical J: [[0, I], [-I, 0]].
            J_canon = torch.zeros(2 * n_q, 2 * n_q)
            J_canon[:n_q, n_q:] = torch.eye(n_q)
            J_canon[n_q:, :n_q] = -torch.eye(n_q)
            self.register_buffer("J_canonical", J_canon)
        # Input matrix B(x_p) (state-dependent, small MLP). MIMO-general (G8):
        # emits a flattened input matrix of shape [batch, 2*n_q * control_dim],
        # reshaped to [batch, 2*n_q, control_dim] in forward(). At control_dim=1
        # the head width is 2*n_q -- identical to the pre-G8 single-input layout,
        # so weights/initialization (and thus the computation) are preserved.
        self.B_net = nn.Sequential(
            nn.Linear(2 * n_q, 32),
            nn.Tanh(),
            nn.Linear(32, 2 * n_q * control_dim),
        )
        # Temporal correction from the attention context.
        self.W_corr = nn.Linear(context_dim, output_dim)

    def get_J(self, q: Tensor) -> Tensor:
        r"""Return the (batched) interconnection matrix ``J``, always skew-symmetric.

        Shape ``[batch, 2*n_q, 2*n_q]``. For the canonical case this is the
        constant :math:`[[0, I], [-I, 0]]` broadcast over the batch; for the
        position-dependent case it is ``make_skew_symmetric(J_net(q))``.
        """
        batch = q.shape[0]
        if self.use_position_dependent_J:
            J_flat = self.J_net(q)
            J = J_flat.view(batch, 2 * self.n_q, 2 * self.n_q)
            return make_skew_symmetric(J)
        return self.J_canonical.unsqueeze(0).expand(batch, -1, -1)

    def forward(
        self,
        q: Tensor,  # [batch, n_q] generalized positions
        p: Tensor,  # [batch, n_q] generalized momenta
        q_dot: Tensor,  # [batch, n_q] velocity (for dissipation)
        u: Tensor,  # [batch, control_dim] control input
        c_t: Tensor,  # [batch, context_dim] attention context
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the decoder.

        Returns ``(f_hat, H_val, P_diss, energy_loss)``:
            f_hat: ``[batch, 2*n_q]`` -- full dynamics prediction.
            H_val: ``[batch, 1]`` -- Hamiltonian energy (for monitoring).
            P_diss: ``[batch]`` -- dissipated power ``(dH/dp)^T R (dH/dp) >= 0``.
            energy_loss: scalar -- port-Hamiltonian energy-residual loss.
        """
        qp = torch.cat([q, p], dim=-1).requires_grad_(True)  # [batch, 2*n_q]
        q_in = qp[:, : self.n_q]
        p_in = qp[:, self.n_q :]

        # 1. Hamiltonian and its gradient (create_graph for higher-order grads).
        H_val = self.H_net(q_in, p_in)  # [batch, 1]
        grad_H = torch.autograd.grad(H_val.sum(), qp, create_graph=True)[0]  # [batch, 2*n_q]

        # 2. Interconnection matrix J (always skew-symmetric).
        J = self.get_J(q)  # [batch, 2n_q, 2n_q]

        # 3. Conservative dynamics: f_cons = J grad_H.
        f_cons = (J @ grad_H.unsqueeze(-1)).squeeze(-1)  # [batch, 2*n_q]

        # 4. Dissipative dynamics (pH-consistent, audit #4). In canonical
        #    coordinates the velocity is dH/dp, and Rayleigh damping enters the
        #    momentum (p-dot) equation: p_dot += -R(q) . dH/dp. Using the
        #    Hamiltonian velocity dH/dp (not the finite-difference q_dot) makes
        #    the dissipated power P_diss = (dH/dp)^T R (dH/dp) equal exactly
        #    -grad_H . f_diss, so the energy residual vanishes by construction.
        #    q_dot is kept in the signature for backward compatibility but is
        #    not used here. Built out-of-place via torch.cat.
        R_theta = self.L_net(q)  # [batch, n_q, n_q]
        grad_H_p = grad_H[:, self.n_q :]  # [batch, n_q] = dH/dp (the velocity)
        f_diss_p = -(R_theta @ grad_H_p.unsqueeze(-1)).squeeze(-1)  # [batch, n_q]
        f_diss = torch.cat([torch.zeros_like(q), f_diss_p], dim=-1)

        # 5. Control input: f_ctrl = B(x_p) @ u (true MIMO input matrix, IP §5.2 /
        #    G8). B_net emits a flattened [batch, 2*n_q * control_dim] which is
        #    reshaped to the input matrix B_mat = [batch, 2*n_q, control_dim], then
        #    contracted with u = [batch, control_dim] via a batched matmul:
        #    f_ctrl = B_mat @ u -> [batch, 2*n_q]. At control_dim=1 this reduces
        #    exactly to the old single-input form B_val * u (since u has a single
        #    column, its sum equals itself).
        batch = q.shape[0]
        B_mat = self.B_net(torch.cat([q, p], dim=-1)).view(
            batch, 2 * self.n_q, self.control_dim
        )  # [batch, 2*n_q, control_dim]
        f_ctrl = torch.bmm(B_mat, u.unsqueeze(-1)).squeeze(-1)  # [batch, 2*n_q]

        # 6. Temporal correction from the attention context.
        f_corr = self.W_corr(c_t)  # [batch, 2*n_q]

        # 7. Total dynamics (all out-of-place).
        f_hat = f_cons + f_diss + f_ctrl + f_corr  # [batch, 2*n_q]

        # 8. Energy-loss terms for the physics loss L_physics. Dissipated power
        #    uses the SAME velocity channel (dH/dp) as f_diss, so that
        #    P_diss == -grad_H . f_diss and the residual cancels analytically.
        P_control = (f_ctrl * grad_H).sum(dim=-1)  # [batch]
        P_diss = (grad_H_p.unsqueeze(1) @ R_theta @ grad_H_p.unsqueeze(-1)).reshape(
            -1
        )  # [batch]  (dH/dp)^T R (dH/dp) >= 0
        # dH/dt ~= f_hat . grad_H (chain rule).
        dH_dt = (f_hat * grad_H).sum(dim=-1)  # [batch]
        energy_loss = port_hamiltonian_energy_loss(H_val.squeeze(-1), dH_dt, P_control, P_diss)
        energy_loss = energy_loss + hamiltonian_positivity_loss(H_val.squeeze(-1))
        return f_hat, H_val, P_diss, energy_loss
