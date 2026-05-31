"""Multi-head physics-informed attention (IP §5.1).

Owning phase: Phase 2 (Neural Network Models).

``PhysicsInformedAttention`` combines three complementary attention types:

1. Temporal attention: scaled dot-product over the LSTM hidden states -- when
   in the past is relevant?
2. Physical attention: which physical quantities (position, velocity, force)
   matter, via a learned map over ``[x_p, x_p_dot, u]``?
3. Error-driven attention: cosine similarity between the current tracking error
   and each past error -- which past moments resembled the present?

A learned 3-way softmax gate fuses the three attention distributions into a
single weight vector ``alpha`` and a context vector ``c_t``. Implemented per the
exact dims/forward math of IP §5.1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PhysicsInformedAttention(nn.Module):
    """Three-headed attention module for the PITNN encoder.

    Inputs (forward):
        H_enc: ``[batch, T, d_k]`` -- encoded hidden states from the LSTM.
        e_hist: ``[batch, T, e_dim]`` -- tracking error history.
        x_p: ``[batch, n_state]`` -- current plant state.
        e_curr: ``[batch, e_dim]`` -- current tracking error.
        x_p_dot: ``[batch, n_state]`` -- current plant-state velocity.
        u_curr: ``[batch, control_dim]`` -- current control input.

    Outputs:
        context: ``[batch, d_k]`` -- weighted context vector ``c_t``.
        alpha: ``[batch, T]`` -- combined attention weights (rows sum to 1).
    """

    def __init__(
        self,
        d_k: int,  # key/query dimension
        e_dim: int,  # tracking error dimension
        n_state: int,  # full state dimension
        n_heads: int = 4,
        control_dim: int | None = None,  # control dim; defaults to n_state
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        # Physical descriptor is [x_p, x_p_dot, u] = 2*n_state + control_dim.
        # IP §5.1 wrote ``n_state * 3`` assuming control_dim == n_state; we size
        # it correctly for the general (control_dim != n_state) case.
        if control_dim is None:
            control_dim = n_state
        self.control_dim = control_dim
        phys_in = 2 * n_state + control_dim
        # Temporal attention: standard scaled dot-product.
        self.W_Q = nn.Linear(d_k, d_k, bias=False)
        self.W_K = nn.Linear(d_k, d_k, bias=False)
        # Physical attention: maps state/velocity/force to an importance score.
        self.W_phys = nn.Linear(phys_in, 1)  # [x_p, x_p_dot, u] concatenated
        # Error-driven attention: optional projection of the current error.
        self.W_e = nn.Linear(e_dim, e_dim, bias=False)
        # Gating network: which attention type to trust?
        gate_input_dim = d_k + e_dim + n_state
        self.W_gate = nn.Linear(gate_input_dim, 3)  # 3-way softmax

    def forward(
        self,
        H_enc: Tensor,  # [batch, T, d_k]
        e_hist: Tensor,  # [batch, T, e_dim]
        x_p: Tensor,  # [batch, n_state]
        e_curr: Tensor,  # [batch, e_dim]
        x_p_dot: Tensor,  # [batch, n_state]
        u_curr: Tensor,  # [batch, control_dim]
    ) -> tuple[Tensor, Tensor]:
        batch, T, _ = H_enc.shape
        h_t = H_enc[:, -1, :]  # [batch, d_k] -- current hidden state

        # --- 1. Temporal attention (scaled dot-product) ---
        Q = self.W_Q(h_t).unsqueeze(1)  # [batch, 1, d_k]
        K = self.W_K(H_enc)  # [batch, T, d_k]
        scores_time = (Q @ K.transpose(-1, -2)).squeeze(1) / math.sqrt(self.d_k)
        alpha_time = F.softmax(scores_time, dim=-1)  # [batch, T]

        # --- 2. Physical attention ---
        # The physical descriptor is time-independent here, so broadcast over T.
        phys_feat = torch.cat([x_p, x_p_dot, u_curr], dim=-1)  # [batch, n_state*3]
        scores_phys = self.W_phys(phys_feat).expand(batch, T)  # [batch, T]
        alpha_phys = F.softmax(scores_phys, dim=-1)  # [batch, T]

        # --- 3. Error-driven attention (cosine similarity) ---
        e_proj = self.W_e(e_curr).unsqueeze(1)  # [batch, 1, e_dim]
        e_hist_norm = F.normalize(e_hist, dim=-1)  # [batch, T, e_dim]
        e_curr_norm = F.normalize(e_proj, dim=-1)  # [batch, 1, e_dim]
        scores_err = (e_curr_norm @ e_hist_norm.transpose(-1, -2)).squeeze(1)
        alpha_err = F.softmax(scores_err, dim=-1)  # [batch, T]

        # --- 4. Learned gating ---
        gate_input = torch.cat([h_t, e_curr, x_p], dim=-1)  # [batch, gate_dim]
        g = F.softmax(self.W_gate(gate_input), dim=-1)  # [batch, 3]
        g1, g2, g3 = g[:, 0:1], g[:, 1:2], g[:, 2:3]  # [batch, 1] each

        # --- 5. Combined attention (convex combination -> rows sum to 1) ---
        alpha = g1 * alpha_time + g2 * alpha_phys + g3 * alpha_err  # [batch, T]

        # --- 6. Context vector ---
        context = (alpha.unsqueeze(-1) * H_enc).sum(dim=1)  # [batch, d_k]
        return context, alpha

    def attention_regularization_loss(
        self, alpha: Tensor, lambda_sparse: float = 0.01
    ) -> Tensor:
        r"""Regularize attention weights: balance entropy with sparsity.

        :math:`L_{attn} = -\mathrm{entropy}(\alpha) + \lambda_{sparse}\,\|\alpha\|_1`,
        where the entropy term encourages spread (avoids collapse onto one step)
        and the L1 term encourages sparsity. Returns a scalar.
        """
        eps = 1e-8
        entropy_term = -(alpha * (alpha + eps).log()).sum(dim=-1).mean()
        sparsity_term = alpha.abs().sum(dim=-1).mean()
        return -entropy_term + lambda_sparse * sparsity_term
