"""Top-level physics-informed time-series network -- PITNN (IP §5.4).

Owning phase: Phase 2 (Neural Network Models).

Implements Algorithm 1: input normalization + embedding -> causal
(forward-only) LSTM encoder -> ``PhysicsInformedAttention`` ->
``PortHamiltonianDecoder``. Mathematical guarantees:

- Causality: forward-only LSTM, no future data leaks into the prediction.
- Energy conservation: port-Hamiltonian decoder structure.
- Physical plausibility: positive dissipation (:math:`R_\\theta = L^\\top L \\succeq 0`).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from pits_mras.config import NetworkConfig, PhysicsConfig
from pits_mras.models.attention import PhysicsInformedAttention
from pits_mras.models.decoders import PortHamiltonianDecoder


class PITNN(nn.Module):
    """Physics-Informed Temporal Neural Network -- the core dynamics model.

    Takes a sliding window of ``(state, control)`` history plus the current
    tracking error and outputs a dynamics prediction
    :math:`\\hat f_\\theta(x_p, u, t)` for the plant. ``n_q`` (generalized
    coordinates) is taken from ``phys_cfg.n_generalized_coords``; the system is
    assumed canonical so ``output_dim == 2 * n_q``.
    """

    mu_x: Tensor
    sigma_x: Tensor

    def __init__(
        self,
        net_cfg: NetworkConfig,
        phys_cfg: PhysicsConfig,
        lagrangian_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.input_dim = net_cfg.input_dim
        self.hidden_dim = net_cfg.hidden_dim
        self.output_dim = net_cfg.output_dim
        self.n_q = phys_cfg.n_generalized_coords
        # Optional PCML Lagrangian-multiplier head (Addendum §2.4). When set, the
        # forward pass emits ``lam_hat`` (KKT warm-start multipliers) for the
        # PCML projection. ``None`` keeps the v0.2.0 output contract unchanged.
        self.lagrangian_head = lagrangian_head

        # -- Input normalization (running statistics, non-trainable buffers) --
        self.register_buffer("mu_x", torch.zeros(net_cfg.input_dim))
        self.register_buffer("sigma_x", torch.ones(net_cfg.input_dim))

        # -- Embedding --
        self.embed_state = nn.Linear(net_cfg.input_dim, net_cfg.embedding_dim)
        self.embed_control = nn.Linear(net_cfg.input_dim, net_cfg.embedding_dim)

        # -- Causal LSTM encoder (forward-only) --
        lstm_input = net_cfg.embedding_dim * 2  # state + control embeddings
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=net_cfg.hidden_dim,
            num_layers=net_cfg.lstm_layers,
            batch_first=True,
        )

        # -- Physics-informed attention --
        # Control enters the canonical port-Hamiltonian system through the
        # momentum channel, so control_dim == n_q for the wired decoder.
        self.attention = PhysicsInformedAttention(
            d_k=net_cfg.hidden_dim,
            e_dim=net_cfg.output_dim,  # error dim ~= output dim for this system
            n_state=net_cfg.input_dim,
            n_heads=net_cfg.attention_heads,
            control_dim=phys_cfg.n_generalized_coords,
        )

        # -- Port-Hamiltonian decoder --
        self.decoder = PortHamiltonianDecoder(
            n_q=phys_cfg.n_generalized_coords,
            context_dim=net_cfg.hidden_dim,
            output_dim=net_cfg.output_dim,
            hamiltonian_hidden=phys_cfg.hamiltonian_hidden,
            dissipation_hidden=phys_cfg.dissipation_hidden,
            use_position_dependent_J=phys_cfg.use_position_dependent_J,
        )

    def update_normalization(self, x_data: Tensor) -> None:
        """Update running normalization statistics from a data batch."""
        self.mu_x.copy_(x_data.mean(dim=0))
        self.sigma_x.copy_(x_data.std(dim=0).clamp(min=1e-6))

    def normalize(self, x: Tensor) -> Tensor:
        """Apply running-statistic normalization."""
        return (x - self.mu_x) / self.sigma_x

    def forward(
        self,
        x_hist: Tensor,  # [batch, T, input_dim] plant-state history
        u_hist: Tensor,  # [batch, T, input_dim] control history
        x_p_curr: Tensor,  # [batch, input_dim] current plant state
        u_curr: Tensor,  # [batch, control_dim] current control
        e_curr: Tensor,  # [batch, e_dim] current tracking error
        e_hist: Tensor,  # [batch, T, e_dim] error history
    ) -> Dict[str, Tensor]:
        """Forward pass (Algorithm 1).

        Returns a dict with the dynamics prediction ``f_hat`` plus the
        monitoring keys ``H_val``, ``context``, ``alpha``, ``h_enc``,
        ``P_diss``, ``energy_loss``, ``attn_reg_loss`` (IP §5.4), and ``lam_hat``
        when a Lagrangian head is attached. (The earlier redundant ``f``/``H``
        aliases of ``f_hat``/``H_val`` were removed in v0.3.1.)
        """
        # 1. Normalize and embed.
        x_norm = self.normalize(x_hist)  # [batch, T, input_dim]
        emb_state = self.embed_state(x_norm)  # [batch, T, emb_dim]
        emb_ctrl = self.embed_control(self.normalize(u_hist))  # [batch, T, emb_dim]
        seq = torch.cat([emb_state, emb_ctrl], dim=-1)  # [batch, T, 2*emb_dim]

        # 2. Causal LSTM (no bidirectional -- preserves causality for deployment).
        H_enc, _ = self.lstm(seq)  # [batch, T, hidden_dim]

        # 3. Velocity approximation (finite difference) for the dissipation term.
        T = x_hist.shape[1]
        if T > 1:
            x_p_dot = (x_hist[:, -1, :] - x_hist[:, -2, :]) / 0.01
        else:
            x_p_dot = torch.zeros_like(x_p_curr)

        # 4. Physics-informed attention.
        context, alpha = self.attention(
            H_enc, e_hist, x_p_curr, e_curr, x_p_dot, u_curr
        )  # context: [batch, hidden_dim], alpha: [batch, T]

        # 5. Port-Hamiltonian decoder -- extract [q, p] from the state.
        q = x_p_curr[:, : self.n_q]  # [batch, n_q]
        p = x_p_curr[:, self.n_q: 2 * self.n_q]  # [batch, n_q]
        q_dot = x_p_dot[:, : self.n_q]  # [batch, n_q]
        f_hat, H_val, P_diss, energy_loss = self.decoder(q, p, q_dot, u_curr, context)

        # 6. Attention regularization.
        attn_reg = self.attention.attention_regularization_loss(alpha)

        # 7. Optional PCML Lagrangian multipliers (KKT warm start, Addendum §2.4).
        lam_hat = self.lagrangian_head(context) if self.lagrangian_head is not None else None

        out: Dict[str, Tensor] = {
            "f_hat": f_hat,  # dynamics prediction (consumed by training/inference)
            "H_val": H_val,
            "context": context,
            "alpha": alpha,
            "h_enc": H_enc,
            "P_diss": P_diss,
            "energy_loss": energy_loss,
            "attn_reg_loss": attn_reg,
        }
        if lam_hat is not None:
            out["lam_hat"] = lam_hat
        return out
