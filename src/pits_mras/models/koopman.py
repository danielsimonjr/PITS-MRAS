r"""Deep Koopman lifting model (ROADMAP proposal #2). NEW capability.

Owning phase: Phase 2 (Neural Network Models).

Faithful to Lusch et al. 2018 ("Deep learning for universal linear embeddings
of nonlinear dynamics") extended with control inputs. PITS-MRAS controls
nonlinear plants, but its *verifiable* core (the quadratic critic
:math:`\hat V = e^\top P e`, :func:`~pits_mras.utils.lyapunov.solve_care` /
``solve_gare``, CLF-CBF) is LINEAR. A deep Koopman model learns an encoder that
lifts the nonlinear state :math:`x` into a latent space :math:`z = g(x)` where
the dynamics are approximately LINEAR

.. math::
    z_{k+1} \approx A_z\, z_k + B_z\, u_k,

so the existing linear machinery can be applied on lifted coordinates.

This module delivers the *lifting model + its training losses* only. It is
intentionally NOT wired into the training/control loop -- full control-loop
integration (passing ``A_z, B_z`` to ``solve_care``/``solve_gare`` and closing
the loop on lifted coordinates) is a documented follow-on. The bridge to the
linear core is exposed via :meth:`KoopmanLiftingModel.latent_matrices`.

Design (stable, testable):
- ``include_state=True`` (default): the lift is :math:`z = [\,x;\ \psi(x)\,]`
  where :math:`\psi` is a Tanh MLP producing the ``latent_dim - state_dim``
  *extra* observables. The state is then directly recoverable, so the decoder is
  the exact slice ``z[:, :state_dim]`` and the reconstruction loss is zero by
  construction. ``A_z, B_z`` are initialized with zero rows on the extra-coord
  block and :math:`\psi` is initialized to output zero, so the model starts as
  the trivial linear-in-state predictor (a stable warm start) before training.
- ``include_state=False``: the encoder MLP outputs all ``latent_dim``
  observables and the decoder is a learned linear readout
  ``nn.Linear(latent_dim, state_dim)``.

Shape convention: ``[batch, dim]``, float32 tensors throughout.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class KoopmanLiftingModel(nn.Module):
    r"""Learnable encoder + exactly-linear latent dynamics + decoder.

    The latent step :math:`z_{k+1} = z_k A_z^\top + u_k B_z^\top` is *exactly*
    linear in :math:`(z, u)` (no bias, no nonlinearity), which is what lets the
    linear control core operate on the lifted coordinates.

    Args:
        state_dim: dimension of the plant state :math:`x`.
        control_dim: dimension of the control input :math:`u`.
        latent_dim: dimension of the lifted observable :math:`z`.
        encoder_hidden: hidden widths of the encoder MLP (Tanh activations).
        include_state: if ``True`` (default), :math:`z = [x; \psi(x)]` and the
            decoder is the exact state slice; requires ``latent_dim >=
            state_dim``. If ``False``, the encoder outputs all ``latent_dim``
            coords and the decoder is a learned linear readout.
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        latent_dim: int,
        encoder_hidden: Tuple[int, ...] = (64, 64),
        include_state: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or control_dim <= 0 or latent_dim <= 0:
            raise ValueError(
                "state_dim, control_dim, latent_dim must all be > 0; got "
                f"state_dim={state_dim}, control_dim={control_dim}, latent_dim={latent_dim}."
            )
        if include_state and latent_dim < state_dim:
            raise ValueError(
                "include_state=True puts the state inside the lift, so "
                f"latent_dim ({latent_dim}) must be >= state_dim ({state_dim})."
            )

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.include_state = include_state

        # -- Encoder MLP (Tanh hidden). --
        # With include_state it emits only the EXTRA coords (psi); otherwise all.
        # When latent_dim == state_dim (include_state) there are no extra coords,
        # so no encoder MLP is built and ``encode`` returns the state directly.
        encoder_out = latent_dim - state_dim if include_state else latent_dim
        self.encoder = (
            _build_mlp(state_dim, encoder_out, encoder_hidden) if encoder_out > 0 else None
        )

        # -- Exactly-linear latent dynamics (no bias). --
        # nn.Parameter, used as z @ A_z^T + u @ B_z^T so the step is linear.
        self.A_z = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.B_z = nn.Parameter(torch.empty(latent_dim, control_dim))

        # -- Decoder. --
        # include_state=True: exact slice (no params). False: linear readout.
        if not include_state:
            self.decoder = nn.Linear(latent_dim, state_dim)

        self._init_parameters()

    def _init_parameters(self) -> None:
        r"""Initialize to a stable warm start.

        ``A_z`` near identity (latent dynamics start as "hold"), ``B_z`` small.
        When ``include_state=True`` the extra-coord block (the bottom
        ``latent_dim - state_dim`` rows) of ``A_z``/``B_z`` is zeroed and the
        encoder's final layer is zeroed, so the model begins as the trivial
        linear-in-state predictor ``x_{k+1} = x_k A[:s,:s]^T + u B[:s]^T`` with
        the extra observables identically zero -- a self-consistent starting
        point that keeps the latent-linearity loss well-behaved.
        """
        with torch.no_grad():
            nn.init.eye_(self.A_z)
            nn.init.normal_(self.B_z, std=0.01)
            if self.include_state:
                s = self.state_dim
                # Extra-coord rows do not (yet) feed back -- they start at zero.
                self.A_z[s:, :].zero_()
                self.B_z[s:, :].zero_()
                # Make psi output exactly zero at init (zero last linear layer).
                if self.encoder is not None:
                    last = _last_linear(self.encoder)
                    if last is not None:
                        last.weight.zero_()
                        if last.bias is not None:
                            last.bias.zero_()

    # ----------------------------------------------------------------- #
    # Encoder / decoder / dynamics
    # ----------------------------------------------------------------- #
    def encode(self, x: Tensor) -> Tensor:
        r"""Lift the state into the latent observable space.

        Returns ``z`` of shape ``[batch, latent_dim]``. With
        ``include_state=True`` this is ``concat([x, psi(x)])`` so the state is
        recoverable exactly; otherwise it is the raw encoder output.
        """
        if self.include_state:
            if self.encoder is None:
                return x  # latent_dim == state_dim: the lift is the state itself
            psi = self.encoder(x)  # [batch, latent_dim - state_dim]
            return torch.cat([x, psi], dim=-1)
        assert self.encoder is not None  # include_state=False always builds it
        return self.encoder(x)

    def latent_step(self, z: Tensor, u: Tensor) -> Tensor:
        r"""One step of the EXACTLY linear latent dynamics.

        :math:`z_{k+1} = z_k A_z^\top + u_k B_z^\top`. Linear in ``(z, u)`` by
        construction (no bias, no activation). Returns ``[batch, latent_dim]``.
        """
        return z @ self.A_z.T + u @ self.B_z.T

    def decode(self, z: Tensor) -> Tensor:
        r"""Map a latent observable back to the state.

        With ``include_state=True`` this is the exact slice ``z[:, :state_dim]``
        (zero-error state recovery). Otherwise it is the learned linear readout.
        Returns ``[batch, state_dim]``.
        """
        if self.include_state:
            return z[:, : self.state_dim]
        return self.decoder(z)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        r"""One-step state prediction: ``decode(latent_step(encode(x), u))``.

        Returns the predicted next state ``[batch, state_dim]``.
        """
        return self.decode(self.latent_step(self.encode(x), u))

    def latent_matrices(self) -> Tuple[Tensor, Tensor]:
        r"""Return the dense latent dynamics ``(A_z, B_z)``.

        ``A_z`` is ``[latent_dim, latent_dim]`` and ``B_z`` is
        ``[latent_dim, control_dim]``. This is the bridge to the linear core: a
        caller may detach these and pass them (as numpy) to
        :func:`~pits_mras.utils.lyapunov.solve_care` / ``solve_gare`` on the
        lifted coordinates. (This method does NOT call those solvers -- control
        integration is a documented follow-on.)
        """
        return self.A_z, self.B_z


def koopman_loss(
    model: KoopmanLiftingModel,
    x: Tensor,
    u: Tensor,
    x_next: Tensor,
    *,
    w_recon: float = 1.0,
    w_pred: float = 1.0,
    w_lin: float = 1.0,
) -> Dict[str, Tensor]:
    r"""Deep-Koopman training losses (Lusch et al. 2018, with control).

    Computes the three canonical terms and a weighted total. All terms are mean
    squared errors (scalars).

    Terms (let ``g = encode``, ``d = decode``, ``L(z, u) = latent_step``):
        ``recon`` -- reconstruction: :math:`\lVert d(g(x)) - x \rVert^2`.
            Zero by construction when ``include_state=True``.
        ``lin`` -- latent linearity (prediction *in latent space*):
            :math:`\lVert g(x_{next}) - L(g(x), u) \rVert^2`.
            Drives the encoded next state to match the linear latent rollout.
        ``pred`` -- state prediction:
            :math:`\lVert d(L(g(x), u)) - x_{next} \rVert^2`.

    Returns:
        Dict with keys ``"recon"``, ``"lin"``, ``"pred"`` (the individual terms)
        and ``"loss"`` (the weighted total
        ``w_recon*recon + w_lin*lin + w_pred*pred``). Gradients flow to the
        encoder, ``A_z``, ``B_z`` and (when present) the decoder.
    """
    z = model.encode(x)
    z_next_pred = model.latent_step(z, u)
    z_next_true = model.encode(x_next)

    x_recon = model.decode(z)
    x_next_pred = model.decode(z_next_pred)

    recon = torch.mean((x_recon - x) ** 2)
    lin = torch.mean((z_next_true - z_next_pred) ** 2)
    pred = torch.mean((x_next_pred - x_next) ** 2)

    loss = w_recon * recon + w_lin * lin + w_pred * pred
    return {"recon": recon, "lin": lin, "pred": pred, "loss": loss}


def _build_mlp(in_dim: int, out_dim: int, hidden: Tuple[int, ...]) -> nn.Sequential:
    """Tanh MLP. If ``hidden`` is empty, a single linear layer (still valid)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _last_linear(module: nn.Module) -> nn.Linear | None:
    """Return the last ``nn.Linear`` in a module (for zero-init), or ``None``."""
    last: nn.Linear | None = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last
