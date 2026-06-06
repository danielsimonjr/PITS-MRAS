r"""GENERIC / GFINN thermodynamic decoder (ROADMAP extension).

Owning phase: Phase 2 (Neural Network Models) -- thermodynamic generalization
of the port-Hamiltonian decoder in ``models.decoders``.

This module implements a GFINN-style decoder for the **GENERIC** formalism
(General Equation for the Non-Equilibrium Reversible-Irreversible Coupling;
Oettinger; Zhang, Shin & Karniadakis 2022, "GFINNs: GENERIC Formalism Informed
Neural Networks"). A GENERIC system evolves as

.. math::

    \dot z = L(z)\,\nabla E(z) + M(z)\,\nabla S(z)

with the **structural requirements**

* :math:`L` skew-symmetric: :math:`L^\top = -L`,
* :math:`M` symmetric positive semidefinite: :math:`M = M^\top \succeq 0`,
* the **degeneracy conditions** :math:`L(z)\,\nabla S(z) = 0` and
  :math:`M(z)\,\nabla E(z) = 0`.

Together these guarantee the **first** and **second** laws of thermodynamics
*by construction* (not via a soft penalty):

* First law (energy conservation):
  :math:`\dot E = \nabla E^\top \dot z = \nabla E^\top L \nabla E
  + \nabla E^\top M \nabla S = 0`, because :math:`L` skew makes the first term
  vanish (:math:`x^\top L x = 0`) and :math:`M\nabla E = 0` (so
  :math:`\nabla E^\top M = (M\nabla E)^\top = 0`) kills the second.
* Second law (entropy production):
  :math:`\dot S = \nabla S^\top \dot z = \nabla S^\top L \nabla E
  + \nabla S^\top M \nabla S = \nabla S^\top M \nabla S \ge 0`, because
  :math:`L\nabla S = 0` removes the reversible term and :math:`M \succeq 0`
  makes the irreversible term non-negative.

Construction (the crux -- degeneracy is exact, by construction)
---------------------------------------------------------------
Let :math:`P_v(x) = x - \dfrac{v^\top x}{v^\top v + \varepsilon}\,v` be the
(soft) projector that removes the component of ``x`` along ``v``. After
projection, :math:`v^\top P_v(x) \approx 0` to machine precision (exactly
zero in the :math:`\varepsilon\to 0` limit; with a tiny ``eps`` the residual
is :math:`O(\varepsilon)` and well below the test tolerance).

* **L (skew, annihilates ``∇S``).** A learned network emits ``n_skew`` pairs of
  vector fields :math:`a_k(z), b_k(z)`. Each is projected orthogonal to
  :math:`\nabla S`: :math:`\hat a_k = P_{\nabla S}(a_k)`,
  :math:`\hat b_k = P_{\nabla S}(b_k)`. Then

  .. math::  L(z) = \sum_k \big(\hat a_k \hat b_k^\top - \hat b_k \hat a_k^\top\big).

  Skew-symmetry is automatic from the antisymmetric :math:`(ab^\top - ba^\top)`
  form. Degeneracy holds because
  :math:`L\nabla S = \sum_k \hat a_k(\hat b_k^\top\nabla S)
  - \hat b_k(\hat a_k^\top\nabla S) = 0` (each projected vector is orthogonal
  to :math:`\nabla S`).

* **M (symmetric PSD, annihilates ``∇E``).** A learned network emits a matrix
  :math:`D(z)` with ``n_friction`` columns :math:`d_k(z)`. Each column is
  projected orthogonal to :math:`\nabla E`: :math:`\hat d_k = P_{\nabla E}(d_k)`.
  Then

  .. math::  M(z) = \hat D \hat D^\top = \sum_k \hat d_k \hat d_k^\top.

  :math:`M = M^\top` and :math:`x^\top M x = \sum_k (\hat d_k^\top x)^2 \ge 0`
  (PSD) by construction. Degeneracy holds because
  :math:`M\nabla E = \sum_k \hat d_k(\hat d_k^\top \nabla E) = 0`.

The scalar potentials :math:`E(z)` and :math:`S(z)` are learned MLPs; their
gradients are taken with autograd and ``create_graph=True`` so the whole field
is differentiable end-to-end (grads flow to the E, S, L and M parameters).
"""

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["GFINNDecoder"]


def _project_orthogonal(x: Tensor, v: Tensor, eps: float = 1e-12) -> Tensor:
    r"""Remove the component of ``x`` along ``v`` (batched, last-dim vectors).

    Computes :math:`P_v(x) = x - \dfrac{v^\top x}{v^\top v + \varepsilon}\,v`
    so that :math:`v^\top P_v(x) = O(\varepsilon)`. Both ``x`` and ``v`` have
    shape ``[batch, ..., dim]`` and broadcast over the leading dims; ``v`` is
    the (broadcast) direction to annihilate.
    """
    vtx = (v * x).sum(dim=-1, keepdim=True)  # [..., 1]
    vtv = (v * v).sum(dim=-1, keepdim=True)  # [..., 1]
    return x - (vtx / (vtv + eps)) * v


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """Build a Tanh MLP ``in_dim -> hidden... -> out_dim`` (no output activation)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GFINNDecoder(nn.Module):
    r"""GENERIC / GFINN thermodynamic decoder.

    Learns scalar potentials :math:`E(z)` (energy) and :math:`S(z)` (entropy)
    and parameterizes the skew operator :math:`L(z)` and the PSD friction
    operator :math:`M(z)` so that the GENERIC degeneracy conditions hold *by
    construction*. ``forward(z)`` returns
    :math:`\dot z = L(z)\nabla E + M(z)\nabla S`.

    Parameters
    ----------
    dim:
        State dimension ``d``. Inputs are ``[batch, dim]``.
    energy_hidden, entropy_hidden:
        Hidden-layer widths for the energy / entropy MLPs.
    skew_hidden, friction_hidden:
        Hidden-layer widths for the ``L`` / ``M`` generator MLPs.
    n_skew:
        Number of antisymmetric ``(a_k b_k^T - b_k a_k^T)`` rank-2 terms in
        ``L``. Each contributes one learned ``a_k`` and one ``b_k`` field.
    n_friction:
        Number of columns of the friction factor ``D`` (rank of ``M``).
    eps:
        Tiny constant in the orthogonal projector denominator (degeneracy is
        exact as ``eps -> 0``; the default keeps residuals far below tolerance
        while avoiding division by zero when a gradient is ~0).
    """

    def __init__(
        self,
        dim: int,
        energy_hidden: tuple[int, ...] = (64, 64),
        entropy_hidden: tuple[int, ...] = (64, 64),
        skew_hidden: tuple[int, ...] = (64,),
        friction_hidden: tuple[int, ...] = (64,),
        n_skew: int = 4,
        n_friction: int = 4,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_skew = n_skew
        self.n_friction = n_friction
        self.eps = eps

        # Scalar potentials E(z), S(z): [B, dim] -> [B, 1].
        self.energy_net = _mlp(dim, energy_hidden, 1)
        self.entropy_net = _mlp(dim, entropy_hidden, 1)

        # L generator: emits 2 * n_skew vector fields (the a_k and b_k stacked).
        self.skew_net = _mlp(dim, skew_hidden, 2 * n_skew * dim)
        # M generator: emits n_friction columns of the factor D.
        self.friction_net = _mlp(dim, friction_hidden, n_friction * dim)

    # ------------------------------------------------------------------ #
    # Scalar potentials and their gradients.
    # ------------------------------------------------------------------ #
    def energy(self, z: Tensor) -> Tensor:
        """Energy potential :math:`E(z)` of shape ``[batch, 1]``."""
        return self.energy_net(z)

    def entropy(self, z: Tensor) -> Tensor:
        """Entropy potential :math:`S(z)` of shape ``[batch, 1]``."""
        return self.entropy_net(z)

    def _grad(self, phi_net: nn.Module, z: Tensor) -> Tensor:
        """Autograd gradient of a scalar potential w.r.t. ``z`` (``create_graph``)."""
        z_req = z if z.requires_grad else z.detach().requires_grad_(True)
        phi = phi_net(z_req)  # [B, 1]
        (grad,) = torch.autograd.grad(phi.sum(), z_req, create_graph=True, retain_graph=True)
        return grad  # [B, dim]

    def grad_E(self, z: Tensor) -> Tensor:
        """Gradient :math:`\\nabla E(z)` of shape ``[batch, dim]``."""
        return self._grad(self.energy_net, z)

    def grad_S(self, z: Tensor) -> Tensor:
        """Gradient :math:`\\nabla S(z)` of shape ``[batch, dim]``."""
        return self._grad(self.entropy_net, z)

    # ------------------------------------------------------------------ #
    # Structural operators L (skew, L∇S=0) and M (PSD, M∇E=0).
    # ------------------------------------------------------------------ #
    def L(self, z: Tensor) -> Tensor:
        r"""Skew-symmetric operator :math:`L(z)` with :math:`L\,\nabla S = 0`.

        Shape ``[batch, dim, dim]``. Built as
        :math:`\sum_k (\hat a_k \hat b_k^\top - \hat b_k \hat a_k^\top)` with
        :math:`\hat a_k, \hat b_k` projected orthogonal to :math:`\nabla S`.
        """
        batch = z.shape[0]
        gS = self.grad_S(z)  # [B, dim]
        raw = self.skew_net(z).view(batch, 2 * self.n_skew, self.dim)
        a = raw[:, : self.n_skew, :]  # [B, n_skew, dim]
        b = raw[:, self.n_skew :, :]  # [B, n_skew, dim]
        gS_b = gS.unsqueeze(1)  # [B, 1, dim] broadcast over the n_skew terms
        a_hat = _project_orthogonal(a, gS_b, self.eps)  # [B, n_skew, dim]
        b_hat = _project_orthogonal(b, gS_b, self.eps)  # [B, n_skew, dim]
        # L = sum_k a_k b_k^T - b_k a_k^T  (batched outer-product accumulation).
        ab = torch.einsum("bki,bkj->bij", a_hat, b_hat)  # [B, dim, dim]
        return ab - ab.transpose(-1, -2)  # skew-symmetric by construction

    def M(self, z: Tensor) -> Tensor:
        r"""Symmetric PSD operator :math:`M(z)` with :math:`M\,\nabla E = 0`.

        Shape ``[batch, dim, dim]``. Built as :math:`\hat D \hat D^\top` with
        the columns of :math:`\hat D` projected orthogonal to :math:`\nabla E`.
        """
        batch = z.shape[0]
        gE = self.grad_E(z)  # [B, dim]
        # Columns of D as [B, n_friction, dim] (each row is one column vector).
        cols = self.friction_net(z).view(batch, self.n_friction, self.dim)
        gE_b = gE.unsqueeze(1)  # [B, 1, dim]
        cols_hat = _project_orthogonal(cols, gE_b, self.eps)  # [B, n_friction, dim]
        # M = sum_k d_k d_k^T  (PSD by construction).
        return torch.einsum("bki,bkj->bij", cols_hat, cols_hat)  # [B, dim, dim]

    # ------------------------------------------------------------------ #
    # GENERIC vector field.
    # ------------------------------------------------------------------ #
    def forward(self, z: Tensor) -> Tensor:
        r"""GENERIC field :math:`\dot z = L(z)\nabla E + M(z)\nabla S`.

        Returns ``[batch, dim]``. By the by-construction degeneracy this
        conserves energy (:math:`\nabla E^\top \dot z = 0`) and produces
        entropy (:math:`\nabla S^\top \dot z \ge 0`).
        """
        gE = self.grad_E(z)  # [B, dim]
        gS = self.grad_S(z)  # [B, dim]
        L = self.L(z)  # [B, dim, dim]
        M = self.M(z)  # [B, dim, dim]
        reversible = torch.bmm(L, gE.unsqueeze(-1)).squeeze(-1)  # [B, dim]
        irreversible = torch.bmm(M, gS.unsqueeze(-1)).squeeze(-1)  # [B, dim]
        return reversible + irreversible
