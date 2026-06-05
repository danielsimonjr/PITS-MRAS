r"""Physics-Constrained Machine Learning (PCML) module for PITS-MRAS.

Implements two constraint-enforcement modes that upgrade the soft, port-
Hamiltonian physics regularizer into provable constraint satisfaction:

1. **Soft PCML** (Patel et al., IFAC 2022): augment the loss with constraint
   residuals, ``L = lambda_diff*||D||^2 + lambda_eq*||h||^2 + lambda_ineq*||ReLU(g)||^2``.
   Reduces violations probabilistically; used in early training.
2. **Hard PCML** (DAE-HardNet, Golder et al., arXiv:2512.05881): project the
   network output onto the DAE constraint manifold by solving the KKT system of
   a minimum-distance problem with a differentiable Newton solver, achieving
   point-wise constraint satisfaction. Activated dynamically once the backbone
   data loss drops below ``eta``.

The :class:`PCMLModule` wraps a backbone's prediction ``f_hat`` and returns the
constrained prediction plus the PCML loss, switching between modes by the
``eta`` threshold (DAE-HardNet dynamic activation).

References:
- Patel, Bhartiya & Gudi, *Physics Constrained Learning in NN based Modeling*,
  IFAC-PapersOnLine 55-7 (2022) 79-85. [soft]
- Golder, Roy & Hasan, *DAE-HardNet*, arXiv:2512.05881 (2025). [hard]
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import jacrev, vmap

from pits_mras.constraints.base import PhysicsConstraints

logger = logging.getLogger(__name__)


class SoftPCMLLoss(nn.Module):
    r"""Soft physics-constraint loss (Patel et al. 2022, Eqs. 3-4).

    ``L = lambda_diff*||D(x,t,y,d)||^2 + lambda_eq*||h(x,t,y)||^2
          + lambda_ineq*||ReLU(g(x,t,y))||^2``

    This generalizes the existing port-Hamiltonian ``L_physics`` (energy / PDE /
    BC residuals are special cases of the differential / equality residuals).
    """

    def __init__(
        self,
        constraints: PhysicsConstraints,
        lambda_diff: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 0.5,
    ) -> None:
        super().__init__()
        self.constraints = constraints
        self.lambda_diff = lambda_diff
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq

    def forward(
        self, x: Tensor, t: Tensor, y_pred: Tensor, d_pred: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return ``(total, breakdown)`` with keys ``diff``, ``eq``, ``ineq``, ``violation``."""
        diff = self.constraints.differential(x, t, y_pred, d_pred)
        eq = self.constraints.equality(x, t, y_pred)
        ineq = self.constraints.inequality(x, t, y_pred)

        zero = y_pred.new_zeros(())
        l_diff = (diff**2).mean() if diff.numel() > 0 else zero
        l_eq = (eq**2).mean() if eq.numel() > 0 else zero
        l_ineq = (F.relu(ineq) ** 2).mean() if ineq.numel() > 0 else zero

        total = self.lambda_diff * l_diff + self.lambda_eq * l_eq + self.lambda_ineq * l_ineq
        violation = self.constraints.violation(x, t, y_pred, d_pred)
        return total, {
            "diff": l_diff,
            "eq": l_eq,
            "ineq": l_ineq,
            "violation": violation,
        }


class TaylorNeighborhoodApproximation(nn.Module):
    r"""Multi-point neighborhood approximation (DAE-HardNet §3, Eq. 9).

    Expresses ``y(x, t)`` as a weighted combination of its values at neighbor
    points minus the derivative corrections, converting differential operators
    into algebraic variables ``d`` for the KKT projection::

        y ~ (1/|X|) sum_i [ y([x,t] + Delta_i) - Delta * d_i - 0.5 * Delta^2 * d_ii ]

    where ``|X| = input_dim`` (one neighbor per independent variable). The
    approximation error vanishes as ``Delta -> 0`` (recommended ``Delta`` in
    ``[1e-3, 0.1]``; too small risks an ill-conditioned KKT Jacobian).
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int,
        delta: float = 0.01,
        order: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.delta = delta
        self.order = order

    def forward(self, inputs: Tensor, derivatives: Tensor) -> Tensor:
        """Return the neighborhood approximation ``[batch, output_dim]``.

        ``inputs``: ``[batch, input_dim]`` (the ``(x, t)`` point). ``derivatives``:
        first-order ``d_i`` in the first ``input_dim`` columns and (if
        ``order >= 2``) second-order ``d_ii`` in the next ``input_dim`` columns.
        """
        neighbor_terms = []
        for i in range(self.input_dim):
            delta_vec = torch.zeros_like(inputs)
            delta_vec[:, i] = self.delta
            y_neighbor = self.backbone(inputs + delta_vec)
            d_i = derivatives[:, i : i + 1]
            term = y_neighbor - self.delta * d_i.expand_as(y_neighbor)
            if self.order >= 2:
                d_ii = derivatives[:, self.input_dim + i : self.input_dim + i + 1]
                term = term - 0.5 * (self.delta**2) * d_ii.expand_as(y_neighbor)
            neighbor_terms.append(term)
        return torch.stack(neighbor_terms, dim=0).mean(dim=0)


class KKTProjectionLayer(nn.Module):
    r"""Differentiable KKT projection onto the DAE constraint manifold (DAE-HardNet §3.1).

    Solves the minimum-distance problem (Eq. 2)::

        y_tilde, d_tilde = argmin 0.5*||y - y_hat||^2
            s.t.  D(x,t,y,d) = 0,  h(x,t,y) = 0,  g(x,t,y) <= 0

    by Newton iteration on the KKT system (Eq. 13) with Fischer-Burmeister
    complementarity for the inequalities. The forward Newton solve runs detached
    for speed; gradients flow via a single implicit-function-theorem step at the
    solution (``z* - J^{-1} F(y_hat, z*)``), so ``d y_tilde / d y_hat`` is the
    correct implicit derivative without unrolling the iterations.

    The KKT Jacobian uses the Gauss-Newton form (constraint Jacobians treated as
    locally constant), exact for affine constraints and convergent for smooth
    nonlinear ones.

    Per-sample layout (length ``N``)::

        z = [ y(n_output), d(n_deriv), lam_eq(n_c), lam_ineq(n_g), s(n_g) ]
        F = [ stat_y(n_output), stat_d(n_deriv), feas_c(n_c), feas_g(n_g), fb(n_g) ]

    with ``n_c = n_differential + n_equality`` and ``n_g = n_inequality``.
    """

    def __init__(
        self,
        constraints: PhysicsConstraints,
        n_output: int,
        n_deriv: int,
        newton_step: float = 1.0,
        max_newton_iter: int = 10,
        newton_tol: float = 1e-6,
        reg: float = 1e-8,
        use_line_search: bool = True,
        line_search_max_halvings: int = 10,
    ) -> None:
        super().__init__()
        self.constraints = constraints
        spec = constraints.spec
        self.n_y = n_output
        self.n_d = n_deriv
        self.n_diff = spec.n_differential
        self.n_eq = spec.n_equality
        self.n_c = spec.n_differential + spec.n_equality
        self.n_g = spec.n_inequality
        self.newton_step = newton_step
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.reg = reg
        # Backtracking line search on the L-inf residual: makes the forward Newton
        # solve robust to overshoot on high-curvature constraints (the full step
        # is a descent direction for the merit, so halving finds a decreasing
        # step). Default on; ``use_line_search=False`` restores the old full step.
        self.use_line_search = use_line_search
        self.line_search_max_halvings = line_search_max_halvings
        self.oy = 0
        self.od = self.n_y
        self.oeq = self.n_y + self.n_d
        self.oineq = self.oeq + self.n_c
        self.os = self.oineq + self.n_g
        self.N = self.os + self.n_g
        # Convergence signal from the most recent forward (debt #2): callers /
        # tests can inspect these instead of the projection silently returning a
        # non-stationary iterate when Newton exhausts max_newton_iter.
        self.last_converged: bool = False
        self.last_residual: float = float("inf")

    def _eval_c(self, x_s: Tensor, t_s: Tensor, yd_s: Tensor) -> Tensor:
        """Per-sample constraint residual ``c = [D; h]`` -> ``[n_c]`` (pure fn).

        Inputs are single-sample vectors (no batch dim) -- the constraint
        callables are batch-first, so we add/strip a singleton batch dim. This
        is the function differentiated by ``jacrev`` (w.r.t. ``yd_s``) and mapped
        over the batch by ``vmap``; it has no in-place mutation or captured
        mutable state, so it is a valid ``torch.func`` transform target.
        """
        xb = x_s.unsqueeze(0)
        tb = t_s.unsqueeze(0)
        y = yd_s[: self.n_y].unsqueeze(0)
        d = yd_s[self.n_y : self.n_y + self.n_d].unsqueeze(0)
        parts: List[Tensor] = []
        if self.n_diff > 0:
            parts.append(self.constraints.differential(xb, tb, y, d))
        if self.n_eq > 0:
            parts.append(self.constraints.equality(xb, tb, y))
        c = torch.cat(parts, dim=-1) if parts else yd_s.new_zeros(1, 0)
        return c.squeeze(0)

    def _eval_g(self, x_s: Tensor, t_s: Tensor, y_s: Tensor) -> Tensor:
        """Per-sample inequality residual ``g`` -> ``[n_g]`` (pure fn; see ``_eval_c``)."""
        xb = x_s.unsqueeze(0)
        tb = t_s.unsqueeze(0)
        y = y_s.unsqueeze(0)
        g = self.constraints.inequality(xb, tb, y)
        return g.squeeze(0)

    def _constraints_and_jac(
        self, x: Tensor, t: Tensor, z: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return ``(c, g, Jc_y, Jc_d, Jg_y)`` -- all detached constants.

        The constraint Jacobians are built with ``torch.func`` (``jacrev`` +
        ``vmap``) rather than a per-constraint-row ``autograd.grad`` loop: for
        each sample ``jacrev`` returns the full ``[n_c, n_y + n_d]`` (resp.
        ``[n_g, n_y]``) Jacobian in one reverse-mode pass, and ``vmap`` maps it
        over the batch. The Gauss-Newton scheme treats these as constants, so the
        result is detached (numerically identical to the old loop).
        """
        zc = z.detach()
        b = z.shape[0]
        yd = zc[:, : self.n_y + self.n_d]
        y = zc[:, : self.n_y]
        d = zc[:, self.n_y : self.n_y + self.n_d]
        parts: List[Tensor] = []
        if self.n_diff > 0:
            parts.append(self.constraints.differential(x, t, y, d))
        if self.n_eq > 0:
            parts.append(self.constraints.equality(x, t, y))
        c = torch.cat(parts, dim=-1) if parts else z.new_zeros(b, 0)
        g = self.constraints.inequality(x, t, y) if self.n_g > 0 else z.new_zeros(b, 0)

        if self.n_c > 0:
            jc = vmap(jacrev(self._eval_c, argnums=2))(x, t, yd)  # [b, n_c, n_y + n_d]
            jc_y = jc[:, :, : self.n_y]
            jc_d = jc[:, :, self.n_y : self.n_y + self.n_d]
        else:
            jc_y = z.new_zeros(b, 0, self.n_y)
            jc_d = z.new_zeros(b, 0, self.n_d)
        if self.n_g > 0:
            jg = vmap(jacrev(self._eval_g, argnums=2))(x, t, y)  # [b, n_g, n_y]
        else:
            jg = z.new_zeros(b, 0, self.n_y)
        return c.detach(), g.detach(), jc_y.detach(), jc_d.detach(), jg.detach()

    def _assemble_F(
        self,
        y_hat: Tensor,
        z: Tensor,
        c: Tensor,
        g: Tensor,
        jc_y: Tensor,
        jc_d: Tensor,
        jg: Tensor,
    ) -> Tensor:
        y = z[:, : self.n_y]
        lam_eq = z[:, self.oeq : self.oeq + self.n_c]
        lam_ineq = z[:, self.oineq : self.oineq + self.n_g]
        s = z[:, self.os : self.os + self.n_g]

        stat_y = y - y_hat
        if self.n_c > 0:
            stat_y = stat_y + torch.bmm(jc_y.transpose(-1, -2), lam_eq.unsqueeze(-1)).squeeze(-1)
        if self.n_g > 0:
            stat_y = stat_y + torch.bmm(jg.transpose(-1, -2), lam_ineq.unsqueeze(-1)).squeeze(-1)
        blocks: List[Tensor] = [stat_y]
        if self.n_d > 0:
            if self.n_c > 0:
                stat_d = torch.bmm(jc_d.transpose(-1, -2), lam_eq.unsqueeze(-1)).squeeze(-1)
            else:
                stat_d = z.new_zeros(z.shape[0], self.n_d)
            blocks.append(stat_d)
        if self.n_c > 0:
            blocks.append(c)
        if self.n_g > 0:
            blocks.append(g + s)
            r = torch.sqrt(lam_ineq**2 + s**2 + 1e-12)
            blocks.append(r - lam_ineq - s)
        return torch.cat(blocks, dim=-1)

    def _assemble_J(self, z: Tensor, jc_y: Tensor, jc_d: Tensor, jg: Tensor) -> Tensor:
        b = z.shape[0]
        J = z.new_zeros(b, self.N, self.N)
        eye_y = torch.eye(self.n_y, device=z.device, dtype=z.dtype).expand(b, -1, -1)
        ry = slice(self.oy, self.oy + self.n_y)
        J[:, ry, self.oy : self.oy + self.n_y] = eye_y
        if self.n_c > 0:
            J[:, ry, self.oeq : self.oeq + self.n_c] = jc_y.transpose(-1, -2)
        if self.n_g > 0:
            J[:, ry, self.oineq : self.oineq + self.n_g] = jg.transpose(-1, -2)
        if self.n_d > 0 and self.n_c > 0:
            rd = slice(self.od, self.od + self.n_d)
            J[:, rd, self.oeq : self.oeq + self.n_c] = jc_d.transpose(-1, -2)
        if self.n_c > 0:
            rc = slice(self.oeq, self.oeq + self.n_c)
            J[:, rc, self.oy : self.oy + self.n_y] = jc_y
            if self.n_d > 0:
                J[:, rc, self.od : self.od + self.n_d] = jc_d
        if self.n_g > 0:
            rg = slice(self.oineq, self.oineq + self.n_g)
            J[:, rg, self.oy : self.oy + self.n_y] = jg
            eye_g = torch.eye(self.n_g, device=z.device, dtype=z.dtype).expand(b, -1, -1)
            J[:, rg, self.os : self.os + self.n_g] = eye_g
            lam_ineq = z[:, self.oineq : self.oineq + self.n_g]
            s = z[:, self.os : self.os + self.n_g]
            r = torch.sqrt(lam_ineq**2 + s**2 + 1e-12)
            rfb = slice(self.os, self.os + self.n_g)
            J[:, rfb, self.oineq : self.oineq + self.n_g] = torch.diag_embed(lam_ineq / r - 1.0)
            J[:, rfb, self.os : self.os + self.n_g] = torch.diag_embed(s / r - 1.0)
        return J

    def _init_z(
        self, x: Tensor, t: Tensor, y_hat: Tensor, d_hat: Tensor, lam_hat: Tensor
    ) -> Tensor:
        b = y_hat.shape[0]
        lam_eq = (
            lam_hat[:, : self.n_c] if lam_hat.shape[1] >= self.n_c else y_hat.new_zeros(b, self.n_c)
        )
        lam_ineq = (
            lam_hat[:, self.n_c : self.n_c + self.n_g]
            if self.n_g > 0 and lam_hat.shape[1] >= self.n_c + self.n_g
            else y_hat.new_zeros(b, self.n_g)
        )
        parts: List[Tensor] = [y_hat.detach()]
        if self.n_d > 0:
            parts.append(d_hat.detach())
        parts.append(lam_eq.detach())
        if self.n_g > 0:
            g0 = self.constraints.inequality(x, t, y_hat.detach())
            parts.append(lam_ineq.detach())
            parts.append(F.relu(-g0) + 0.1)
        return torch.cat(parts, dim=-1)

    def forward(
        self, x: Tensor, t: Tensor, y_hat: Tensor, d_hat: Tensor, lam_hat: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Project ``(y_hat, d_hat)`` onto the manifold -> ``(y_tilde, d_tilde, lam_tilde)``."""
        eye_N = torch.eye(self.N, device=y_hat.device, dtype=y_hat.dtype)
        z = self._init_z(x, t, y_hat, d_hat, lam_hat)
        yh = y_hat.detach()
        converged = False
        # Evaluate constraints + residual at the initial iterate, then reuse each
        # accepted iterate's constraints/Jacobian for the next step (so the happy
        # path costs ~1 constraints+Jacobian build per iteration, as before).
        c, g, jc_y, jc_d, jg = self._constraints_and_jac(x, t, z)
        Fv = self._assemble_F(yh, z, c, g, jc_y, jc_d, jg)
        res = float(Fv.abs().max())
        for _ in range(self.max_newton_iter):
            if res < self.newton_tol:
                converged = True
                break
            J = self._assemble_J(z, jc_y, jc_d, jg)
            delta = torch.linalg.solve(J + self.reg * eye_N, Fv.unsqueeze(-1)).squeeze(-1)
            if self.use_line_search:
                # Backtrack alpha until the L-inf residual strictly decreases (the
                # Newton direction is a descent direction for the merit while the
                # Gauss-Newton Jacobian is a good local model). This keeps the
                # iterate BOUNDED on stiff constraints where the full step would
                # diverge. If no halving decreases the residual, the local model
                # has stalled -- stop (the residual is then non-increasing across
                # the whole solve, and non-convergence is reported honestly).
                alpha = self.newton_step
                for _h in range(self.line_search_max_halvings + 1):
                    z_trial = z - alpha * delta
                    c_t, g_t, jcy_t, jcd_t, jg_t = self._constraints_and_jac(x, t, z_trial)
                    F_t = self._assemble_F(yh, z_trial, c_t, g_t, jcy_t, jcd_t, jg_t)
                    res_trial = float(F_t.abs().max())
                    if res_trial < res:
                        z = z_trial
                        c, g, jc_y, jc_d, jg = c_t, g_t, jcy_t, jcd_t, jg_t
                        Fv, res = F_t, res_trial
                        break
                    alpha *= 0.5
                else:
                    break  # no descent step found -> Gauss-Newton stalled
            else:
                z = z - self.newton_step * delta
                c, g, jc_y, jc_d, jg = self._constraints_and_jac(x, t, z)
                Fv = self._assemble_F(yh, z, c, g, jc_y, jc_d, jg)
                res = float(Fv.abs().max())
        # Final check: ``res`` now reflects the *returned* iterate (post-step), so
        # flag convergence if the last step reached tolerance (the loop only
        # re-checks at the top of the next iteration, which may not run).
        converged = converged or (res < self.newton_tol)
        self.last_residual = res
        self.last_converged = converged
        last_cj = (c, g, jc_y, jc_d, jg)
        if not converged:
            logger.warning(
                "KKT projection did not converge: residual=%.3e after %d Newton "
                "iterations (the implicit-function gradient is taken at a "
                "non-stationary point).",
                self.last_residual,
                self.max_newton_iter,
            )

        # Implicit-function one-step for differentiability w.r.t. y_hat. When the
        # Newton loop converged via the tolerance break, ``z`` is unchanged since
        # the last ``_constraints_and_jac`` call (which detaches internally), so
        # those values are exactly at ``z_star`` -- reuse them instead of
        # recomputing (output-identical; saves one constraints+Jacobian build).
        z_star = z.detach()
        if converged:
            # ``last_cj`` is at z_star (the loop keeps c/g/jac in sync with z),
            # so reuse it (output-identical; saves one constraints+Jacobian build).
            c, g, jc_y, jc_d, jg = last_cj
        else:
            c, g, jc_y, jc_d, jg = self._constraints_and_jac(x, t, z_star)
        J = self._assemble_J(z_star, jc_y, jc_d, jg).detach()
        F_live = self._assemble_F(y_hat, z_star, c, g, jc_y, jc_d, jg)
        z_out = z_star - torch.linalg.solve(J + self.reg * eye_N, F_live.unsqueeze(-1)).squeeze(-1)

        y_tilde = z_out[:, : self.n_y]
        d_tilde = (
            z_out[:, self.n_y : self.n_y + self.n_d]
            if self.n_d > 0
            else y_hat.new_zeros(y_hat.shape[0], 0)
        )
        lam_tilde = z_out[:, self.oeq : self.oeq + self.n_c + self.n_g]
        return y_tilde, d_tilde, lam_tilde


class PCMLModule(nn.Module):
    """Unified PCML wrapper managing the soft and hard modes (PCML Addendum §2.2).

    Soft mode (pre-training) returns the unconstrained prediction and the
    :class:`SoftPCMLLoss`. Once the backbone data loss drops below ``eta``
    (DAE-HardNet dynamic activation), :meth:`update_activation` flips the module
    into hard mode, where the prediction is projected onto the constraint
    manifold and the loss is ``MSE(y_tilde, y_true) + omega * MSE(d_tilde,
    d_hat)`` (DAE-HardNet Eq. 15; ``d_hat`` is the backbone's autodiff
    derivative -- the ``AD(d y_tilde)`` target).
    """

    def __init__(
        self,
        constraints: PhysicsConstraints,
        backbone: nn.Module,
        input_dim: int,
        n_output: int,
        n_deriv: int,
        n_lambda: int,
        lambda_soft: float = 1.0,
        omega: float = 1.0,
        delta: float = 0.01,
        taylor_order: int = 1,
        eta: float = 0.01,
        newton_step: float = 1.0,
        max_newton_iter: int = 10,
    ) -> None:
        super().__init__()
        self.constraints = constraints
        self.eta = eta
        self.omega = omega
        self._hard_mode_active = False
        self.soft_loss = SoftPCMLLoss(constraints, lambda_diff=lambda_soft)
        self.taylor_approx = TaylorNeighborhoodApproximation(
            backbone, input_dim, delta=delta, order=taylor_order
        )
        self.projection = KKTProjectionLayer(
            constraints,
            n_output,
            n_deriv,
            newton_step=newton_step,
            max_newton_iter=max_newton_iter,
        )

    def update_activation(self, current_data_loss: float) -> bool:
        """Activate hard mode when the data loss first drops below ``eta``.

        Returns ``True`` exactly on the call that flips the mode.
        """
        if not self._hard_mode_active and current_data_loss < self.eta:
            self._hard_mode_active = True
            return True
        return False

    @property
    def mode(self) -> str:
        return "hard" if self._hard_mode_active else "soft"

    @property
    def n_deriv(self) -> int:
        return self.projection.n_d

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y_hat: Tensor,
        d_hat: Tensor,
        lam_hat: Tensor,
        y_true: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Return ``(y_pcml, pcml_loss, info)`` for the active mode."""
        if self._hard_mode_active:
            y_tilde, d_tilde, _ = self.projection(x, t, y_hat, d_hat, lam_hat)
            if y_true is not None:
                l_data = F.mse_loss(y_tilde, y_true)
                l_deriv = F.mse_loss(d_tilde, d_hat) if self.n_deriv > 0 else y_hat.new_zeros(())
                loss = l_data + self.omega * l_deriv
            else:
                l_data = y_hat.new_zeros(())
                l_deriv = y_hat.new_zeros(())
                loss = y_hat.new_zeros(())
            violation = self.constraints.violation(x, t, y_tilde, d_tilde)
            info: Dict[str, Tensor] = {
                "mode": "hard",  # type: ignore[dict-item]
                "violation": violation,
                "data": l_data,
                "deriv": l_deriv,
            }
            return y_tilde, loss, info
        soft, parts = self.soft_loss(x, t, y_hat, d_hat)
        info = {"mode": "soft", **parts}  # type: ignore[dict-item]
        return y_hat, soft, info
