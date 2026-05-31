r"""Critic / value network and costate head (IP §5.3). NEW -- Identity 1 & 2.

Owning phase: Phase 2 (Neural Network Models).

- ``QuadraticCritic`` -- :math:`\hat V(e) = W_c^\top \phi_c(e)` where
  :math:`\phi_c` is the upper-triangular Kronecker basis. This guarantees
  :math:`\hat V(e) = e^\top \hat P e` with :math:`\hat P` symmetric by
  construction, so the LQR limit (:math:`\hat P \to P_{CARE}`) is exactly
  representable and the gradient :math:`\nabla_e \hat V = 2\hat P e` is analytic
  (Identity 1). An optional MLP residual extends it to the nonlinear regime.
- ``CostateHead`` -- :math:`\hat\lambda = \nabla\hat V`,
  :math:`u^* = -R^{-1}B^\top\hat\lambda`. The action head IS the autodiff
  gradient of the critic, enforcing Identity 2 (PMP costate = critic gradient)
  by construction.
"""

import torch
import torch.nn as nn
from torch import Tensor

from pits_mras.utils.lyapunov import quadratic_basis


class QuadraticCritic(nn.Module):
    r"""Linear-in-parameters quadratic value-function approximator.

    :math:`\hat V(e) = W_c^\top \phi(e)` with :math:`\phi` the upper-triangular
    basis ``[e1^2, e1 e2, ..., e2^2, ...]`` (see ``utils.lyapunov.quadratic_basis``).
    ``W_c`` is initialized so that :math:`\hat P \approx I` (diagonal weights
    set to 1). If ``nonlinear_residual`` is set, a small MLP ``deltaV(e)`` is
    added for the nonlinear regime.
    """

    def __init__(
        self,
        state_dim: int,
        nonlinear_residual: bool = False,
        residual_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.basis_dim = state_dim * (state_dim + 1) // 2
        # Linear-in-parameters layer (no bias -- V(0) = 0 is required for a CLF).
        self.W_c = nn.Linear(self.basis_dim, 1, bias=False)
        # Initialize near P ~ I: set the e_i^2 coefficients to 1, rest to 0.
        with torch.no_grad():
            self.W_c.weight.data.zero_()
            idx = 0
            for i in range(state_dim):
                for j in range(i, state_dim):
                    if i == j:
                        self.W_c.weight.data[0, idx] = 1.0
                    idx += 1
        self.use_residual = nonlinear_residual
        if nonlinear_residual:
            self.residual_net = nn.Sequential(
                nn.Linear(state_dim, residual_hidden),
                nn.Tanh(),
                nn.Linear(residual_hidden, residual_hidden),
                nn.Tanh(),
                nn.Linear(residual_hidden, 1),
            )

    def forward(self, e: Tensor) -> Tensor:
        """Return :math:`\\hat V(e)` of shape ``[batch]``.

        ``e`` must have grad enabled if the costate (``gradient``) will be taken.
        """
        phi = quadratic_basis(e)  # [batch, basis_dim]
        V = self.W_c(phi).squeeze(-1)  # [batch]
        if self.use_residual:
            V = V + self.residual_net(e).squeeze(-1)
        return V

    def gradient(self, e: Tensor) -> Tensor:
        r"""Compute :math:`\nabla_e \hat V` via autograd -- this IS the costate.

        Returns shape ``[batch, state_dim]``. For the pure quadratic critic this
        equals :math:`2\hat P e`.
        """
        e = e.requires_grad_(True)
        V = self.forward(e)
        grad = torch.autograd.grad(V.sum(), e, create_graph=True)[0]
        return grad  # [batch, state_dim]

    def extract_P(self) -> Tensor:
        r"""Reconstruct the symmetric :math:`\hat P` from the ``W_c`` weights.

        Returns shape ``[state_dim, state_dim]``. Diagonal entries get the
        ``e_i^2`` coefficient; off-diagonals split the ``e_i e_j`` coefficient
        symmetrically so that :math:`e^\top \hat P e = W_c^\top \phi(e)`.
        Useful for monitoring IRL convergence to the CARE solution.
        """
        n = self.state_dim
        P = torch.zeros(
            n, n, device=self.W_c.weight.device, dtype=self.W_c.weight.dtype
        )
        w = self.W_c.weight.detach().squeeze(0)  # [basis_dim]
        idx = 0
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    P[i, j] = w[idx]
                else:
                    P[i, j] = w[idx] / 2.0
                    P[j, i] = w[idx] / 2.0
                idx += 1
        return P

    def set_P(self, P: Tensor) -> None:
        r"""Write a symmetric :math:`\hat P` into the ``W_c`` basis weights.

        Inverse of :meth:`extract_P`: given the symmetric matrix ``P`` (so that
        :math:`\hat V(e) = e^\top P e`), set the upper-triangular basis weights
        ``W_c`` such that the diagonal coefficient of :math:`e_i^2` is
        ``P[i, i]`` and the coefficient of the cross term :math:`e_i e_j`
        (``i < j``) is ``P[i, j] + P[j, i]``. Used to warm-start the critic to
        the LQR / CARE solution so the costate matches the optimum.

        Args:
            P: ``[state_dim, state_dim]`` matrix; symmetrized internally.
        """
        n = self.state_dim
        if P.shape != (n, n):
            raise ValueError(
                f"set_P expects a [{n}, {n}] matrix, got {tuple(P.shape)}."
            )
        P = P.to(device=self.W_c.weight.device, dtype=self.W_c.weight.dtype)
        with torch.no_grad():
            w = torch.zeros(self.basis_dim, device=P.device, dtype=P.dtype)
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        w[idx] = P[i, i]
                    else:
                        # extract_P splits this weight in half across (i,j),(j,i);
                        # the inverse recombines both off-diagonal entries.
                        w[idx] = P[i, j] + P[j, i]
                    idx += 1
            self.W_c.weight.data.copy_(w.unsqueeze(0))

    def positivity_loss(self) -> Tensor:
        r"""Penalize if :math:`\hat P` is not positive definite.

        Loss :math:`= \mathrm{ReLU}(-\lambda_{\min}(\hat P))` (scalar).
        """
        P = self.extract_P()
        eigvals = torch.linalg.eigvalsh(P)
        return torch.relu(-eigvals.min())


class CostateHead(nn.Module):
    r"""Costate / optimal-control head enforcing Identity 2 by construction.

    Implements :math:`\hat\lambda(t) = \partial\hat V/\partial e` and
    :math:`u^* = -R^{-1}B^\top\hat\lambda`. The action head is never an
    independent network -- it IS the gradient of the critic.
    """

    R_inv: Tensor
    B_mat: Tensor

    def __init__(self, critic: QuadraticCritic, R_inv: Tensor, B: Tensor) -> None:
        super().__init__()
        self.critic = critic
        self.register_buffer("R_inv", R_inv)  # [control_dim, control_dim]
        self.register_buffer("B_mat", B)  # [state_dim, control_dim]

    def forward(self, e: Tensor) -> tuple[Tensor, Tensor]:
        r"""Return ``(lambda_hat, u_optimal)``.

        lambda_hat: ``[batch, state_dim]`` -- the costate :math:`\nabla\hat V`.
        u_optimal: ``[batch, control_dim]`` -- :math:`u^* = -R^{-1}B^\top\hat\lambda`.
        """
        lambda_hat = self.critic.gradient(e)  # [batch, state_dim]
        u_opt = -(lambda_hat @ self.B_mat) @ self.R_inv.T  # [batch, control_dim]
        return lambda_hat, u_opt
