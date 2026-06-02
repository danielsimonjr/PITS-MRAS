"""Lagrangian-multiplier head for the KKT projection (PCML Addendum §2.3).

DAE-HardNet's backbone outputs ``Y_hat = [y_hat, lambda_hat, d_hat]``: the soft
prediction, the predicted Lagrangian multipliers (warm-start for the Newton
projection), and the autodiff derivatives. This head produces ``lambda_hat``
from the PITNN attention context ``c_t``.

Equality/differential multipliers are unconstrained (any sign); inequality
multipliers must be non-negative (KKT dual feasibility), enforced via Softplus.
"""

import torch
import torch.nn as nn
from torch import Tensor


class LagrangianMultiplierHead(nn.Module):
    """Predict the KKT warm-start multipliers ``lambda_hat`` from a context vector.

    Args:
        context_dim: dimension of the input context ``c_t``.
        n_lambda_eq: number of equality + differential multipliers (any sign).
        n_lambda_ineq: number of inequality multipliers (non-negative).
        hidden_dim: hidden width of the 2-layer MLP.
    """

    def __init__(
        self,
        context_dim: int,
        n_lambda_eq: int,
        n_lambda_ineq: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_lambda_eq = n_lambda_eq
        self.n_lambda_ineq = n_lambda_ineq
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_lambda_eq + n_lambda_ineq),
        )

    def forward(self, context: Tensor) -> Tensor:
        """Return ``lambda_hat`` ``[batch, n_lambda_eq + n_lambda_ineq]``.

        Equality multipliers are free; inequality multipliers are passed through
        ``Softplus`` so they are non-negative.
        """
        raw = self.net(context)
        if self.n_lambda_ineq > 0:
            lam_eq = raw[:, : self.n_lambda_eq]
            lam_ineq = nn.functional.softplus(raw[:, self.n_lambda_eq :])
            return torch.cat([lam_eq, lam_ineq], dim=-1)
        return raw
