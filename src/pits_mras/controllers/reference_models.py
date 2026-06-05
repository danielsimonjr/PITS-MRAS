r"""Linear reference model (IP §7.1).

Owning phase: Phase 4 (Controllers).

ARCHITECTURE.md §2.1 / §7.1 names ``LinearReferenceModel``:
:math:`\dot x_m = A_m x_m + B_m r`, :math:`y_m = C_m x_m`, Hurwitz :math:`A_m`
verified at construction; on init it solves the Lyapunov equation for ``P`` and
runs ``kleinman_iteration`` -> ``(P_opt, K_opt)`` (linking the reference model to
the value function, Identity 1). Euler ``step``.

The reference model defines the desired closed-loop behavior; the same ``A_m``
feeds the MRAS Lyapunov equation :math:`A_m^\top P + P A_m = -Q`, linking the
reference model directly to the value function (Identity 1).
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from pits_mras.utils.lyapunov import (
    check_hurwitz,
    kleinman_iteration,
    solve_lyapunov,
)


class LinearReferenceModel(nn.Module):
    r"""Hurwitz linear reference model :math:`\dot x_m = A_m x_m + B_m r`.

    On construction it (a) asserts ``A_m`` is Hurwitz, (b) solves the Lyapunov
    equation :math:`A_m^\top P + P A_m = -Q` for the CLF/value matrix ``P``, and
    (c) runs Kleinman policy iteration to obtain the LQR optimum
    ``(P_opt, K_opt)``. All matrices are stored as float32 buffers.

    Args:
        A_m: ``[n, n]`` reference dynamics (must be Hurwitz).
        B_m: ``[n, m]`` reference control input matrix.
        C_m: ``[p, n]`` reference output matrix.
        Q: ``[n, n]`` state-cost / Lyapunov RHS (positive definite).
        R: ``[m, m]`` control-cost (positive definite).
    """

    A_m: Tensor
    B_m: Tensor
    C_m: Tensor
    Q: Tensor
    R: Tensor
    R_inv: Tensor
    P: Tensor
    P_opt: Tensor
    K_opt: Tensor

    def __init__(
        self,
        A_m: np.ndarray,
        B_m: np.ndarray,
        C_m: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        super().__init__()
        A_m = np.asarray(A_m, dtype=np.float64)
        B_m = np.asarray(B_m, dtype=np.float64)
        C_m = np.asarray(C_m, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        if not check_hurwitz(A_m):
            raise ValueError(
                "A_m must be Hurwitz (all eigenvalues have strictly negative " "real parts)."
            )

        # Policy-evaluation P (Identity 1): A_m^T P + P A_m = -Q.
        P = solve_lyapunov(A_m, Q)
        # Policy-iteration LQR optimum (P_opt, K_opt).
        P_opt, K_opt = kleinman_iteration(A_m, B_m, Q, R)

        self.register_buffer("A_m", torch.tensor(A_m, dtype=torch.float32))
        self.register_buffer("B_m", torch.tensor(B_m, dtype=torch.float32))
        self.register_buffer("C_m", torch.tensor(C_m, dtype=torch.float32))
        self.register_buffer("Q", torch.tensor(Q, dtype=torch.float32))
        self.register_buffer("R", torch.tensor(R, dtype=torch.float32))
        self.register_buffer("R_inv", torch.tensor(np.linalg.inv(R), dtype=torch.float32))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))
        self.register_buffer("P_opt", torch.tensor(P_opt, dtype=torch.float32))
        self.register_buffer("K_opt", torch.tensor(K_opt, dtype=torch.float32))

    def reset(self, batch: int = 1) -> Tensor:
        r"""Return a zero reference state :math:`x_m = 0` of shape ``[batch, n]``."""
        n = self.A_m.shape[0]
        return torch.zeros(batch, n, dtype=self.A_m.dtype, device=self.A_m.device)

    def step(self, x_m: Tensor, r: Tensor, dt: float) -> Tensor:
        r"""Forward-Euler integration of :math:`\dot x_m = A_m x_m + B_m r`.

        Shapes: ``x_m`` ``[batch, n]``, ``r`` ``[batch, m]`` -> ``[batch, n]``.
        """
        dx_m = x_m @ self.A_m.T + r @ self.B_m.T
        return x_m + dx_m * dt

    def output(self, x_m: Tensor) -> Tensor:
        r"""Reference output :math:`y_m = C_m x_m`, shape ``[batch, p]``."""
        return x_m @ self.C_m.T
