r"""CLF-CBF-QP safety filter (IP §7.2 / §3.4). NEW -- Identity 3.

Owning phase: Phase 4 (Controllers).

Implements the closed-form single-constraint CBF projection from §3.4 for the
affine tracking-error system :math:`\dot e = f(e) + g(e) u` with
:math:`f(e) = A_m e`, :math:`g(e) = B`:

- CBF:        :math:`h(e) = c - e^\top P e`  (safe set ``e^T P e <= c``).
- Lie derivs: :math:`L_f h = -2 e^\top P A_m e`, :math:`L_g h = -2 e^\top P B`.
- Constraint: :math:`L_f h + L_g h\, u \ge -\gamma h(e)`.
- Safety index ``a = L_f h + L_g h . u_nom + gamma h(e)`` (>= 0 means safe).

Closed-form minimum-norm projection (single constraint, no QP solver needed):

    u_safe = u_nom + (relu(-a) / ||L_g h||^2) * L_g h        (§7.2 form)
           = u_nom - (a / ||L_g h||^2) * L_g h   when a < 0   (§3.4 form)

These are identical: when ``a < 0``, ``relu(-a) = -a``, so both add
``(-a / ||L_g h||^2) * L_g h``; when ``a >= 0`` the correction is zero (the
nominal control is kept). After the correction the new safety index is
``a + relu(-a) >= 0``, which guarantees forward invariance of the safe set.

Reference: Ames et al. (2017), IEEE TAC. arXiv:1609.06408.
"""

from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CLFCBFSafetyFilter(nn.Module):
    r"""Closed-form CLF-CBF safety filter (Identity 3).

    The same ``P`` used for the Lyapunov CLF (:math:`V = e^\top P e`) serves as
    the CBF (:math:`h = c - e^\top P e`): one ``P`` simultaneously certifies
    stability (CLF) and safety (forward invariance of the error ellipsoid).
    """

    P: Tensor
    A_m: Tensor
    B_ctrl: Tensor

    def __init__(
        self,
        P: Tensor,
        A_m: Tensor,
        B_ctrl: Tensor,
        safety_margin: float = 10.0,
        decay_rate: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("P", P)  # [n, n] Lyapunov/CBF matrix
        self.register_buffer("A_m", A_m)  # [n, n] reference dynamics
        self.register_buffer("B_ctrl", B_ctrl)  # [n, m] control input matrix
        self.safety_margin = safety_margin  # c: ellipsoid size
        self.decay_rate = decay_rate  # gamma: class-K rate

    def forward(self, e: Tensor, u_nom: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Apply the CBF safety filter.

        Args:
            e: ``[batch, n]`` tracking error.
            u_nom: ``[batch, m]`` nominal control.

        Returns:
            u_safe: ``[batch, m]`` safety-filtered control.
            h_e: ``[batch]`` CBF value :math:`h(e) = c - e^\top P e` (>0 = safe).
            slack: ``[batch]`` correction magnitude (0 = filter inactive).
        """
        # CBF value h(e) = c - e^T P e.
        ePe = (e @ self.P * e).sum(dim=-1)  # [batch]
        h_e = self.safety_margin - ePe  # [batch]

        # Lie derivatives.
        Pe = e @ self.P  # [batch, n]
        L_f_h = -2.0 * (Pe * (e @ self.A_m.T)).sum(dim=-1)  # [batch]
        L_g_h = -2.0 * (Pe @ self.B_ctrl)  # [batch, m]

        # Safety index a = L_f h + L_g h . u_nom + gamma h(e).
        L_g_h_u_nom = (L_g_h * u_nom).sum(dim=-1)  # [batch]
        a = L_f_h + L_g_h_u_nom + self.decay_rate * h_e  # [batch]

        # Closed-form minimum-norm correction (active only where a < 0).
        L_g_h_sq = (L_g_h * L_g_h).sum(dim=-1) + 1e-8  # [batch], avoid div/0
        correction_scale = F.relu(-a) / L_g_h_sq  # [batch], = 0 when safe
        correction = correction_scale.unsqueeze(-1) * L_g_h  # [batch, m]
        u_safe = u_nom + correction  # [batch, m]
        slack = correction.norm(dim=-1)  # [batch]
        return u_safe, h_e, slack

    def cbf_constraint_loss(self, e: Tensor, u: Tensor) -> Tensor:
        r"""Soft CBF-constraint loss for training (penalize :math:`h(e) < 0`).

        Returns a scalar :math:`\mathrm{mean}(\mathrm{relu}(-h(e)))` that can be
        added to the total training loss.
        """
        _, h_e, _ = self.forward(e, u)
        return F.relu(-h_e).mean()
