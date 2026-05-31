r"""HJB residual loss (IP §6.5). NEW -- Identity 8.

Owning phase: Phase 3 (Loss Functions).

Identity 8 (HJB Residual Loss). ARCHITECTURE.md §2.1 / §4.1 names
``HJBResidualLoss``
(:math:`\|e^\top Q e+(u^*)^\top R u^*+\nabla_e\hat V\cdot(A_m e+B u^*+f_{corr})\|^2`)
and ``LyapunovDecreaseEnforcer`` (:math:`L_{dec}=\mathbb E[\mathrm{ReLU}(\nabla
\hat V\cdot\hat f+\ell)]`). Start weight :math:`\lambda_{HJB}=0.01` (regularizer).

TODO(phase-3): implement per docs/ARCHITECTURE.md §6.5.
"""
