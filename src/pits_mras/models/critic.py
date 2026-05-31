r"""Critic / value network and costate head (IP §5.3). NEW -- Identity 1 & 2.

Owning phase: Phase 2 (Neural Network Models).

Identity 1 (Lyapunov = Value Function) and Identity 2 (Costate = Critic
Gradient). ARCHITECTURE.md §2.1 / §4.2 explicitly name two classes here:

- ``QuadraticCritic`` -- :math:`\hat V(e)=W_c^\top\phi_c(e)`; methods named in the
  source: ``forward``, ``gradient`` (= costate), ``extract_P``,
  ``positivity_loss``; optional ``nonlinear_residual`` MLP (Connection 10).
- ``CostateHead`` -- :math:`\hat\lambda=\nabla\hat V`,
  :math:`u^*=-R^{-1}B^\top\hat\lambda`; enforces Identity 2 by construction.

These names are taken verbatim from the design doc; bodies raise
``NotImplementedError`` until Phase 2.

TODO(phase-2): implement per docs/ARCHITECTURE.md §5.3.
"""


class QuadraticCritic:
    r"""Quadratic value-function critic :math:`\hat V(e)=e^\top \hat P e`.

    Named in ARCHITECTURE.md §5.3 (Identity 1). Concrete signatures for the
    planned ``forward`` / ``gradient`` / ``extract_P`` / ``positivity_loss``
    methods are not reproduced in the available sources, so they are not
    fabricated here.
    """

    def __init__(self) -> None:
        raise NotImplementedError("phase 2")


class CostateHead:
    r"""Costate head :math:`\hat\lambda=\nabla\hat V`, :math:`u^*=-R^{-1}B^\top\hat\lambda`.

    Named in ARCHITECTURE.md §5.3 (Identity 2). The action head IS the gradient
    of the critic by construction.
    """

    def __init__(self) -> None:
        raise NotImplementedError("phase 2")
