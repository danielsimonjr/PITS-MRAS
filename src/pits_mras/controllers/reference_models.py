r"""Linear reference model (IP §7.1).

Owning phase: Phase 4 (Controllers).

ARCHITECTURE.md §2.1 / §5.2 names ``LinearReferenceModel``:
:math:`\dot x_m = A_m x_m + B_m r`, :math:`y_m = C_m x_m`, Hurwitz :math:`A_m`
verified at construction; on init it solves the Lyapunov equation for ``P`` and
runs ``kleinman_iteration`` -> ``(P_opt, K_opt)`` (linking the reference model to
the value function, Identity 1). Euler ``step``.

TODO(phase-4): implement per docs/ARCHITECTURE.md §7.1.
"""


class LinearReferenceModel:
    r"""Hurwitz linear reference model :math:`\dot x_m = A_m x_m + B_m r`.

    Named in ARCHITECTURE.md §7.1. Buffers ``P, P_opt, K_opt, R_inv``. Concrete
    constructor signature is not reproduced in the available sources.
    """

    def __init__(self) -> None:
        raise NotImplementedError("phase 4")
