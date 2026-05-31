r"""CLF-CBF-QP safety filter (IP §7.2). NEW -- Identity 3.

Owning phase: Phase 4 (Controllers).

Identity 3 (CLF-CBF-QP Safety Filter). ARCHITECTURE.md §2.1 / §7.2 names
``CLFCBFSafetyFilter`` -- closed-form single-constraint CBF projection with
:math:`h(e)=c-e^\top P e`, :math:`L_f h=-2e^\top P A_m e`,
:math:`L_g h=-2e^\top P B`; one ``P`` serves both CLF and CBF. Also provides a
``cbf_constraint_loss`` soft penalty. Owns ``tests/test_safety.py``.

TODO(phase-4): implement per docs/ARCHITECTURE.md §7.2.
"""


class CLFCBFSafetyFilter:
    r"""Closed-form CLF-CBF safety filter (Identity 3).

    Named in ARCHITECTURE.md §7.2. Projects the nominal control onto the safe
    half-space :math:`L_f h + L_g h\,u \ge -\gamma h(e)`. Concrete signature is
    not reproduced in the available sources.
    """

    def __init__(self) -> None:
        raise NotImplementedError("phase 4")
