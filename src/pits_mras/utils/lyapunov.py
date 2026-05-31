r"""Lyapunov / Riccati engine (IP §4.3). "The mathematical engine for all P."

Owning phase: Phase 1 (Foundation Layer).

Identity 1 foundation. ARCHITECTURE.md §2.1 / §4.3 names six scipy-backed
functions: ``solve_lyapunov``, ``kleinman_iteration``, ``solve_care``,
``check_hurwitz``, ``lyapunov_derivative`` (:math:`\dot V = 2e^\top P(A_m e+Bu)`),
``quadratic_basis``. Built on ``scipy.linalg.solve_continuous_lyapunov`` /
``solve_continuous_are``. Owns ``tests/test_identity_lyapunov_value.py``.

Phase-1 sanity gate (IP §13): ``solve_lyapunov(-I, I)`` must return
``[[0.5, 0], [0, 0.5]]``.

TODO(phase-1): implement the six functions per docs/ARCHITECTURE.md §4.3.
"""
