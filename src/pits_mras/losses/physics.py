r"""Physics-informed loss (IP §6.1).

Owning phase: Phase 3 (Loss Functions).

ARCHITECTURE.md §2.1 names ``PhysicsLoss``:
:math:`\lambda_1 L_{energy}+\lambda_2 L_{PDE}+\lambda_3 L_{BC}+\lambda_4 L_{sym}`
(ported from the technical spec §2.2). The exact PDE/BC/symmetry operators live
only in the 1,543-line technical spec (Gap G4) and are not reproduced here.

TODO(phase-3): implement per docs/ARCHITECTURE.md §6.1.
"""
