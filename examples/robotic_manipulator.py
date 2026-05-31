r"""Example: 2-DOF planar robotic manipulator (IP §10.1).

Owning phase: Phase 7 (Examples).

ARCHITECTURE.md §2.1 / ROADMAP §Phase 7: 2-DOF planar manipulator,
:math:`H=\tfrac12\dot q^\top M(q)\dot q+V(q)`, sinusoidal joint-angle reference.
Diagnostic plots: (a) :math:`\|e(t)\|`, (b) :math:`\hat V(e(t))`, (c) CBF
activation flag, (d) critic-convergence
:math:`\|\hat P-P_{CARE}\|_F/\|P_{CARE}\|_F`. This is the Phase-6 acceptance
gate (100-step run generates plots without error).

TODO(phase-7): implement per docs/ARCHITECTURE.md §10.1.
"""
