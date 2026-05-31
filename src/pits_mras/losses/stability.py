"""MRAS stability losses (IP §6.3).

Owning phase: Phase 3 (Loss Functions).

ARCHITECTURE.md §2.1 names ``LyapunovConstraintLoss``,
``ParameterBoundednessLoss``, ``ControlEffortLoss``, and the aggregating
``MRASStabilityLoss``; imports ``utils.lyapunov.lyapunov_derivative``. Concrete
formulas live in the technical spec (Gap G4); left as a documented placeholder.

TODO(phase-3): implement per docs/ARCHITECTURE.md §6.3.
"""
