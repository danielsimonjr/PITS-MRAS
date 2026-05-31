"""Losses subpackage: physics, temporal, stability, IRL, HJB + TotalLoss.

Owning phase: Phase 3 (Loss Functions) per ROADMAP.md / ARCHITECTURE.md §2.1.

ARCHITECTURE.md §2 also assigns the ``TotalLoss`` aggregator (with per-sub-loss
TensorBoard/wandb logging under ``loss/physics``, ``loss/temporal``,
``loss/stability``, ``loss/irl``, ``loss/hjb``, ``loss/costate``, ``loss/data``)
to this package ``__init__`` (IP §6.6). It is left as a documented placeholder --
the design doc otherwise mandates docstring-only subpackage inits with no
imports; ``TotalLoss`` arrives in Phase 3.

TODO(phase-3): implement the ``TotalLoss`` aggregator per docs/ARCHITECTURE.md
§6.6.
"""
