r"""Integral-RL Bellman loss (IP §6.4). NEW -- Identity 1.

Owning phase: Phase 3 (Loss Functions).

Identity 1 (Lyapunov = Value Function). ARCHITECTURE.md §2.1 / §4.1 names
``IRLBellmanAccumulator`` and ``IRLBellmanLoss``:
:math:`\delta_{IRL}(t)=\int_{t-T}^{t} r\,d\tau - [\hat V(e(t))-\hat V(e(t-T))]`,
:math:`L_{IRL}=\tfrac12\mathbb E[\delta_{IRL}^2]` -- model-free (no drift matrix
A). Owns ``tests/test_irl.py``.

TODO(phase-3): implement per docs/ARCHITECTURE.md §6.4.
"""
