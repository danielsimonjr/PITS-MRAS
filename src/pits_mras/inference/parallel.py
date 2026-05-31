"""Parallel thread architecture for deployment (IP §9.2).

Owning phase: Phase 6 (Inference Engine).

ARCHITECTURE.md §2.1 / §6.4 names three threads: ``ControlThread`` (1 kHz, calls
``engine.step()``), ``AdaptationThread`` (100 Hz, IRL critic update + policy
improvement via ``copy.deepcopy`` + atomic double-buffer swap), ``MonitorThread``
(10 Hz). Graceful shutdown via ``threading.Event``. Concrete signatures are not
reproduced in the available sources.

TODO(phase-6): implement per docs/ARCHITECTURE.md §9.2.
"""
