"""PITS-MRAS: Physics-Informed Time-Series Model-Reference Adaptive Systems.

Top-level package. The canonical module layout is defined in
``docs/ARCHITECTURE.md`` §2; phase ordering for filling in the stubs is in
``ROADMAP.md``.

Owning phase: Phase 1 (Foundation Layer) per ROADMAP.md (the plan calls the
package-creation phase "Phase 1"; this scaffold task is its first step).

NOTE on version: ``setup.py`` declares ``version="1.0.0"`` and the scaffold task
mandates matching it, so ``__version__`` is "1.0.0" here. The design docs
(ARCHITECTURE.md §2 / ROADMAP §Phase 1, citing IP §4.1) instead specify
``__version__ = "0.1.0"`` together with a top-level re-export block importing
``PITNN``, ``QuadraticCritic``, ``MRASController``, ``LinearReferenceModel``,
``CLFCBFSafetyFilter``, ``pretrain_pitnn``, ``cotraining_loop``, and
``RealtimeInferenceEngine``. That version mismatch (Gap G2 in ARCHITECTURE.md
§8.3) is an ADR-level decision deferred to the user, and the eager re-export
block is intentionally NOT reproduced yet because those symbols are unimplemented
stubs -- importing them here would break the import smoke test. Both are wired up
once Phases 2-6 land.
"""

__version__ = "1.0.0"

# Public API is intentionally empty during scaffolding (ARCHITECTURE.md §5 keeps
# it minimal "until APIs stabilize"). Populated as later phases land.
# TODO(phase-1): re-export the eight top-level symbols and reconcile __version__
# with the design docs (Gap G2) once the modules are implemented.
__all__: list[str] = []
