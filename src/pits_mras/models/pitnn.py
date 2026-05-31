"""Top-level physics-informed time-series network (IP §5.4).

Owning phase: Phase 2 (Neural Network Models).

ARCHITECTURE.md §2.1 names ``PITNN``: embedding -> causal (forward-only) LSTM ->
``PhysicsInformedAttention`` -> ``PortHamiltonianDecoder`` (Algorithm 1). Exported
as a top-level symbol in the package ``__init__`` per the design doc.

TODO(phase-2): implement per docs/ARCHITECTURE.md §5.4.
"""


class PITNN:
    """Physics-informed time-series network (Algorithm 1).

    Named in ARCHITECTURE.md §5.4. The forward-pass signature lives in the
    technical spec (Gap G4) and is not fabricated here.
    """

    def __init__(self) -> None:
        raise NotImplementedError("phase 2")
