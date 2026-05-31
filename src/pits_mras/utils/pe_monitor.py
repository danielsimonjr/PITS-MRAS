"""Persistence-of-excitation monitor (IP §4.5).

Owning phase: Phase 1 (Foundation Layer).

ARCHITECTURE.md §2.1 / §4.5 names ``PEMonitor`` -- tracks the min eigenvalue of
the regressor Gram matrix over a sliding window and injects probing noise if PE is
violated (required because IRL/ADP parameter convergence needs persistence of
excitation of the regressor). Caveat: probing noise biases estimates unless
handled.

TODO(phase-1): implement ``PEMonitor`` per docs/ARCHITECTURE.md §4.5.
"""


class PEMonitor:
    """Persistence-of-excitation monitor.

    Named in ARCHITECTURE.md §4.5. Concrete constructor / method signatures are
    not reproduced in the available sources.
    """

    def __init__(self) -> None:
        raise NotImplementedError("phase 1")
