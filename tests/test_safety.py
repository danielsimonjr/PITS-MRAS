"""Safety test (IP §11.3): CLF-CBF-QP filter / forward invariance (Identity 3).

Targets ``pits_mras.controllers.safety`` (``CLFCBFSafetyFilter``).
Owning phase: Phase 4 per ROADMAP.md (authored alongside its target phase;
§11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.3):
``test_cbf_projects_unsafe_control``, ``test_cbf_identity_when_safe``,
``test_cbf_forward_invariance`` (100-step sim stays in the safe set).
Placeholders skipped until Phase 4 implements the safety filter.
"""

import pytest


@pytest.mark.skip(reason="phase 4 not implemented")
def test_cbf_projects_unsafe_control() -> None:
    """An unsafe nominal control is projected onto the safe half-space."""


@pytest.mark.skip(reason="phase 4 not implemented")
def test_cbf_identity_when_safe() -> None:
    """A safe nominal control passes through the filter unchanged."""


@pytest.mark.skip(reason="phase 4 not implemented")
def test_cbf_forward_invariance() -> None:
    """A 100-step closed-loop sim stays inside the safe error ellipsoid."""
