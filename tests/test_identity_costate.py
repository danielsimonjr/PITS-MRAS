"""Identity test (IP §11.2): Costate = Critic Gradient (Identity 2).

Targets ``pits_mras.models.critic`` (``CostateHead``).
Owning phase: Phase 2 per ROADMAP.md (authored alongside its target phase;
§11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.2):
``test_costate_equals_grad_V``, ``test_optimal_control_equals_lqr_gain``.
Placeholders skipped until Phase 2 implements the critic / costate head.
"""

import pytest


@pytest.mark.skip(reason="phase 2 not implemented")
def test_costate_equals_grad_V() -> None:
    """I2: the costate equals the autodiff gradient of the value head."""


@pytest.mark.skip(reason="phase 2 not implemented")
def test_optimal_control_equals_lqr_gain() -> None:
    """u* = -R^-1 B^T grad V recovers the LQR gain in the linear limit."""
