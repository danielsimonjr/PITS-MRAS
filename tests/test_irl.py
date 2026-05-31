"""IRL test (IP §11.5): Integral-RL Bellman loss (Identity 1).

Targets ``pits_mras.losses.irl`` (``IRLBellmanAccumulator``, ``IRLBellmanLoss``).
Owning phase: Phase 3 per ROADMAP.md (authored alongside its target phase;
§11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.5):
``test_irl_bellman_error_zero_at_true_value``,
``test_irl_loss_decreases_with_correct_update``. Placeholders skipped until
Phase 3 implements the IRL loss.
"""

import pytest


@pytest.mark.skip(reason="phase 3 not implemented")
def test_irl_bellman_error_zero_at_true_value() -> None:
    """The IRL Bellman residual is ~0 when the critic equals the true value."""


@pytest.mark.skip(reason="phase 3 not implemented")
def test_irl_loss_decreases_with_correct_update() -> None:
    """L_IRL decreases under a correct critic update step."""
