"""Models test (IP §11.4): port-Hamiltonian structure.

Targets ``pits_mras.models.decoders`` (and ``models.pitnn``).
Owning phase: Phase 2 per ROADMAP.md (authored alongside its target phase;
§11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.4):
``test_dissipation_matrix_psd``, ``test_J_skew_symmetric``,
``test_hamiltonian_positive``. Placeholders skipped until Phase 2 implements the
model backbone.
"""

import pytest


@pytest.mark.skip(reason="phase 2 not implemented")
def test_dissipation_matrix_psd() -> None:
    """R_theta = L^T L is positive semidefinite."""


@pytest.mark.skip(reason="phase 2 not implemented")
def test_J_skew_symmetric() -> None:
    """The interconnection matrix J is skew-symmetric (J = -J^T)."""


@pytest.mark.skip(reason="phase 2 not implemented")
def test_hamiltonian_positive() -> None:
    """The learned Hamiltonian H_theta is strictly positive."""
