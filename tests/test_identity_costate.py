"""Identity test (IP §11.2): Costate = Critic Gradient (Identity 2).

Targets ``pits_mras.models.critic`` (``CostateHead`` / ``QuadraticCritic``).
Owning phase: Phase 2 per ROADMAP.md (authored alongside its target phase;
§11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.2):
``test_costate_equals_grad_V``, ``test_optimal_control_equals_lqr_gain``.

Phase-2 status:
- ``test_costate_equals_grad_V`` is UN-SKIPPED: the costate head's action IS the
  autodiff gradient of the critic by construction, so the identity holds with
  only Phase-2 code.
- ``test_optimal_control_equals_lqr_gain`` stays SKIPPED: recovering the LQR gain
  requires the reference model's CARE solution + critic warm-start that land in
  Phase 4 (``controllers/mras.py``). It is left as a placeholder for that phase.
"""

import pytest
import torch

from pits_mras.models.critic import CostateHead, QuadraticCritic


def test_costate_equals_grad_V() -> None:
    """I2: the costate equals the autodiff gradient of the value head.

    The ``CostateHead`` returns ``lambda_hat = critic.gradient(e)``; this is, by
    construction, ``grad_e V(e)``. We verify it against a fresh autograd pass and
    against the closed form ``2 P e`` for the quadratic critic.
    """
    torch.manual_seed(0)
    n, m = 4, 2
    critic = QuadraticCritic(state_dim=n)
    head = CostateHead(critic, torch.eye(m), torch.randn(n, m))

    e = torch.randn(6, n)
    lam_head, _ = head(e)

    # Independent autograd gradient of V.
    e2 = e.clone().detach().requires_grad_(True)
    V = critic(e2)
    grad_V = torch.autograd.grad(V.sum(), e2)[0]
    assert torch.allclose(lam_head, grad_V, atol=1e-5)

    # Closed form 2 P e.
    P = critic.extract_P()
    assert torch.allclose(lam_head, 2.0 * (e @ P.T), atol=1e-5)


@pytest.mark.skip(reason="phase 4: needs CARE solution + critic LQR warm-start (controllers/mras)")
def test_optimal_control_equals_lqr_gain() -> None:
    """u* = -R^-1 B^T grad V recovers the LQR gain in the linear limit."""
