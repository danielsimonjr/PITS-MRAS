"""Shared pytest fixtures for the PITS-MRAS test suite."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _functorch_warmup() -> None:
    """Pay the one-time ``torch.func`` / functorch lazy-init cost ONCE, up front.

    The first ``jacrev`` / ``vmap`` / ``autograd.functional.jacobian`` call in a
    process triggers a ~15s functorch lazy initialization. Without this fixture
    that cost lands unpredictably inside whichever test happens to make the first
    such call (notably the example tests and the ``torch.func`` code in
    ``models/pcml.py``), making timing noisy. This session-scoped autouse fixture
    forces the init at session start with trivial 1-2D calls.

    It is pure warmup: side-effect-free and seed-free (it deliberately does NOT
    touch ``torch.manual_seed`` or any global state), so it cannot change any
    test outcome.
    """

    def _f(x: torch.Tensor) -> torch.Tensor:
        return (x * x).sum()

    x = torch.ones(2)

    # torch.func.jacrev + torch.func.vmap (the functorch transforms).
    torch.func.jacrev(_f)(x)
    batched = torch.ones(3, 2)
    torch.func.vmap(torch.func.jacrev(_f))(batched)

    # The classic autograd.functional.jacobian path also pays a lazy cost.
    torch.autograd.functional.jacobian(_f, x)
