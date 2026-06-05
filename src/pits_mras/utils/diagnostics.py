r"""Long-horizon rollout-stability + conservation-drift diagnostics (ROADMAP #4).

The port-Hamiltonian decoder enforces an energy residual that vanishes *by
construction* at each step, but that says nothing about whether a learned map
stays physically consistent or numerically stable when iterated over a long
rollout. These pure, dependency-free (``torch``-only) diagnostics validate that:

- :func:`energy_drift` / :func:`max_energy_drift` -- how far a nominally
  conserved quantity (e.g. the Hamiltonian :math:`H(t)`) drifts from its
  initial value along a rollout. Zero drift = perfect conservation.
- :func:`valid_prediction_time` -- the Valid Prediction Time (VPT): the elapsed
  time before a rollout's normalized error first exceeds a tolerance. A standard
  long-horizon forecast-quality metric.
- :func:`rollout_jacobian_spectral_radius` -- the spectral radius of the
  one-step Jacobian, i.e. the local error-amplification factor of the learned
  map. A value ``> 1`` indicates errors grow geometrically (instability).

Shape convention (repo-wide): ``[batch, dim]`` and ``[batch, T, dim]`` -- never
``[dim, batch]``. Conserved-quantity series are ``[T]`` or ``[batch, T]``.
"""

from typing import Callable

import torch
from torch import Tensor


def energy_drift(quantity: Tensor, relative: bool = True, eps: float = 1e-8) -> Tensor:
    r"""Drift of a conserved quantity relative to its initial value.

    Given a conserved-quantity series ``quantity`` (e.g. the Hamiltonian
    :math:`H(t)` sampled along a rollout), returns the per-timestep drift versus
    the first sample:

    - absolute (``relative=False``): :math:`q_t - q_0`
    - relative (``relative=True``, default):
      :math:`(q_t - q_0) / (|q_0| + \varepsilon)`

    Args:
        quantity: shape ``[T]`` or ``[batch, T]``. Time is the last axis.
        relative: if ``True``, normalize the drift by ``|q_0| + eps``.
        eps: small constant guarding the relative normalization against a
            near-zero initial value.

    Returns:
        The drift series with the same shape as ``quantity``. By construction
        ``drift[..., 0] == 0``.
    """
    q0 = quantity[..., 0:1]  # keep last dim for broadcasting -> [...,1]
    drift = quantity - q0
    if relative:
        drift = drift / (q0.abs() + eps)
    return drift


def max_energy_drift(quantity: Tensor, relative: bool = True, eps: float = 1e-8) -> Tensor:
    r"""Maximum absolute drift of a conserved quantity over time.

    The worst-case excursion of ``quantity`` from its initial value, i.e.
    :math:`\max_t |\text{drift}_t|` where ``drift`` is :func:`energy_drift`.

    Args:
        quantity: shape ``[T]`` or ``[batch, T]``. Time is the last axis.
        relative: passed through to :func:`energy_drift`.
        eps: passed through to :func:`energy_drift`.

    Returns:
        A scalar tensor for ``[T]`` input, or a ``[batch]`` tensor (one max per
        batch element) for ``[batch, T]`` input.
    """
    return energy_drift(quantity, relative=relative, eps=eps).abs().amax(dim=-1)


def valid_prediction_time(
    pred: Tensor,
    truth: Tensor,
    threshold: float,
    dt: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    r"""Valid Prediction Time (VPT): elapsed time before error exceeds tolerance.

    Computes the per-timestep normalized L2 error
    :math:`\varepsilon_t = \lVert \text{pred}_t - \text{truth}_t \rVert /
    (\lVert \text{truth}_t \rVert + \varepsilon)` (the norm is over the ``dim``
    axis), then returns :math:`t^\star \cdot dt` where :math:`t^\star` is the
    first time index with :math:`\varepsilon_{t^\star} > \text{threshold}`. If
    the threshold is never exceeded, returns the full valid horizon
    :math:`(T - 1)\,dt`.

    Args:
        pred: predicted rollout, shape ``[T, dim]`` or ``[batch, T, dim]``.
        truth: ground-truth rollout, same shape as ``pred``.
        threshold: normalized-error tolerance above which a step is "invalid".
        dt: time step, used to convert the index to elapsed time.
        eps: small constant guarding the per-step normalization.

    Returns:
        A scalar tensor for ``[T, dim]`` input, or a ``[batch]`` tensor of VPTs
        for ``[batch, T, dim]`` input.
    """
    err = torch.linalg.vector_norm(pred - truth, dim=-1)  # [...,T]
    norm = torch.linalg.vector_norm(truth, dim=-1)  # [...,T]
    rel_err = err / (norm + eps)  # [...,T]
    exceeded = rel_err > threshold  # bool [...,T]
    T = rel_err.shape[-1]

    any_exceeded = exceeded.any(dim=-1)  # [...]
    # argmax on a bool tensor returns the index of the FIRST True (ties -> first).
    first_idx = exceeded.to(torch.long).argmax(dim=-1)  # [...]
    full_horizon = torch.full_like(first_idx, T - 1)
    idx = torch.where(any_exceeded, first_idx, full_horizon)
    return idx.to(rel_err.dtype) * dt


def rollout_jacobian_spectral_radius(step_fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    r"""Spectral radius of the one-step rollout Jacobian at state ``x``.

    Computes the Jacobian :math:`J = \partial(\text{step\_fn}) / \partial x`
    evaluated at the single state ``x`` (no batch axis) via
    :func:`torch.autograd.functional.jacobian`, then returns the spectral radius
    :math:`\rho(J) = \max_i |\lambda_i(J)|`.

    The spectral radius is the local error-amplification factor of the learned
    map: :math:`\rho > 1` means a small perturbation grows geometrically under
    repeated application (instability), while :math:`\rho < 1` means it
    contracts.

    Args:
        step_fn: a callable mapping a single state ``[dim]`` to the next state
            ``[dim]`` (one rollout step). Must be differentiable w.r.t. ``x``.
        x: a single state, shape ``[dim]`` (no batch axis).

    Returns:
        A 0-dim scalar tensor: the spectral radius ``max(abs(eig(J)))``.
    """
    jac = torch.autograd.functional.jacobian(step_fn, x)  # [dim, dim]
    eigvals = torch.linalg.eigvals(jac)  # complex [dim]
    return eigvals.abs().max()
