r"""First-order linearization of a dynamics callable (ROADMAP integration #6).

Owning phase: Phase 1 (Foundation Layer) — pure-math utility on top of
``torch.func``.

The neural H-infinity min-max loop
(:func:`~pits_mras.training.hinf_minmax.hinf_minmax_train`) and the analytic GARE
core (:func:`~pits_mras.utils.lyapunov.solve_gare`) both consume FIXED linear
matrices :math:`(A, B)` for the error dynamics :math:`\dot e = A e + B u (+ D
w)`. Many plants of interest are nonlinear, and learned models (a Koopman
``latent_step``, an analytic plant ``f``) expose their dynamics as a callable
:math:`f(x, u) \to \dot x` rather than as matrices.

This module provides the missing building block: given any continuous-time
dynamics callable ``f(x, u)`` and an operating point ``(x0, u0)``, return the
Jacobians

.. math::
    A = \left.\frac{\partial f}{\partial x}\right|_{(x_0, u_0)}, \qquad
    B = \left.\frac{\partial f}{\partial u}\right|_{(x_0, u_0)},

via :func:`torch.func.jacrev` (``argnums=0`` and ``argnums=1``). This is a
FIRST-ORDER (tangent) linearization about ``(x0, u0)``: it is exact for an affine
``f(x, u) = A x + B u + c`` and a local approximation otherwise. The returned
``(A, B)`` can be fed straight into the linear control core.
"""

from __future__ import annotations

from typing import Callable, Tuple

from torch import Tensor
from torch.func import jacrev


def linearize_dynamics(
    dynamics_fn: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    u0: Tensor,
) -> Tuple[Tensor, Tensor]:
    r"""First-order linearization of ``dynamics_fn`` about the point ``(x0, u0)``.

    Computes the state and control Jacobians of a continuous-time dynamics
    callable ``f(x, u) -> xdot`` at a single operating point, using
    :func:`torch.func.jacrev` with ``argnums=0`` (state) and ``argnums=1``
    (control).

    The result is the tangent linear model

    .. math::
        \dot x \approx f(x_0, u_0) + A\,(x - x_0) + B\,(u - u_0),

    where ``A = df/dx`` and ``B = df/du`` evaluated at ``(x0, u0)``. This is
    EXACT for an affine ``f`` (e.g. ``f(x, u) = A x + B u + c``) and a local
    first-order approximation for a general nonlinear ``f``. The operating-point
    drift ``f(x0, u0)`` is not returned (the linear control core works on the
    error dynamics about the operating point).

    Args:
        dynamics_fn: callable mapping a single state ``x`` of shape
            ``[state_dim]`` and a single control ``u`` of shape ``[control_dim]``
            to the time-derivative ``xdot`` of shape ``[state_dim]``. Must be a
            valid ``torch.func`` transform target (functional / no in-place
            mutation of captured tensors).
        x0: operating-point state, shape ``[state_dim]``.
        u0: operating-point control, shape ``[control_dim]``.

    Returns:
        ``(A, B)`` where ``A`` is ``df/dx`` of shape ``[state_dim, state_dim]``
        and ``B`` is ``df/du`` of shape ``[state_dim, control_dim]``, as
        detached tensors on the dtype/device of ``x0``.

    Raises:
        ValueError: if ``x0`` or ``u0`` is not 1-D, or if the Jacobian shapes do
            not match the expected ``[state_dim, state_dim]`` / ``[state_dim,
            control_dim]`` (e.g. ``dynamics_fn`` does not return a
            ``[state_dim]`` vector).

    Note:
        ``dynamics_fn`` must accept SINGLE (unbatched) ``x`` ``[state_dim]`` and
        ``u`` ``[control_dim]`` inputs. The full sequence-based ``PITNN.forward``
        consumes a history window rather than a single ``(x, u)``; collapsing that
        history into an operating point is an ADR-level design decision and is NOT
        handled here. Wrap PITNN (or any multi-step model) in a one-step
        ``f(x, u)`` adapter before calling this.
    """
    if x0.dim() != 1:
        raise ValueError(f"x0 must be 1-D [state_dim]; got shape {tuple(x0.shape)}.")
    if u0.dim() != 1:
        raise ValueError(f"u0 must be 1-D [control_dim]; got shape {tuple(u0.shape)}.")

    state_dim = x0.shape[0]
    control_dim = u0.shape[0]

    # jacrev(argnums=0) -> df/dx, jacrev(argnums=1) -> df/du, both at (x0, u0).
    A = jacrev(dynamics_fn, argnums=0)(x0, u0).detach()
    B = jacrev(dynamics_fn, argnums=1)(x0, u0).detach()

    if A.shape != (state_dim, state_dim):
        raise ValueError(
            "df/dx has shape "
            f"{tuple(A.shape)}, expected {(state_dim, state_dim)}; "
            "dynamics_fn must return an xdot of shape [state_dim]."
        )
    if B.shape != (state_dim, control_dim):
        raise ValueError(
            "df/du has shape "
            f"{tuple(B.shape)}, expected {(state_dim, control_dim)}; "
            "dynamics_fn must return an xdot of shape [state_dim]."
        )
    return A, B
