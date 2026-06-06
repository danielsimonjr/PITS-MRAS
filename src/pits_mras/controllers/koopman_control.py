r"""Koopman-LQR controller (ROADMAP integration #5). NEW capability.

Owning phase: Phase 4 (Controllers).

Wires the deep Koopman lifting model
(:class:`~pits_mras.models.koopman.KoopmanLiftingModel`) into the control loop.
The Koopman latent dynamics :math:`z_{k+1} = z_k A_z^\top + u_k B_z^\top` are
*exactly linear* in :math:`(z, u)` (see ``koopman.py``), so the existing
verifiable linear core applies directly on the **lifted coordinates**: we solve
the Riccati problem on the learned latent system ``(A_z, B_z)`` via
:func:`~pits_mras.utils.lyapunov.solve_care` and close the loop on the lifted
tracking error. This is the bridge from the nonlinear plant (encoder) to the
linear control core.

This controller is purely additive: it reads ``A_z, B_z`` from a frozen model
via ``latent_matrices()`` and never mutates the model or the analytic core.

Q_z lifting convention (``include_state=True`` only, no ``q_latent``):
    With ``include_state=True`` the first ``state_dim`` lifted coordinates *are*
    the state (the lift is :math:`z = [x; \psi(x)]`). The provided state-cost
    ``Q`` (``[state_dim, state_dim]``) is therefore embedded into the leading
    state block of the latent cost and **zero everywhere else**::

        Q_z = [[Q,   0],
               [0,   0]]   shape [latent_dim, latent_dim]

    i.e. only state-error is penalized; the learned extra observables carry no
    direct cost. Pass an explicit ``q_latent`` to penalize the full lifted state.

Shape convention: ``[batch, dim]``, float32 tensors throughout.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from pits_mras.models.koopman import KoopmanLiftingModel
from pits_mras.utils.lyapunov import solve_care

ArrayLike = Union[np.ndarray, Tensor]


def _to_numpy(name: str, m: ArrayLike, shape: tuple[int, int]) -> np.ndarray:
    """Coerce a matrix to a float64 numpy array and validate its shape."""
    if isinstance(m, Tensor):
        arr = m.detach().cpu().numpy()
    else:
        arr = np.asarray(m)
    arr = arr.astype(np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(arr.shape)}.")
    return arr


class KoopmanLQRController(nn.Module):
    r"""LQR controller on Koopman-lifted coordinates.

    Builds the latent state-cost ``Q_z`` (see module docstring for the embedding
    convention), solves ``P_z, K_z = solve_care(A_z, B_z, Q_z, R)`` at
    construction time (the model is frozen then), and applies
    ``u = -(z - z_ref) @ K_z^T`` with ``z = encode(x)`` -- control on the lifted
    tracking error.

    Args:
        koopman_model: a (frozen) Koopman lifting model. Its ``latent_matrices()``
            are detached to numpy for the Riccati solve.
        Q: state-cost ``[state_dim, state_dim]``. Embedded into the leading state
            block of ``Q_z`` when ``q_latent`` is not given (see module docstring).
        R: control-cost ``[control_dim, control_dim]`` (must be PD for
            ``solve_care``).
        q_latent: optional explicit latent state-cost ``[latent_dim, latent_dim]``;
            when given it is used verbatim and ``Q`` is ignored for ``Q_z``.

    Raises:
        ValueError: on shape mismatches, or when the embedding convention is
            requested for an ``include_state=False`` model without ``q_latent``.
        Exception: ``solve_care`` raises (propagated) if ``(A_z, B_z)`` is not
            stabilizable with the given costs -- e.g. a generic untrained model
            whose extra-coord rows are still zeroed.
    """

    Q_z: Tensor
    K_z: Tensor
    P_z: Tensor

    def __init__(
        self,
        koopman_model: KoopmanLiftingModel,
        Q: ArrayLike,
        R: ArrayLike,
        *,
        q_latent: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__()
        self.koopman_model = koopman_model
        state_dim = koopman_model.state_dim
        control_dim = koopman_model.control_dim
        latent_dim = koopman_model.latent_dim
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim

        # -- Read the frozen latent dynamics. --
        A_t, B_t = koopman_model.latent_matrices()
        A_z = _to_numpy("A_z", A_t, (latent_dim, latent_dim))
        B_z = _to_numpy("B_z", B_t, (latent_dim, control_dim))

        # -- Control cost. --
        R_np = _to_numpy("R", R, (control_dim, control_dim))

        # -- Build the latent state-cost Q_z. --
        if q_latent is not None:
            Q_z = _to_numpy("q_latent", q_latent, (latent_dim, latent_dim))
        else:
            Q_np = _to_numpy("Q", Q, (state_dim, state_dim))
            if not koopman_model.include_state:
                raise ValueError(
                    "Q embedding requires include_state=True (the first state_dim "
                    "lifted coords are the state). For an include_state=False model, "
                    "pass an explicit q_latent of shape "
                    f"[{latent_dim}, {latent_dim}]."
                )
            # Embed Q into the leading state block; zero elsewhere (see docstring).
            Q_z = np.zeros((latent_dim, latent_dim), dtype=np.float64)
            Q_z[:state_dim, :state_dim] = Q_np

        # -- Solve the Riccati problem on the lifted coordinates. --
        # solve_care surfaces a clear error if (A_z, B_z) is not stabilizable.
        P_z, K_z = solve_care(A_z, B_z, Q_z, R_np)

        self.register_buffer("Q_z", torch.as_tensor(Q_z, dtype=torch.float32))
        self.register_buffer("P_z", torch.as_tensor(P_z, dtype=torch.float32))
        self.register_buffer("K_z", torch.as_tensor(K_z, dtype=torch.float32))

    def control(self, x: Tensor, x_ref: Tensor) -> Tensor:
        r"""Control on the lifted tracking error.

        Computes ``z = encode(x)``, ``z_ref = encode(x_ref)`` and returns
        ``u = -(z - z_ref) @ K_z^T`` of shape ``[batch, control_dim]``. ``x`` and
        ``x_ref`` follow the usual broadcasting rules of the encoder + subtraction
        (e.g. a single ``[1, state_dim]`` reference broadcasts across a batch).
        """
        z = self.koopman_model.encode(x)
        z_ref = self.koopman_model.encode(x_ref)
        return -(z - z_ref) @ self.K_z.T

    def latent_gain(self) -> Tensor:
        """Return the latent LQR gain ``K_z`` of shape ``[control_dim, latent_dim]``."""
        return self.K_z
