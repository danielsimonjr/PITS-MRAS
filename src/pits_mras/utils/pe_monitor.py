r"""Persistence-of-excitation monitor (IP §4.5).

Owning phase: Phase 1 (Foundation Layer).

IRL/ADP parameter convergence requires PE of the regressor :math:`\phi_c(e, t)`.
This module checks the PE condition and, if not met, supplies a small probing
signal to add to the control input to satisfy it.

PE condition: :math:`\exists\,\delta, T > 0` such that
:math:`\int_t^{t+T} \phi(\tau)\phi(\tau)^\top \, d\tau \succeq \delta I \;\forall t`.

Caveat (Blueprint): injecting probing noise biases estimates unless handled.
"""

from collections import deque
from typing import Deque

import torch
from torch import Tensor


class PEMonitor:
    """Monitor the min eigenvalue of the regressor Gram matrix.

    Tracks regressor vectors over a sliding window and reports whether the
    persistence-of-excitation condition holds; if it does not, callers can add
    probing noise (via :meth:`get_probing_noise`) to the control input.
    """

    def __init__(
        self,
        regressor_dim: int,
        window_size: int = 200,
        pe_threshold: float = 1e-3,
        noise_std: float = 0.01,
    ) -> None:
        self.regressor_dim = regressor_dim
        self.window_size = window_size
        self.pe_threshold = pe_threshold
        self.noise_std = noise_std
        self._buffer: Deque[Tensor] = deque(maxlen=window_size)

    def update(self, phi: Tensor) -> None:
        """Add a new regressor vector (shape ``[n]``) to the buffer."""
        self._buffer.append(phi.detach().cpu())

    def is_pe_satisfied(self) -> bool:
        """Return ``True`` if the PE condition holds for the current window."""
        if len(self._buffer) < self.window_size:
            return False
        Phi = torch.stack(list(self._buffer))  # [window_size, n]
        gram = (Phi.T @ Phi) / self.window_size
        min_eig = torch.linalg.eigvalsh(gram).min().item()
        return bool(min_eig > self.pe_threshold)

    def get_probing_noise(self, control_dim: int, device: str = "cpu") -> Tensor:
        """Return small probing noise to add to ``u`` when PE is not satisfied."""
        return torch.randn(control_dim, device=device) * self.noise_std
