r"""Uncertainty quantification utilities (ROADMAP proposal #3).

A small, self-contained toolkit for two complementary kinds of predictive
uncertainty, both lightweight and framework-agnostic (torch + numpy only):

* **Epistemic uncertainty** via a :class:`DeepEnsemble` --- the spread across
  ``K`` independently-trained models is a cheap, well-calibrated estimate of
  *model* uncertainty (Lakshminarayanan et al., 2017).
* **Distribution-free coverage** via *split conformal prediction*
  (:func:`split_conformal_quantile` / :func:`conformal_interval`) and its
  online, non-stationarity-robust extension, *Adaptive Conformal Inference*
  (:class:`AdaptiveConformalInference`; Gibbs & Candès, 2021), which is well
  suited to the time-series setting of MRAS.

Conventions: tensors are ``torch.float32`` with shape ``[batch, d]``; member
models are plain callables (no ``nn.Module`` assumption); asserts in tests are
tolerance-based.
"""

import math
from typing import Callable, Sequence

import torch
from torch import Tensor


class DeepEnsemble:
    """Epistemic uncertainty from ``K`` independently-trained members.

    Each member is a callable mapping an input ``[batch, ...]`` to a prediction
    ``[batch, d]``. The disagreement (std) across members is used as the
    epistemic-uncertainty estimate.
    """

    def __init__(self, members: Sequence[Callable[[Tensor], Tensor]]) -> None:
        """Store the ``K`` member callables.

        Args:
            members: A non-empty sequence of callables, each mapping an input
                tensor to a prediction of shape ``[batch, d]``.

        Raises:
            ValueError: If ``members`` is empty.
        """
        if len(members) == 0:
            raise ValueError("DeepEnsemble requires at least one member model.")
        self.members: tuple[Callable[[Tensor], Tensor], ...] = tuple(members)

    def predict_all(self, x: Tensor) -> Tensor:
        """Return all member predictions stacked along a new leading axis.

        Args:
            x: Input tensor ``[batch, ...]``.

        Returns:
            Tensor of shape ``[K, batch, d]``.
        """
        return torch.stack([member(x) for member in self.members], dim=0)

    def mean_and_std(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the ensemble mean and population std over the member axis.

        The std (computed with ``unbiased=False``, i.e. the population/MLE
        estimator dividing by ``K``) is the epistemic-uncertainty estimate.

        Args:
            x: Input tensor ``[batch, ...]``.

        Returns:
            A pair ``(mean, std)``, each of shape ``[batch, d]``.
        """
        preds = self.predict_all(x)  # [K, batch, d]
        mean = preds.mean(dim=0)
        std = preds.std(dim=0, unbiased=False)
        return mean, std


def split_conformal_quantile(scores: Tensor, alpha: float) -> Tensor:
    """Finite-sample split-conformal quantile of calibration scores.

    Given nonconformity ``scores`` on a calibration set (e.g. absolute
    residuals ``|y - y_hat|``), returns the empirical quantile at level
    ``ceil((n + 1) * (1 - alpha)) / n`` --- the standard split-conformal level
    with the finite-sample (``n + 1``) correction. Using this half-width yields
    marginal coverage of at least ``1 - alpha`` on exchangeable data.

    Args:
        scores: 1-D tensor of nonconformity scores, shape ``[n]``.
        alpha: Target miscoverage level in ``(0, 1)``.

    Returns:
        A scalar tensor: the conformal half-width. If the required rank
        ``ceil((n + 1) * (1 - alpha))`` exceeds ``n`` (``alpha`` too small for
        the calibration size), returns ``+inf`` --- the prediction interval is
        unbounded.

    Raises:
        ValueError: If ``scores`` is not 1-D / is empty, or ``alpha`` is not in
            ``(0, 1)``.
    """
    if scores.dim() != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty 1-D tensor.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in the open interval (0, 1).")

    n = scores.numel()
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return torch.tensor(float("inf"), dtype=scores.dtype, device=scores.device)
    sorted_scores, _ = torch.sort(scores)
    # rank is 1-based; convert to a 0-based index.
    return sorted_scores[rank - 1]


def conformal_interval(pred: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric conformal prediction interval of half-width ``q``.

    Args:
        pred: Point predictions; any shape.
        q: Conformal half-width (scalar tensor or broadcastable to ``pred``).

    Returns:
        A pair ``(lower, upper) = (pred - q, pred + q)``.
    """
    return pred - q, pred + q


class AdaptiveConformalInference:
    r"""Online (adaptive) conformal inference for non-stationary data.

    Implements the Adaptive Conformal Inference (ACI) update of Gibbs & Candès
    (2021). Rather than fixing the miscoverage level, ACI maintains a running
    level ``alpha_t`` that reacts to observed coverage:

    .. math::

        \alpha_{t+1} = \alpha_t + \gamma\,(\alpha^\star - \mathrm{err}_t),

    where :math:`\alpha^\star` is the target miscoverage and
    :math:`\mathrm{err}_t = 1` if the latest true value fell *outside* the
    interval (a miss), else ``0``. Intuition: when recent intervals under-cover
    (``err = 1`` often), ``alpha_t`` shrinks toward 0, which widens the interval;
    when they over-cover, ``alpha_t`` grows, tightening the interval. The running
    empirical miscoverage thus tracks ``target_alpha``. ``alpha_t`` is clamped to
    ``[0, 1]``.
    """

    def __init__(self, target_alpha: float, gamma: float = 0.005) -> None:
        """Initialize the online level.

        Args:
            target_alpha: Desired long-run miscoverage, in ``(0, 1)``.
            gamma: Positive learning rate for the ACI update.

        Raises:
            ValueError: If ``target_alpha`` is not in ``(0, 1)`` or ``gamma`` is
                not positive.
        """
        if not (0.0 < target_alpha < 1.0):
            raise ValueError("target_alpha must be in the open interval (0, 1).")
        if not (gamma > 0.0):
            raise ValueError("gamma must be positive.")
        self.target_alpha: float = target_alpha
        self.gamma: float = gamma
        self._alpha_t: float = target_alpha

    @property
    def current_alpha(self) -> float:
        """The current online miscoverage level ``alpha_t``."""
        return self._alpha_t

    def update(self, covered: bool) -> float:
        """Apply one ACI update after observing the latest coverage outcome.

        Args:
            covered: Whether the latest true value fell inside the interval.

        Returns:
            The updated (clamped) ``alpha_t``.
        """
        err_t = 0.0 if covered else 1.0
        new_alpha = self._alpha_t + self.gamma * (self.target_alpha - err_t)
        self._alpha_t = min(1.0, max(0.0, new_alpha))
        return self._alpha_t
