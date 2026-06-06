"""Physics-informed pre-training curriculum (IP §8.1, Algorithm 2).

Implements the three-stage curriculum that warms up the physics-informed
temporal neural network (:class:`~pits_mras.models.pitnn.PITNN`) before
closed-loop co-training:

* **Stage 1A** (``epoch <= stage1_epochs``): minimize
  ``lambda_physics * L_physics + 0.1 * L_data`` only.
* **Stage 1B** (next ``stage2_epochs`` epochs): cosine-anneal the data weight
  from 0.1 to 1.0.
* **Stage 1C** (afterwards): keep the data weight saturated at 1.0 and add the
  temporal loss with a linear warm-up.

A validation guard halves the data weight (and logs a warning) whenever the
physics residual spikes above ``epsilon_tol``.

The loop is fully parameterizable so tests can run a single epoch on tiny
synthetic collocation data and still exercise every code path.

Design notes (signatures NOT pinned by the source; flagged Gap G6):

* The source pins only the *lambda schedules*. The function signature, the
  synthetic data generator, the choice of physics/data/temporal surrogate
  terms, and the returned history dict are designed here.
* ``L_physics`` is taken to be the PITNN's own port-Hamiltonian energy residual
  (``output["energy_loss"]``) -- the architecturally-exposed physics term.
* ``L_data`` regresses the PITNN dynamics prediction ``f_hat`` onto the
  derivative of a fixed stable linear surrogate plant
  ``f(x, u) = A_plant x + B_plant u`` (no external dataset; Gap G7).
* ``L_temporal`` re-uses the PITNN's attention-regularization term
  (``output["attn_reg_loss"]``), which is the temporal/attention objective the
  network actually exposes.

Collocation points are sampled uniformly in the state/control/time domain.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable, Iterator

import torch
from torch import Tensor

if TYPE_CHECKING:
    from pits_mras.config import PITSMRASConfig
    from pits_mras.data import TrajectoryDataset
    from pits_mras.models import PITNN

logger = logging.getLogger(__name__)

FTargetFn = Callable[[Tensor, Tensor], Tensor]


def data_weight_schedule(epoch: int, stage1_epochs: int, stage2_epochs: int) -> float:
    """Curriculum weight on the data-fit term ``lambda_data`` (§8.1).

    * Stage 1A (``epoch <= stage1_epochs``): constant 0.1.
    * Stage 1B (next ``stage2_epochs`` epochs): cosine anneal 0.1 -> 1.0 via
      ``0.1 + 0.9 * (1 - cos(pi * (epoch - stage1_epochs) / stage2_epochs)) / 2``.
    * Stage 1C (afterwards): saturated at 1.0.
    """
    if epoch <= stage1_epochs:
        return 0.1
    if epoch <= stage1_epochs + stage2_epochs:
        progress = (epoch - stage1_epochs) / stage2_epochs
        return 0.1 + 0.9 * (1.0 - math.cos(math.pi * progress)) / 2.0
    return 1.0


def temporal_weight_schedule(
    epoch: int,
    stage2_epochs: int,
    lambda_temp_final: float,
    stage1_epochs: int = 1000,
) -> float:
    """Linear warm-up of the temporal weight in Stage 1C (§8.1).

    Zero until ``epoch`` enters Stage 1C (``epoch > stage1_epochs +
    stage2_epochs``), then ramps linearly so that ``lambda_temp ==
    lambda_temp_final`` after another ``stage2_epochs`` epochs. This is the
    canonical ``lambda_temp_final * (epoch - 3000) / 2000`` form with the
    default 1000/2000 stage boundaries.
    """
    boundary = stage1_epochs + stage2_epochs
    if epoch <= boundary:
        return 0.0
    progress = (epoch - boundary) / stage2_epochs
    return lambda_temp_final * min(progress, 1.0)


def _default_f_target(state_dim: int, control_dim: int) -> FTargetFn:
    """Return a fixed stable linear surrogate ``f(x, u) = A x + B u``.

    The matrices are deterministic so the regression target is reproducible
    across runs. Output dimension equals ``state_dim`` (== PITNN ``output_dim``).
    """
    generator = torch.Generator().manual_seed(12345)
    a = torch.randn(state_dim, state_dim, generator=generator) * 0.3
    a = a - torch.eye(state_dim) * (state_dim * 0.5)  # mildly stable
    b = torch.randn(state_dim, control_dim, generator=generator) * 0.3

    def f_target(state: Tensor, control: Tensor) -> Tensor:
        return torch.einsum("ij,bj->bi", a, state) + torch.einsum("ij,bj->bi", b, control)

    return f_target


def _sample_collocation(
    batch_size: int,
    input_dim: int,
    control_dim: int,
    output_dim: int,
    history_length: int,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Uniformly sample collocation points for one PITNN forward pass.

    Returns a dict of the six PITNN ``forward`` arguments plus the convenience
    ``x_p_curr``/``u_curr`` used to build the regression target.
    """

    def uni(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator) * 2.0 - 1.0

    return {
        "x_hist": uni(batch_size, history_length, input_dim),
        "u_hist": uni(batch_size, history_length, input_dim),
        "x_p_curr": uni(batch_size, input_dim),
        "u_curr": uni(batch_size, control_dim),
        "e_curr": uni(batch_size, output_dim),
        "e_hist": uni(batch_size, history_length, output_dim),
    }


def _fit_dim(x: Tensor, dim: int) -> Tensor:
    """Slice or zero-pad the last axis of ``x`` to width ``dim``.

    Lets a dataset whose ``state_dim`` / ``control_dim`` differs from the
    PITNN's wired ``input_dim`` be fed through the forward pass without error:
    extra channels are dropped, missing channels are zero-filled.
    """
    have = x.shape[-1]
    if have == dim:
        return x
    if have > dim:
        return x[..., :dim]
    pad_shape = list(x.shape)
    pad_shape[-1] = dim - have
    return torch.cat([x, x.new_zeros(pad_shape)], dim=-1)


def _dataset_pitnn_batch(
    batch: dict[str, Tensor],
    input_dim: int,
    control_dim: int,
    output_dim: int,
    dt: float,
) -> tuple[dict[str, Tensor], Tensor]:
    """Map a :class:`TrajectoryDataset` batch to PITNN inputs + a regression target.

    The dataset carries no tracking error or reference model, so the error
    tensors ``e_curr`` / ``e_hist`` are passed as zeros (pre-training is a
    physics/data warm-up, not a closed-loop step). The data-fit target is the
    finite-difference state derivative ``(next_state - state) / dt`` -- the
    discrete analogue of ``x_dot`` produced by the same plant the generator
    rolls out -- sliced to ``output_dim``.

    Returns ``(forward_kwargs, f_target)``.
    """
    state_hist = _fit_dim(batch["state_hist"], input_dim)  # [B, W, input_dim]
    control_hist = _fit_dim(batch["control_hist"], input_dim)  # [B, W, input_dim]
    x_p_curr = _fit_dim(batch["state"], input_dim)  # [B, input_dim]
    u_curr = _fit_dim(batch["control"], control_dim)  # [B, control_dim]
    b = x_p_curr.shape[0]
    forward_kwargs = {
        "x_hist": state_hist,
        "u_hist": control_hist,
        "x_p_curr": x_p_curr,
        "u_curr": u_curr,
        "e_curr": x_p_curr.new_zeros(b, output_dim),
        "e_hist": state_hist.new_zeros(b, state_hist.shape[1], output_dim),
    }
    x_dot = (batch["next_state"] - batch["state"]) / dt  # [B, state_dim]
    f_target = _fit_dim(x_dot, output_dim)
    return forward_kwargs, f_target


def _cycle_batches(loader: "torch.utils.data.DataLoader") -> Iterator[dict[str, Tensor]]:
    """Yield batches from ``loader`` indefinitely (re-iterating each pass)."""
    while True:
        for batch in loader:
            yield batch


def pretrain_pitnn(
    pitnn: "PITNN",
    cfg: "PITSMRASConfig",
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    f_target_fn: FTargetFn | None = None,
    epsilon_tol: float = 1e3,
    history_length: int = 8,
    seed: int | None = None,
    dataset: "TrajectoryDataset | None" = None,
) -> dict[str, list[float]]:
    """Run the three-stage curriculum pre-training (Algorithm 2).

    Args:
        pitnn: the model to train (updated in place). Built from
            ``cfg.network`` / ``cfg.physics`` so its ``input_dim``,
            ``output_dim`` and ``n_q`` (== control_dim) are read off the model.
        cfg: full configuration; ``cfg.training`` and ``cfg.losses`` are read.
        epochs: number of epochs (defaults to ``cfg.training.pretrain_epochs``).
        batch_size: collocation batch size per epoch.
        lr: Adam learning rate (defaults to ``cfg.training.pretrain_lr``).
        f_target_fn: optional regression target ``f(x, u)``; defaults to a
            fixed stable linear surrogate plant.
        epsilon_tol: physics-residual threshold; if exceeded the data weight is
            halved and a warning is logged (validation criterion, §8.1).
        history_length: length of the synthetic history window.
        seed: optional RNG seed for reproducible synthetic data.
        dataset: optional :class:`~pits_mras.data.TrajectoryDataset` (Gap G7,
            opt-in). When ``None`` (default) the function is unchanged -- it
            synthesizes inline collocation data with the same RNG consumption as
            before. When supplied, each epoch instead draws a windowed batch
            from the dataset (cycled), maps it to the PITNN forward inputs, and
            uses the finite-difference state derivative ``(x_next - x) / dt`` as
            the data-fit target (``f_target_fn`` is ignored on this path). The
            dataset's ``memory_horizon`` replaces ``history_length`` for the
            window width.

    Returns:
        A history dict mapping metric names to per-epoch lists:
        ``total_loss``, ``physics_loss``, ``data_loss``, ``temporal_loss``,
        ``lambda_data``, ``lambda_temp``.
    """
    train_cfg = cfg.training
    loss_cfg = cfg.losses

    epochs = train_cfg.pretrain_epochs if epochs is None else epochs
    batch_size = train_cfg.pretrain_batch_size if batch_size is None else batch_size
    lr = train_cfg.pretrain_lr if lr is None else lr
    if seed is None:
        seed = train_cfg.seed

    stage1_epochs = train_cfg.stage1_epochs
    stage2_epochs = train_cfg.stage2_epochs

    input_dim = pitnn.input_dim
    output_dim = pitnn.output_dim
    control_dim = pitnn.n_q  # control enters via the momentum channel

    if f_target_fn is None:
        f_target_fn = _default_f_target(output_dim, control_dim)

    optimizer = torch.optim.Adam(pitnn.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    # ── Opt-in dataset path (Gap G7). Default off -> ``batch_iter is None`` ->
    # the inline-collocation branch below runs unchanged (same RNG consumption).
    batch_iter: Iterator[dict[str, Tensor]] | None = None
    if dataset is not None:
        from pits_mras.data import make_dataloader

        loader_gen = torch.Generator().manual_seed(seed)
        loader = make_dataloader(dataset, batch_size=batch_size, shuffle=True, generator=loader_gen)
        batch_iter = _cycle_batches(loader)

    history: dict[str, list[float]] = {
        "total_loss": [],
        "physics_loss": [],
        "data_loss": [],
        "temporal_loss": [],
        "lambda_data": [],
        "lambda_temp": [],
    }

    for epoch in range(1, epochs + 1):
        lambda_data = data_weight_schedule(epoch, stage1_epochs, stage2_epochs)
        lambda_temp = temporal_weight_schedule(
            epoch,
            stage2_epochs,
            loss_cfg.lambda_temporal,
            stage1_epochs=stage1_epochs,
        )

        if batch_iter is None:
            # ── default path: inline synthetic collocation (unchanged) ──
            batch = _sample_collocation(
                batch_size, input_dim, control_dim, output_dim, history_length, generator
            )
            f_target = f_target_fn(batch["x_p_curr"][:, :output_dim], batch["u_curr"])
            forward_kwargs = {
                "x_hist": batch["x_hist"],
                "u_hist": batch["u_hist"],
                "x_p_curr": batch["x_p_curr"],
                "u_curr": batch["u_curr"],
                "e_curr": batch["e_curr"],
                "e_hist": batch["e_hist"],
            }
        else:
            # ── opt-in dataset path (Gap G7): draw a windowed batch ──
            forward_kwargs, f_target = _dataset_pitnn_batch(
                next(batch_iter), input_dim, control_dim, output_dim, train_cfg.dt
            )

        output = pitnn(**forward_kwargs)
        # L_physics: the architecturally-exposed port-Hamiltonian energy residual.
        l_physics = output["energy_loss"]
        # L_data: regress the dynamics prediction onto the surrogate target.
        l_data = (output["f_hat"] - f_target).pow(2).mean()

        # Validation criterion (§8.1): halve the data weight on a physics spike.
        if float(l_physics.detach()) > epsilon_tol:
            lambda_data *= 0.5
            logger.warning(
                "L_physics=%.4g exceeded epsilon_tol=%.4g at epoch %d; "
                "halving lambda_data to %.4g",
                float(l_physics.detach()),
                epsilon_tol,
                epoch,
                lambda_data,
            )

        total = loss_cfg.lambda_physics * l_physics + lambda_data * l_data

        l_temporal = output["attn_reg_loss"]
        if lambda_temp > 0.0:
            total = total + lambda_temp * l_temporal

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        history["total_loss"].append(float(total.detach()))
        history["physics_loss"].append(float(l_physics.detach()))
        history["data_loss"].append(float(l_data.detach()))
        history["temporal_loss"].append(float(l_temporal.detach()))
        history["lambda_data"].append(lambda_data)
        history["lambda_temp"].append(lambda_temp)

        if epoch % max(train_cfg.log_every, 1) == 0:
            logger.info(
                "pretrain epoch %d/%d  total=%.4g physics=%.4g data=%.4g",
                epoch,
                epochs,
                history["total_loss"][-1],
                history["physics_loss"][-1],
                history["data_loss"][-1],
            )

    return history
