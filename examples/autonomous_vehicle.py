r"""Example: autonomous vehicle lateral control (IP §10.2).

Owning phase: Phase 7 (Examples).

ARCHITECTURE.md §2.1 / ROADMAP §Phase 7: lateral control at 80 km/h, wind-gust
disturbance :math:`\Delta(t)=0.5\sin(2\pi t/10)`, with-CBF vs without-CBF
lane-departure comparison. Per-example numerical acceptance beyond the described
plots is not specified in the sources.

Simplifications (flagged): the plant is the linear reference-model surrogate
driven by :class:`RealtimeInferenceEngine` (state = ``[lateral_offset, rate]``);
a full bicycle / tyre model is NOT simulated. The "without-CBF" branch is built
as a second engine whose controller has the safety filter disabled, so the two
branches differ only in the CBF projection -- a faithful A/B of the safety
layer. The wind gust is injected as the reference command and as an additive
plant disturbance.

Import-safe: all model construction, simulation and plotting live inside
:func:`run` / :func:`main`, guarded by ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import math
from typing import Any


def _build_engine(use_safety_filter: bool) -> Any:
    """Build a fresh inference engine; CBF on/off per flag."""
    import numpy as np
    import torch

    from pits_mras.config import NetworkConfig, PhysicsConfig, PITSMRASConfig
    from pits_mras.controllers.mras import MRASController
    from pits_mras.controllers.reference_models import LinearReferenceModel
    from pits_mras.inference.realtime import RealtimeInferenceEngine
    from pits_mras.models import PITNN

    torch.manual_seed(0)
    np.random.seed(0)

    # Lateral dynamics surrogate: state = [lateral offset, lateral rate].
    a_m = np.array([[0.0, 1.0], [-2.0, -3.0]])
    b_m = np.array([[0.0], [2.0]])
    c_m = np.eye(2)
    q_mat = np.eye(2)
    r_mat = np.eye(1)
    ref_model = LinearReferenceModel(a_m, b_m, c_m, q_mat, r_mat)

    cfg = PITSMRASConfig()
    cfg.network = NetworkConfig(
        input_dim=2, hidden_dim=16, output_dim=2, lstm_layers=1,
        attention_heads=2, embedding_dim=8,
    )
    cfg.physics = PhysicsConfig(
        n_generalized_coords=1, hamiltonian_hidden=16, dissipation_hidden=8,
    )
    pitnn = PITNN(cfg.network, cfg.physics)

    controller = MRASController(
        reference_model=ref_model,
        state_dim=2,
        control_dim=1,
        ref_dim=1,
        plant_dim=2,
        use_safety_filter=use_safety_filter,
    )
    if use_safety_filter:
        controller.setup_safety_filter()

    return RealtimeInferenceEngine(
        pitnn, controller, ref_model, horizon=50, device="cpu"
    )


def _simulate(engine: Any, steps: int) -> dict[str, list]:
    """Run one branch; return per-step lateral offset, error-norm, CBF flag."""
    import torch

    dt = 0.01
    x_p = torch.zeros(2)
    lateral_offset: list[float] = []
    error_norm: list[float] = []
    cbf_active: list[bool] = []

    for k in range(steps):
        t = k * dt
        # Wind gust Delta(t) = 0.5 sin(2 pi t / 10) added to the reference.
        gust = 0.5 * math.sin(2.0 * math.pi * t / 10.0)
        r = torch.tensor([gust], dtype=torch.float32)
        out = engine.step(x_p, r, dt=dt)

        e = out["e"].detach().cpu().reshape(-1)
        error_norm.append(float(torch.linalg.vector_norm(e)))
        cbf_active.append(bool(out["cbf_active"]))

        u = float(out["u_safe"].detach().cpu().reshape(-1)[0])
        x0, x1 = float(x_p[0]), float(x_p[1])
        # Toy plant advance with the gust acting as a lateral disturbance.
        x_p = torch.tensor(
            [x0 + dt * x1, x1 + dt * (u - 2.0 * x0 - 3.0 * x1 + gust)],
            dtype=torch.float32,
        )
        lateral_offset.append(float(x_p[0]))

    return {
        "lateral_offset": lateral_offset,
        "error_norm": error_norm,
        "cbf_active": cbf_active,
    }


def run(steps: int = 100, show: bool = False) -> dict[str, Any]:
    """Run with-CBF vs without-CBF lateral control under a wind gust.

    Args:
        steps: number of fixed-``dt`` control steps to simulate.
        show: when ``True`` display + save the figure; otherwise headless (Agg).

    Returns:
        Dict with per-step series for both branches, scalar summary metrics, and
        the comparison ``Figure`` under ``"figure"``.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    with_cbf = _simulate(_build_engine(use_safety_filter=True), steps)
    without_cbf = _simulate(_build_engine(use_safety_filter=False), steps)

    import matplotlib.pyplot as plt

    dt = 0.01
    tgrid = [k * dt for k in range(steps)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Autonomous vehicle lateral control: CBF vs no-CBF (wind gust)")

    axes[0].plot(tgrid, with_cbf["lateral_offset"], label="with CBF")
    axes[0].plot(
        tgrid, without_cbf["lateral_offset"], label="without CBF", linestyle="--"
    )
    axes[0].axhline(0.0, color="gray", linewidth=0.8)
    axes[0].set_title("lateral offset (lane departure)")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("offset [m]")
    axes[0].legend()

    axes[1].plot(tgrid, with_cbf["error_norm"], label="with CBF")
    axes[1].plot(
        tgrid, without_cbf["error_norm"], label="without CBF", linestyle="--"
    )
    axes[1].set_title(r"tracking error $\|e(t)\|$")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(r"$\|e\|$")
    axes[1].legend()

    fig.tight_layout()
    if show:
        fig.savefig("autonomous_vehicle.png", dpi=120)
        if matplotlib.get_backend().lower() != "agg":
            plt.show()

    max_dep_cbf = max((abs(x) for x in with_cbf["lateral_offset"]), default=0.0)
    max_dep_nocbf = max(
        (abs(x) for x in without_cbf["lateral_offset"]), default=0.0
    )

    return {
        "error_norm": with_cbf["error_norm"],
        "error_norm_no_cbf": without_cbf["error_norm"],
        "lateral_offset_cbf": with_cbf["lateral_offset"],
        "lateral_offset_no_cbf": without_cbf["lateral_offset"],
        "max_departure_cbf": float(max_dep_cbf),
        "max_departure_no_cbf": float(max_dep_nocbf),
        "cbf_activation_rate": (
            sum(with_cbf["cbf_active"]) / len(with_cbf["cbf_active"])
            if with_cbf["cbf_active"] else 0.0
        ),
        "steps": steps,
        "figure": fig,
    }


def main() -> None:
    """Run the demo with the comparison figure displayed and saved."""
    out = run(steps=100, show=True)
    print(f"max lateral departure  with CBF = {out['max_departure_cbf']:.4f} m")
    print(f"max lateral departure   no CBF = {out['max_departure_no_cbf']:.4f} m")
    print(f"CBF activation rate = {out['cbf_activation_rate']:.2%}")


if __name__ == "__main__":
    main()
