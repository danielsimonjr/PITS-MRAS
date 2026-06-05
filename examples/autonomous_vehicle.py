r"""Example: autonomous vehicle lateral control (IP §10.2).

Owning phase: Phase 7 (Examples).

ARCHITECTURE.md §2.1 / ROADMAP §Phase 7: lateral control under a wind-gust
disturbance, with-CBF vs without-CBF lane-departure comparison. Per-example
numerical acceptance beyond the described plots is not specified in the sources.

The scenario is a *lane-hold under a strong wind gust*: the car is commanded to
hold the lane centre (``r = 0``) while a strong lateral gust
:math:`\Delta(t)=20\sin(2\pi t/2)` acts on the plant. The CBF safety filter uses
a tight ellipsoid (:math:`c=0.5` in :math:`h(e)=c-e^\top P e`, with ``P`` the
CARE Lyapunov matrix), so it actively engages during the gust (panel (c) shows
when). The two branches differ only in the CBF projection -- a faithful A/B of
the safety layer.

Honest interpretation: the nominal MRAS controller is near-LQR-optimal and the
tracking-error dynamics are stable, so the CBF behaves as a *minimally-invasive
safety backstop* -- it engages on the gust peaks and slightly reduces the
worst-case lateral departure and the safe-set violation
:math:`\int\max(0, e^\top P e - c)\,dt`, without degrading nominal tracking. A
dramatic departure gap would require a deliberately mistuned nominal controller;
here the value shown is that the filter is active yet non-disruptive.

Simplifications (flagged): the plant is the linear reference-model surrogate
driven by :class:`RealtimeInferenceEngine` (state = ``[lateral_offset, rate]``);
a full bicycle / tyre model is NOT simulated.

Import-safe: all model construction, simulation and plotting live inside
:func:`run` / :func:`main`, guarded by ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import math
from typing import Any

# Wind-gust amplitude (strong lateral disturbance) and period [s].
_GUST_AMPLITUDE = 20.0
_GUST_PERIOD = 2.0
# CBF safety-set size c in h(e) = c - e^T P e (tight, so the filter is active).
_SAFETY_MARGIN = 0.5


def _build_engine(use_safety_filter: bool) -> Any:
    """Build a fresh inference engine; CBF on/off per flag.

    The CBF safety filter is built from the controller's critic, which is
    warm-started to the CARE solution ``P_opt`` -- so the safety ellipsoid uses
    the proper Lyapunov matrix regardless of the (online-adapting) critic.
    """
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
        input_dim=2,
        hidden_dim=16,
        output_dim=2,
        lstm_layers=1,
        attention_heads=2,
        embedding_dim=8,
    )
    cfg.physics = PhysicsConfig(
        n_generalized_coords=1,
        hamiltonian_hidden=16,
        dissipation_hidden=8,
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
        controller.setup_safety_filter(safety_margin=_SAFETY_MARGIN, decay_rate=2.0)

    engine = RealtimeInferenceEngine(pitnn, controller, ref_model, horizon=50, device="cpu")
    # Expose the safety matrix P for the constraint-violation metric.
    engine._cbf_P = ref_model.P_opt.detach().cpu().numpy()  # type: ignore[attr-defined]
    return engine


def _simulate(engine: Any, steps: int) -> dict[str, list]:
    """Run one lane-hold branch under a strong wind gust.

    The car is commanded to hold the lane centre (``r = 0``); a strong lateral
    wind gust acts on the plant. Returns per-step lateral offset, tracking-error
    norm, CBF-activation flag, and the safe-set violation ``max(0, e^T P e - c)``.
    """
    import pathlib
    import sys

    import torch

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from plants import lateral_tyre_step  # noqa: E402  (sibling examples module)

    dt = 0.01
    x_p = torch.zeros(2)
    P = engine._cbf_P  # noqa: SLF001  (safety matrix stashed by _build_engine)
    lateral_offset: list[float] = []
    error_norm: list[float] = []
    cbf_active: list[bool] = []
    violation: list[float] = []

    for k in range(steps):
        t = k * dt
        # Strong lateral wind gust acting on the plant (NOT the reference); the
        # controller is asked to hold the lane centre against it.
        gust = _GUST_AMPLITUDE * math.sin(2.0 * math.pi * t / _GUST_PERIOD)
        out = engine.step(x_p, torch.zeros(1, dtype=torch.float32), dt=dt)

        e = out["e"].detach().cpu().reshape(-1)
        error_norm.append(float(torch.linalg.vector_norm(e)))
        cbf_active.append(bool(out["cbf_active"]))
        e_np = e.numpy()
        violation.append(max(0.0, float(e_np @ P @ e_np) - _SAFETY_MARGIN))
        # Record the lateral offset at the CURRENT time (before the plant step),
        # so all series align on the same tgrid.
        lateral_offset.append(float(x_p[0]))

        # Nonlinear single-track lateral plant with tyre-force saturation
        # (tanh); linearizes to the reference model near the lane centre.
        u = float(out["u_safe"].detach().cpu().reshape(-1)[0])
        x_p = lateral_tyre_step(x_p, u, dt, tyre_stiffness=2.0, damping=3.0, gust=gust)

    return {
        "lateral_offset": lateral_offset,
        "error_norm": error_norm,
        "cbf_active": cbf_active,
        "violation": violation,
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Autonomous vehicle lane-hold under a strong wind gust: CBF vs no-CBF")

    axes[0].plot(tgrid, with_cbf["lateral_offset"], label="with CBF")
    axes[0].plot(tgrid, without_cbf["lateral_offset"], label="without CBF", linestyle="--")
    axes[0].axhline(0.0, color="gray", linewidth=0.8)
    axes[0].set_title("(a) lateral offset (lane departure)")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("offset [m]")
    axes[0].legend()

    axes[1].plot(tgrid, with_cbf["error_norm"], label="with CBF")
    axes[1].plot(tgrid, without_cbf["error_norm"], label="without CBF", linestyle="--")
    axes[1].set_title(r"(b) tracking error $\|e(t)\|$")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(r"$\|e\|$")
    axes[1].legend()

    axes[2].step(
        tgrid,
        [int(c) for c in with_cbf["cbf_active"]],
        where="post",
        color="tab:red",
    )
    axes[2].set_title("(c) CBF activation flag (with-CBF branch)")
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("active")
    axes[2].set_ylim(-0.1, 1.1)

    fig.tight_layout()
    if show:
        fig.savefig("autonomous_vehicle.png", dpi=120)
        if matplotlib.get_backend().lower() != "agg":
            plt.show()

    max_dep_cbf = max((abs(x) for x in with_cbf["lateral_offset"]), default=0.0)
    max_dep_nocbf = max((abs(x) for x in without_cbf["lateral_offset"]), default=0.0)
    viol_cbf = sum(with_cbf["violation"]) * dt
    viol_nocbf = sum(without_cbf["violation"]) * dt

    return {
        "error_norm": with_cbf["error_norm"],
        "error_norm_no_cbf": without_cbf["error_norm"],
        "lateral_offset_cbf": with_cbf["lateral_offset"],
        "lateral_offset_no_cbf": without_cbf["lateral_offset"],
        "max_departure_cbf": float(max_dep_cbf),
        "max_departure_no_cbf": float(max_dep_nocbf),
        "safeset_violation_cbf": float(viol_cbf),
        "safeset_violation_no_cbf": float(viol_nocbf),
        "cbf_activation_rate": (
            sum(with_cbf["cbf_active"]) / len(with_cbf["cbf_active"])
            if with_cbf["cbf_active"]
            else 0.0
        ),
        "steps": steps,
        "figure": fig,
    }


def main() -> None:
    """Run the demo with the comparison figure displayed and saved."""
    out = run(steps=100, show=True)
    print(f"CBF activation rate = {out['cbf_activation_rate']:.2%}")
    print(f"max lateral departure  with CBF = {out['max_departure_cbf']:.4f} m")
    print(f"max lateral departure   no CBF = {out['max_departure_no_cbf']:.4f} m")
    print(f"safe-set violation  with CBF = {out['safeset_violation_cbf']:.4f}")
    print(f"safe-set violation   no CBF = {out['safeset_violation_no_cbf']:.4f}")


if __name__ == "__main__":
    main()
