r"""Example: building HVAC thermal-zone control (IP §10.3).

Owning phase: Phase 7 (Examples).

ARCHITECTURE.md §2.1 / ROADMAP §Phase 7: thermal-zone control with a
thermal-energy Hamiltonian; energy savings vs a PID baseline plus seasonal
P-hat adaptation. Per-example numerical acceptance beyond the described plots is
not specified in the sources.

Simplifications (flagged): the zone is the linear reference-model surrogate
driven by :class:`RealtimeInferenceEngine` (state = ``[zone_temp_error, rate]``);
no detailed RC building model is simulated. The baseline is a *simple
proportional* controller (not full PID) on the same toy zone, and "energy" is
the accumulated squared control effort :math:`\sum u^2\,dt` -- an effort proxy,
not a metered kWh figure. Seasonal P-hat adaptation is not separately driven
here (the critic adapts online within the run). The setpoint schedule is a slow
day/night sinusoid.

Import-safe: all model construction, simulation and plotting live inside
:func:`run` / :func:`main`, guarded by ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import math
from typing import Any


def _setpoint(t: float) -> float:
    """Slow day/night zone-temperature setpoint deviation [deg C]."""
    return 1.5 * math.sin(2.0 * math.pi * t / 20.0)


def _zone_step(x0: float, x1: float, u: float, dt: float) -> tuple[float, float]:
    """One step of the 2-node RC building-thermal network (zone + thermal mass)
    with a saturated heater. ``x0`` = zone temp, ``x1`` = thermal-mass temp
    (deviations from ambient). Delegates to :func:`examples.plants.rc_thermal_step`.
    """
    import pathlib
    import sys

    import torch

    _dir = str(pathlib.Path(__file__).resolve().parent)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from plants import rc_thermal_step  # noqa: E402  (sibling examples module)

    nxt = rc_thermal_step(torch.tensor([x0, x1], dtype=torch.float32), u, dt)
    return float(nxt[0]), float(nxt[1])


def _run_pits(steps: int) -> dict[str, list]:
    """Run the PITNN -> MRAS -> CBF stack on the toy thermal zone."""
    import numpy as np
    import torch

    from pits_mras.config import NetworkConfig, PhysicsConfig, PITSMRASConfig
    from pits_mras.controllers.mras import MRASController
    from pits_mras.controllers.reference_models import LinearReferenceModel
    from pits_mras.inference.realtime import RealtimeInferenceEngine
    from pits_mras.models import PITNN

    torch.manual_seed(0)
    np.random.seed(0)

    # Reference model = the 2-node RC network's linearization (zone, mass), so
    # the LQR tracks the zone temperature. A_rc = [[-(a_zm+a_za), a_zm],
    # [a_mz, -a_mz]], B = [[heater_gain], [0]] with the rc_thermal_step defaults.
    a_m = np.array([[-3.0, 2.0], [1.0, -1.0]])
    b_m = np.array([[3.0], [0.0]])
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
        use_safety_filter=True,
    )
    controller.setup_safety_filter()
    engine = RealtimeInferenceEngine(
        pitnn, controller, ref_model, horizon=50, device="cpu"
    )

    dt = 0.01
    x0, x1 = 0.0, 0.0
    temp_error: list[float] = []
    energy_cum: list[float] = []
    energy = 0.0

    for k in range(steps):
        t = k * dt
        sp = _setpoint(t)
        r = torch.tensor([sp], dtype=torch.float32)
        x_p = torch.tensor([x0, x1], dtype=torch.float32)
        out = engine.step(x_p, r, dt=dt)
        u = float(out["u_safe"].detach().cpu().reshape(-1)[0])
        energy += u * u * dt
        energy_cum.append(energy)
        x0, x1 = _zone_step(x0, x1, u, dt)
        temp_error.append(abs(x0 - sp))

    return {"temp_error": temp_error, "energy_cum": energy_cum}


def _run_baseline(steps: int, gain: float = 6.0) -> dict[str, list]:
    """Run a simple proportional baseline controller on the same toy zone."""
    dt = 0.01
    x0, x1 = 0.0, 0.0
    temp_error: list[float] = []
    energy_cum: list[float] = []
    energy = 0.0

    for k in range(steps):
        t = k * dt
        sp = _setpoint(t)
        u = gain * (sp - x0)  # proportional control toward the setpoint
        energy += u * u * dt
        energy_cum.append(energy)
        x0, x1 = _zone_step(x0, x1, u, dt)
        temp_error.append(abs(x0 - sp))

    return {"temp_error": temp_error, "energy_cum": energy_cum}


def run(steps: int = 100, show: bool = False) -> dict[str, Any]:
    """Run PITS-MRAS HVAC control vs a simple proportional baseline.

    Args:
        steps: number of fixed-``dt`` control steps to simulate.
        show: when ``True`` display + save the figure; otherwise headless (Agg).

    Returns:
        Dict with per-step series for both controllers, scalar energy/error
        summaries, and the comparison ``Figure`` under ``"figure"``.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    pits = _run_pits(steps)
    base = _run_baseline(steps)

    import matplotlib.pyplot as plt

    dt = 0.01
    tgrid = [k * dt for k in range(steps)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Building HVAC: PITS-MRAS vs proportional baseline")

    axes[0].plot(tgrid, pits["temp_error"], label="PITS-MRAS")
    axes[0].plot(tgrid, base["temp_error"], label="P baseline", linestyle="--")
    axes[0].set_title("zone temperature tracking error")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("|T - setpoint| [deg C]")
    axes[0].legend()

    axes[1].plot(tgrid, pits["energy_cum"], label="PITS-MRAS")
    axes[1].plot(tgrid, base["energy_cum"], label="P baseline", linestyle="--")
    axes[1].set_title(r"cumulative control energy $\sum u^2\,dt$")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("energy [proxy]")
    axes[1].legend()

    fig.tight_layout()
    if show:
        fig.savefig("building_hvac.png", dpi=120)
        if matplotlib.get_backend().lower() != "agg":
            plt.show()

    pits_energy = pits["energy_cum"][-1] if pits["energy_cum"] else 0.0
    base_energy = base["energy_cum"][-1] if base["energy_cum"] else 0.0
    savings = (
        (base_energy - pits_energy) / base_energy if base_energy > 0.0 else 0.0
    )

    return {
        "temp_error": pits["temp_error"],
        "error_norm": pits["temp_error"],
        "temp_error_baseline": base["temp_error"],
        "energy_cum": pits["energy_cum"],
        "energy_cum_baseline": base["energy_cum"],
        "pits_energy": float(pits_energy),
        "baseline_energy": float(base_energy),
        "energy_savings_fraction": float(savings),
        "steps": steps,
        "figure": fig,
    }


def main() -> None:
    """Run the demo with the comparison figure displayed and saved."""
    out = run(steps=100, show=True)
    print(f"PITS-MRAS control energy = {out['pits_energy']:.4f}")
    print(f"baseline  control energy = {out['baseline_energy']:.4f}")
    print(f"energy savings = {out['energy_savings_fraction']:.2%}")


if __name__ == "__main__":
    main()
