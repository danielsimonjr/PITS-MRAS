r"""Example: hard PCML on the 1-D heat equation with real coordinates.

Owning phase: Phase 7 (Examples) / PCML (v0.3.0).

This is the coordinate-bearing PCML demonstration: unlike the control-loop
examples (whose linear-surrogate plants have no spatial/temporal coordinates),
here the problem genuinely has independent variables :math:`(x, t)` and the
constraint is a partial differential equation, so the PCML derivative variables
:math:`d=[\partial_x T,\,\partial_t T,\,\partial_{xx} T,\,\partial_{tt} T]` are
the *real* autodiff derivatives of the network output.

Problem: 1-D heat diffusion :math:`\partial_t T = \alpha\,\partial_{xx} T`
(:class:`~pits_mras.constraints.HeatConductionDAE`). With :math:`\alpha=1` the
field :math:`T^\*(x,t)=e^{-t}\sin x` satisfies it exactly, and is used as the
data target.

Two phases (DAE-HardNet soft -> hard):

1. **Soft PCML** (:class:`~pits_mras.models.pcml.SoftPCMLLoss`): train a small MLP
   backbone :math:`T_\theta(x,t)` with ``data_mse + lambda * soft_pcml`` over
   collocation points; the heat-equation residual is computed from the *real*
   autodiff derivatives. Both the data loss and the constraint violation fall.
2. **Hard PCML** (:class:`~pits_mras.models.pcml.PCMLModule` /
   :class:`~pits_mras.models.pcml.KKTProjectionLayer`): the differentiable KKT
   projection maps the network prediction onto the heat-equation manifold,
   driving the point-wise violation to ~0 regardless of the soft-trained
   residual.

Import-safe: all model construction, training and plotting live inside
:func:`run` / :func:`main`, guarded by ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import math
from typing import Any


def _derivatives(backbone: Any, inputs: Any) -> Any:
    """Autodiff ``[dT/dx, dT/dt, d2T/dx2, d2T/dt2]`` of ``T(x,t)`` -> ``[N, 4]``."""
    import torch

    inp = inputs.requires_grad_(True)
    T = backbone(inp)
    grad1 = torch.autograd.grad(T.sum(), inp, create_graph=True)[0]  # [N, 2]
    dT_dx = grad1[:, 0:1]
    dT_dt = grad1[:, 1:2]
    d2T_dx2 = torch.autograd.grad(dT_dx.sum(), inp, create_graph=True)[0][:, 0:1]
    d2T_dt2 = torch.autograd.grad(dT_dt.sum(), inp, create_graph=True)[0][:, 1:2]
    return T, torch.cat([dT_dx, dT_dt, d2T_dx2, d2T_dt2], dim=-1)


def run(steps: int = 250, show: bool = False) -> dict[str, Any]:
    """Train soft PCML on the heat equation, then hard-project; return diagnostics.

    Args:
        steps: number of soft-training iterations.
        show: when ``True`` display + save the figure; otherwise headless (Agg).

    Returns:
        Dict with per-step ``data_loss`` and ``violation`` series, the pre/post
        hard-projection violations, summary scalars, and the ``Figure``.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import torch

    from pits_mras.constraints import HeatConductionDAE
    from pits_mras.models.pcml import PCMLModule, SoftPCMLLoss

    torch.manual_seed(0)

    alpha = 1.0
    dae = HeatConductionDAE(alpha=alpha, T_min=-2.0, T_max=2.0)
    backbone = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )

    # ---- Collocation points over (x, t) in [0, pi] x [0, 1]. ----------------
    n_points = 128
    x = torch.rand(n_points, 1) * math.pi
    t = torch.rand(n_points, 1)
    t_true = torch.exp(-t) * torch.sin(x)  # T*(x,t) = e^{-t} sin x

    soft = SoftPCMLLoss(dae, lambda_diff=1.0, lambda_eq=1.0, lambda_ineq=0.1)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=2e-3)

    data_loss: list[float] = []
    violation: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad()
        inputs = torch.cat([x, t], dim=-1)
        T, d = _derivatives(backbone, inputs)
        l_data = torch.nn.functional.mse_loss(T, t_true)
        l_soft, parts = soft(x, t, T, d)
        loss = l_data + 0.5 * l_soft
        loss.backward()
        optimizer.step()
        data_loss.append(float(l_data.detach()))
        violation.append(float(parts["violation"].detach()))

    # ---- Hard PCML: KKT projection onto the heat-equation manifold. ---------
    n_lambda = dae.spec.n_differential + dae.spec.n_inequality
    pcml = PCMLModule(
        constraints=dae, backbone=backbone, input_dim=2,
        n_output=1, n_deriv=4, n_lambda=n_lambda, eta=1.0, max_newton_iter=12,
    )
    pcml.update_activation(0.0)  # force hard mode

    inputs = torch.cat([x, t], dim=-1)
    T_hat, d_hat = _derivatives(backbone, inputs)
    lam_hat = torch.zeros(n_points, n_lambda)
    viol_before = float(dae.violation(x, t, T_hat.detach(), d_hat.detach()))
    _, _, info = pcml(x, t, T_hat.detach(), d_hat.detach(), lam_hat)
    viol_after = float(info["violation"])

    # ---- Figure: soft-training curves + hard-projection violation drop. -----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Hard PCML on the 1-D heat equation (real x, t, derivatives)")

    it = range(steps)
    axes[0].semilogy(it, data_loss, label="data loss")
    axes[0].semilogy(it, violation, label="constraint violation", linestyle="--")
    axes[0].set_title("(a) soft-PCML training")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss / violation (log)")
    axes[0].legend()

    axes[1].bar(
        ["soft (before)", "hard (after KKT)"],
        [max(viol_before, 1e-12), max(viol_after, 1e-12)],
        color=["tab:orange", "tab:green"],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("(b) constraint violation: soft vs hard projection")
    axes[1].set_ylabel("mean |heat-eq residual| (log)")

    fig.tight_layout()
    if show:
        fig.savefig("pcml_heat_diffusion.png", dpi=120)
        if matplotlib.get_backend().lower() != "agg":
            plt.show()

    return {
        "data_loss": data_loss,
        "violation": violation,
        "final_data_loss": data_loss[-1] if data_loss else 0.0,
        "final_soft_violation": violation[-1] if violation else 0.0,
        "violation_before_projection": viol_before,
        "violation_after_projection": viol_after,
        "steps": steps,
        "figure": fig,
    }


def main() -> None:
    """Run the demo with the diagnostic figure displayed and saved."""
    out = run(steps=250, show=True)
    print(f"final data loss        = {out['final_data_loss']:.3e}")
    print(f"final soft violation   = {out['final_soft_violation']:.3e}")
    print(f"violation before KKT   = {out['violation_before_projection']:.3e}")
    print(f"violation after  KKT   = {out['violation_after_projection']:.3e}")


if __name__ == "__main__":
    main()
