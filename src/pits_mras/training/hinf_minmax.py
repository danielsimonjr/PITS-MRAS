r"""Neural H-infinity adversarial min-max training loop (ROADMAP #1, capstone).

Owning phase: Phase 5 (Training Pipelines).

Three-network approximate-dynamic-programming (ADP) solver for the
Hamilton-Jacobi-Isaacs (HJI) equation of the linear H-infinity game

.. math::
    \min_u \max_w \int_0^\infty \big(e^\top Q e + u^\top R u
        - \gamma^2 \lVert w\rVert^2\big)\,dt,
    \qquad \dot e = A e + B u + D w.

This is the LEARNED counterpart to the analytic GARE core
(:func:`~pits_mras.utils.lyapunov.solve_gare`,
:class:`~pits_mras.models.critic.AdversaryHead`): instead of pinning the
disturbance to the analytic ``w* = gamma^-2 D^T P e``, an INDEPENDENT
:class:`~pits_mras.models.adversary.NeuralAdversary` is co-trained by gradient
ASCENT, so the protagonist + critic must become robust to a learned worst case.

Three networks (the standard three-NN ADP architecture for the HJI):

* **Critic** ``V_hat(e) = e^T P_hat e`` -- :class:`QuadraticCritic`.
* **Protagonist control** ``u = -1/2 R^-1 B^T grad V_hat = -K_hat e`` --
  :class:`CostateHead` (``half_grad=True``). The control is the critic gradient,
  so it descends the value automatically; no separate actor network is needed.
* **Adversary** ``w = NeuralAdversary(e)`` -- an independent MLP, trained by
  ascent.

HJI / Bellman residual driving the critic (the SINGLE source of correctness):

.. math::
    \rho(e) = e^\top Q e + u^\top R u - \gamma^2 \lVert w\rVert^2
            + \nabla V_\text{hat}\cdot(A e + B u + D w).

At the GARE optimum (``P_hat = P*``, ``u = -K* e``, ``w = L* e``) this is
``e^T (A^T P + P A + Q - P M P) e = 0`` for all ``e`` -- the GARE residual --
which is the training-free objective-correctness check in the tests.

Updates (two-timescale -- the standard min-max stabilizer):

* **Critic** minimizes ``E[rho^2]`` (+ optional positivity penalty). Fast.
* **Adversary** MAXIMIZES the value, i.e. ascends the term it controls. We
  realize ascent by MINIMIZING ``-objective`` where the objective for the
  adversary is the value it can inflate; concretely the adversary minimizes
  ``-E[rho_w]`` (negated). Fast-ish (inner player).
* **Protagonist** is the critic gradient, so it improves only as the critic
  improves -- it is the implicit SLOW player. The critic learning rate is the
  pace knob; ``adv_lr`` is set faster than the effective protagonist pace.

Two-timescale rationale (Borkar 1997; Heusel et al. 2017 "TTUR"): the inner
maximizer (adversary) and the value estimate (critic) must adapt faster than the
slow outer minimizer (protagonist) for the min-max iteration to track the
saddle point rather than oscillate. Here ``adv_lr >= critic_lr`` and both are
fast; the protagonist is implicitly slow because it is read off the critic.

States ``e`` are sampled i.i.d. each step (seeded). The loop returns a metrics
dict including the GARE-oracle distance ``||P_hat - P*|| / ||P*||`` so callers
(and tests) can check convergence or, failing tight convergence, a downward
TREND from the warm start.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional

import numpy as np
import torch
from torch import Tensor

from pits_mras.models.adversary import NeuralAdversary
from pits_mras.models.critic import CostateHead, QuadraticCritic
from pits_mras.models.pitnn import PITNN
from pits_mras.utils.linearization import linearize_dynamics
from pits_mras.utils.lyapunov import solve_gare

logger = logging.getLogger(__name__)


def hji_residual(
    critic: QuadraticCritic,
    costate: CostateHead,
    adversary: NeuralAdversary,
    e: Tensor,
    A: Tensor,
    B: Tensor,
    D: Tensor,
    Q: Tensor,
    R: Tensor,
    gamma: float,
) -> Tensor:
    r"""HJI / game-Bellman residual ``rho(e)`` of shape ``[batch]``.

    ``rho = e^T Q e + u^T R u - gamma^2 ||w||^2 + grad V_hat . (A e + B u + D w)``
    with ``u = -1/2 R^-1 B^T grad V_hat`` (the costate head) and
    ``w = adversary(e)``. The critic gradient ``grad V_hat = 2 P_hat e`` is taken
    with ``create_graph=True`` (inside :meth:`CostateHead.forward`) so the residual
    is differentiable w.r.t. the critic AND the adversary parameters.

    Shapes: ``e`` ``[batch, n]``; ``A`` ``[n, n]``; ``B`` ``[n, m]``;
    ``D`` ``[n, n_w]``; ``Q`` ``[n, n]``; ``R`` ``[m, m]``.
    """
    grad_v, u = costate(e)  # grad_v: [batch, n] (= 2 P e), u: [batch, m]
    w = adversary(e)  # [batch, n_w]

    # Running cost e^T Q e + u^T R u - gamma^2 ||w||^2.
    eQe = torch.einsum("bi,ij,bj->b", e, Q, e)
    uRu = torch.einsum("bi,ij,bj->b", u, R, u)
    w_sq = (w * w).sum(dim=-1)
    running = eQe + uRu - (gamma**2) * w_sq

    # Drift: A e + B u + D w  -> [batch, n].
    e_dot = e @ A.T + u @ B.T + w @ D.T
    grad_term = (grad_v * e_dot).sum(dim=-1)  # grad V_hat . e_dot

    return running + grad_term  # [batch]


def hinf_minmax_train(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    gamma: float,
    D: Optional[np.ndarray] = None,
    *,
    n_iters: int = 4000,
    batch_size: int = 256,
    critic_lr: float = 5e-3,
    adv_lr: float = 1e-2,
    adv_steps: int = 1,
    critic_steps: int = 1,
    positivity_weight: float = 1e-2,
    sample_scale: float = 1.0,
    adv_hidden: tuple[int, ...] = (64, 64),
    warm_start_identity: bool = True,
    seed: Optional[int] = None,
    grad_clip: float = 10.0,
    device: Optional[torch.device] = None,
) -> dict[str, object]:
    r"""Co-train critic + protagonist + neural adversary as an HJI min-max game.

    Solves the linear H-infinity game (see module docstring) by three-network ADP
    against the analytic GARE oracle. The critic ``V_hat = e^T P_hat e`` is
    trained to drive the HJI residual to zero; the protagonist control is the
    critic gradient (so it descends the value); the
    :class:`~pits_mras.models.adversary.NeuralAdversary` is trained by ASCENT to
    inflate the value (worst-case disturbance). Two-timescale: the adversary +
    critic adapt fast, the protagonist is implicitly slow (read off the critic).

    The GARE solution ``(P*, K*, L*)`` is computed up front as the ORACLE for the
    returned convergence metrics. The critic is NOT warm-started to ``P*`` (that
    would defeat the convergence test); with ``warm_start_identity`` it begins at
    the :class:`QuadraticCritic` default ``P_hat ~ I`` (away from ``P*``).

    Args:
        A, B, Q, R: system + cost matrices.
        gamma: H-infinity attenuation level (must be GARE-feasible).
        D: disturbance input matrix ``[n, n_w]``; defaults to ``B``.
        n_iters: number of outer min-max iterations.
        batch_size: states ``e`` sampled per iteration.
        critic_lr: Adam learning rate for the critic (fast inner player).
        adv_lr: Adam learning rate for the adversary; ``>= critic_lr`` realizes
            the two-timescale ordering with the slow (implicit) protagonist.
        adv_steps: adversary ascent steps per outer iteration (inner-loop count).
        critic_steps: critic descent steps per outer iteration.
        positivity_weight: weight on ``relu(-lambda_min(P_hat))`` keeping the
            critic PD (a CLF must be PD); 0 disables it.
        sample_scale: std of the ``randn`` state sampler.
        adv_hidden: adversary MLP hidden widths.
        warm_start_identity: if False, randomize the critic away from identity
            (still NOT at ``P*``).
        seed: RNG seed for reproducible sampling + init.
        grad_clip: max grad-norm for both optimizers (0 disables).
        device: torch device (defaults to CPU).

    Returns:
        Metrics dict with:

        * ``residual`` -- per-iter mean ``rho^2`` (list of floats),
        * ``value`` -- per-iter mean ``V_hat`` (list of floats),
        * ``P_dist`` -- per-iter ``||P_hat - P*|| / ||P*||`` (list of floats),
        * ``K_dist`` -- per-iter ``||K_hat - K*|| / ||K*||`` (list of floats),
        * ``adv_dist`` -- per-iter ``||w(e) - L* e|| / ||L* e||`` on the batch,
        * ``P_star`` / ``K_star`` / ``L_star`` -- the GARE oracle (numpy),
        * ``P_hat`` / ``K_hat`` -- the final learned matrices (numpy),
        * ``critic`` / ``adversary`` / ``costate`` -- the trained modules.
    """
    if device is None:
        device = torch.device("cpu")
    if D is None:
        D = B

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    n = A.shape[0]
    n_w = D.shape[1]

    # ── Analytic GARE oracle (float64) for the convergence metrics. ──
    P_star_np, K_star_np, L_star_np = solve_gare(A, B, Q, R, gamma, D)

    if seed is not None:
        torch.manual_seed(seed)

    # float32 tensors for the nets / dynamics.
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    B_t = torch.tensor(B, dtype=torch.float32, device=device)
    D_t = torch.tensor(D, dtype=torch.float32, device=device)
    Q_t = torch.tensor(Q, dtype=torch.float32, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    R_inv_t = torch.tensor(np.linalg.inv(R), dtype=torch.float32, device=device)

    K_star = torch.tensor(K_star_np, dtype=torch.float32, device=device)
    L_star = torch.tensor(L_star_np, dtype=torch.float32, device=device)
    P_star = torch.tensor(P_star_np, dtype=torch.float32, device=device)
    P_star_norm = float(torch.linalg.norm(P_star))
    K_star_norm = float(torch.linalg.norm(K_star))

    # ── Networks. ──
    critic = QuadraticCritic(state_dim=n).to(device)
    if not warm_start_identity:
        # Move away from identity (but NOT to P*): scale the default weights.
        with torch.no_grad():
            critic.W_c.weight.mul_(1.5)
    costate = CostateHead(critic, R_inv=R_inv_t, B=B_t, half_grad=True).to(device)
    adversary = NeuralAdversary(state_dim=n, dist_dim=n_w, hidden=adv_hidden).to(device)

    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    adv_opt = torch.optim.Adam(adversary.parameters(), lr=adv_lr)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    metrics: dict[str, object] = {
        "residual": [],
        "value": [],
        "P_dist": [],
        "K_dist": [],
        "adv_dist": [],
    }
    res_hist: list[float] = metrics["residual"]  # type: ignore[assignment]
    val_hist: list[float] = metrics["value"]  # type: ignore[assignment]
    p_hist: list[float] = metrics["P_dist"]  # type: ignore[assignment]
    k_hist: list[float] = metrics["K_dist"]  # type: ignore[assignment]
    a_hist: list[float] = metrics["adv_dist"]  # type: ignore[assignment]

    def _sample() -> Tensor:
        return sample_scale * torch.randn(batch_size, n, generator=gen, device=device)

    for it in range(n_iters):
        # ── Adversary ASCENT (inner / fast player) ──
        # The adversary inflates the game value; ascent == descend the negated
        # residual contribution. We let it maximize E[rho] (the residual it can
        # push up via the +D w drift and the -gamma^2 ||w||^2 it must trade off),
        # implemented as minimizing -E[rho]. Critic params are frozen here.
        last_w_val = 0.0
        for _ in range(adv_steps):
            e = _sample()
            for p in critic.parameters():
                p.requires_grad_(False)
            rho = hji_residual(critic, costate, adversary, e, A_t, B_t, D_t, Q_t, R_t, gamma)
            adv_loss = -rho.mean()  # ascent on the value
            adv_opt.zero_grad()
            adv_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adversary.parameters(), grad_clip)
            adv_opt.step()
            for p in critic.parameters():
                p.requires_grad_(True)
            last_w_val = float(rho.mean().detach())

        # ── Critic DESCENT (fast player) ──
        last_res = 0.0
        last_val = 0.0
        for _ in range(critic_steps):
            e = _sample()
            for p in adversary.parameters():
                p.requires_grad_(False)
            rho = hji_residual(critic, costate, adversary, e, A_t, B_t, D_t, Q_t, R_t, gamma)
            loss = (rho**2).mean()
            if positivity_weight > 0:
                loss = loss + positivity_weight * critic.positivity_loss()
            critic_opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
            critic_opt.step()
            for p in adversary.parameters():
                p.requires_grad_(True)
            last_res = float((rho**2).mean().detach())
            with torch.no_grad():
                last_val = float(critic(e).mean())

        # ── Metrics (oracle distances). ──
        with torch.no_grad():
            P_hat = critic.extract_P()
            P_dist = float(torch.linalg.norm(P_hat - P_star)) / max(P_star_norm, 1e-12)
            K_hat = R_inv_t @ B_t.T @ P_hat
            K_dist = float(torch.linalg.norm(K_hat - K_star)) / max(K_star_norm, 1e-12)
            e_eval = _sample()
            w_eval = adversary(e_eval)
            w_oracle = e_eval @ L_star.T
            denom = float(torch.linalg.norm(w_oracle))
            adv_dist = float(torch.linalg.norm(w_eval - w_oracle)) / max(denom, 1e-12)

        res_hist.append(last_res)
        val_hist.append(last_val)
        p_hist.append(P_dist)
        k_hist.append(K_dist)
        a_hist.append(adv_dist)
        _ = last_w_val

        if it % max(n_iters // 10, 1) == 0:
            logger.info(
                "hinf_minmax iter %d/%d  res=%.4g P_dist=%.4g K_dist=%.4g adv_dist=%.4g",
                it,
                n_iters,
                last_res,
                P_dist,
                K_dist,
                adv_dist,
            )

    with torch.no_grad():
        P_hat_final = critic.extract_P().cpu().numpy()
        K_hat_final = (R_inv_t @ B_t.T @ critic.extract_P()).cpu().numpy()

    metrics["P_star"] = P_star_np
    metrics["K_star"] = K_star_np
    metrics["L_star"] = L_star_np
    metrics["P_hat"] = P_hat_final
    metrics["K_hat"] = K_hat_final
    metrics["critic"] = critic
    metrics["adversary"] = adversary
    metrics["costate"] = costate
    return metrics


def hinf_minmax_from_dynamics(
    dynamics_fn: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    u0: Tensor,
    Q: np.ndarray,
    R: np.ndarray,
    gamma: float,
    D: Optional[np.ndarray] = None,
    *,
    linearize_backend: Literal["jacrev", "autograd"] = "jacrev",
    **kwargs: object,
) -> dict[str, object]:
    r"""Run the H-infinity min-max loop on a learned/analytic dynamics callable.

    The bridge (ROADMAP integration #6) that lets the neural adversarial min-max
    loop run on ANY continuous-time dynamics ``f(x, u) -> xdot`` instead of fixed
    matrices. It linearizes ``dynamics_fn`` at the operating point ``(x0, u0)``
    via :func:`~pits_mras.utils.linearization.linearize_dynamics`, then drives the
    UNCHANGED :func:`hinf_minmax_train` with the extracted ``(A, B)``.

    Use it to run the robust min-max on a learned model's LOCAL linear dynamics
    (e.g. a one-step plant ``f``), a Koopman ``latent_step`` (treat the latent
    ``z`` as the "state"), or any analytic plant. The linearization is exact for
    affine ``f`` (so a Koopman ``latent_step`` recovers ``(A_z, B_z)`` exactly)
    and a first-order approximation otherwise.

    Args:
        dynamics_fn: callable mapping a single state ``x`` ``[state_dim]`` and a
            single control ``u`` ``[control_dim]`` to ``xdot`` ``[state_dim]``.
        x0: operating-point state, shape ``[state_dim]``.
        u0: operating-point control, shape ``[control_dim]``.
        Q, R: cost matrices on the (linearized) state and control.
        gamma: H-infinity attenuation level (must be GARE-feasible at ``(A, B)``).
        D: disturbance input matrix ``[state_dim, n_w]``; defaults to ``B``.
        linearize_backend: Jacobian engine forwarded to
            :func:`~pits_mras.utils.linearization.linearize_dynamics`. ``"jacrev"``
            (default, unchanged) for pure ``torch.func`` callables; ``"autograd"``
            for callables that use ``torch.autograd.grad`` internally (e.g. a
            ``PITNN`` one-step adapter).
        **kwargs: forwarded verbatim to :func:`hinf_minmax_train` (e.g.
            ``n_iters``, ``batch_size``, ``seed``).

    Returns:
        The :func:`hinf_minmax_train` metrics dict, plus the extracted linear
        model under keys ``"A"`` and ``"B"`` (numpy float64) for transparency.

    Note:
        ``dynamics_fn`` must be a SINGLE-step ``f(x, u)``. Collapsing a
        sequence/history-window model (e.g. ``PITNN.forward``) into a one-step
        operating point is an ADR-level design decision and is the caller's
        responsibility — wrap such a model in a one-step adapter first.
    """
    A_t, B_t = linearize_dynamics(dynamics_fn, x0, u0, backend=linearize_backend)
    A = A_t.detach().cpu().numpy().astype(np.float64)
    B = B_t.detach().cpu().numpy().astype(np.float64)

    metrics = hinf_minmax_train(A, B, Q, R, gamma, D=D, **kwargs)  # type: ignore[arg-type]
    metrics["A"] = A
    metrics["B"] = B
    return metrics


def pitnn_one_step(
    pitnn: PITNN,
    *,
    history: Optional[Tensor] = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    r"""Collapse a sequence ``PITNN`` into a one-step ``f(x, u) -> xdot`` adapter.

    ``PITNN.forward`` is a SEQUENCE model: it consumes a history window (state +
    control + error sequences of shape ``[batch, T, dim]``) plus the current
    state/control/error, and its port-Hamiltonian decoder returns the
    instantaneous state derivative ``f_hat`` of shape ``[batch, output_dim]``
    where ``output_dim == 2 * n_q`` (the canonical ``[q, p]`` block). The neural
    H-infinity min-max loop and :func:`~pits_mras.utils.linearization.linearize_dynamics`
    instead need a one-step ``f(x, u) -> xdot`` on a SINGLE (unbatched) state.

    Operating-point / history convention (the ADR-level design decision)
    --------------------------------------------------------------------
    The returned ``f`` holds a FIXED operating-point CONTEXT and varies ONLY the
    current state ``x`` and control ``u``:

    * **State** ``x`` has dimension ``state_dim = output_dim = 2 * n_q`` -- the
      canonical ``[q, p]`` block that the decoder actually reads (the only part of
      the plant state that affects ``f_hat``), and exactly the dimension of the
      returned derivative, so ``A = df/dx`` is square ``[state_dim, state_dim]``.
    * ``x`` is placed BOTH as ``x_p_curr`` (the decoder's current state) AND as
      the ``[q, p]`` channels of the LATEST entry of the history window. The
      remaining ``input_dim - 2*n_q`` plant channels (if any) and all earlier
      window rows come from the FIXED ``history`` (held constant); they form the
      operating-point context that the LSTM encoder + attention condition on.
    * **Control** ``u`` (dim ``control_dim`` = the wired ``n_q``) is placed as
      ``u_curr`` and as the corresponding channels of the latest control-history
      entry.
    * The current/history tracking error is held at ZERO (the linearization is
      about the tracking operating point ``e = 0``; the error only feeds the
      attention gate, not the derivative directly).

    ``f`` returns the decoder ``f_hat`` for the single sample, squeezed to
    ``[state_dim]``. It is built FUNCTIONALLY (out-of-place ``cat``/``stack``, no
    buffer mutation) and runs UNDER grad (not ``no_grad``), so
    ``torch.func.jacrev`` / ``torch.autograd`` flow through it and
    :func:`linearize_dynamics` works.

    Args:
        pitnn: a constructed :class:`~pits_mras.models.pitnn.PITNN`. Read-only
            here -- no parameters are updated; ``pitnn`` is just evaluated.
        history: optional FIXED operating-point window of shape
            ``[T, input_dim]`` (state/control history; the same tensor seeds both
            the state and control history windows). ``None`` (default) uses a
            single zero step ``T = 1`` -- the simplest neutral operating point.

    Returns:
        A callable ``f(x, u) -> xdot`` mapping a single state ``x``
        ``[state_dim]`` and control ``u`` ``[control_dim]`` to ``xdot``
        ``[state_dim]`` (``= 2 * n_q``), suitable for
        :func:`linearize_dynamics` and hence :func:`hinf_minmax_from_dynamics`.

    Note:
        A learned PITNN is a NONLINEAR model, so this is a FIRST-ORDER (tangent)
        linearization about ``(x, u, e=0)``; it is not a clean LTI system and
        oracle (GARE ``P*``) recovery is NOT expected -- the honest verification
        is finiteness/shape/differentiability/end-to-end runnability.
    """
    n_q = pitnn.n_q
    state_dim = 2 * n_q
    input_dim = pitnn.input_dim
    output_dim = pitnn.output_dim
    e_dim = output_dim  # error dim wired to output dim (see PITNN.__init__)
    control_dim = n_q  # wired control dim (momentum channel)

    if input_dim < state_dim:
        raise ValueError(
            f"PITNN input_dim ({input_dim}) < 2*n_q ({state_dim}); the [q, p] "
            "block does not fit in the plant-state vector."
        )

    # Resolve the fixed operating-point window [T, input_dim] (default: one zero step).
    if history is None:
        hist = torch.zeros(1, input_dim, dtype=pitnn.mu_x.dtype, device=pitnn.mu_x.device)
    else:
        if history.dim() != 2 or history.shape[1] != input_dim:
            raise ValueError(
                f"history must have shape [T, input_dim={input_dim}]; got "
                f"{tuple(history.shape)}."
            )
        hist = history.to(dtype=pitnn.mu_x.dtype, device=pitnn.mu_x.device)
    T = hist.shape[0]
    ref_dtype = hist.dtype
    ref_device = hist.device

    # Fixed "tail" of the plant-state vector beyond the [q, p] block (channels
    # 2*n_q .. input_dim), taken from the last history row -- the operating-point
    # context for those non-canonical channels.
    tail_dim = input_dim - state_dim

    def f(x: Tensor, u: Tensor) -> Tensor:
        x = x.to(dtype=ref_dtype, device=ref_device)
        u = u.to(dtype=ref_dtype, device=ref_device)

        # Current plant state x_p_curr [1, input_dim]: [q, p] = x, tail = fixed.
        if tail_dim > 0:
            tail = hist[-1, state_dim:]
            x_full = torch.cat([x, tail], dim=0)  # [input_dim]
        else:
            x_full = x
        x_p_curr = x_full.unsqueeze(0)  # [1, input_dim]

        # Current control u_curr [1, control_dim].
        u_curr = u.unsqueeze(0)  # [1, control_dim]

        # State history [1, T, input_dim]: fixed window, but the LATEST row's
        # [q, p] channels are overwritten with the current x (functionally).
        if T > 1:
            x_hist = torch.cat(
                [hist[:-1].unsqueeze(0), x_full.unsqueeze(0).unsqueeze(0)], dim=1
            )  # [1, T, input_dim]
        else:
            x_hist = x_full.unsqueeze(0).unsqueeze(0)  # [1, 1, input_dim]

        # Control history [1, T, input_dim]: fixed window with the latest row's
        # leading control_dim channels set to the current u (rest = fixed tail).
        u_last_tail = hist[-1, control_dim:]  # [input_dim - control_dim]
        u_last = torch.cat([u, u_last_tail], dim=0)  # [input_dim]
        if T > 1:
            u_hist = torch.cat(
                [hist[:-1].unsqueeze(0), u_last.unsqueeze(0).unsqueeze(0)], dim=1
            )  # [1, T, input_dim]
        else:
            u_hist = u_last.unsqueeze(0).unsqueeze(0)  # [1, 1, input_dim]

        # Tracking error pinned at the operating point e = 0.
        e_curr = torch.zeros(1, e_dim, dtype=ref_dtype, device=ref_device)
        e_hist = torch.zeros(1, T, e_dim, dtype=ref_dtype, device=ref_device)

        out = pitnn(x_hist, u_hist, x_p_curr, u_curr, e_curr, e_hist)
        return out["f_hat"].squeeze(0)  # [output_dim] == [2*n_q] == [state_dim]

    _ = control_dim  # documented contract; control_dim == n_q by wiring.
    return f


def hinf_minmax_from_pitnn(
    pitnn: PITNN,
    x0: Tensor,
    u0: Tensor,
    Q: np.ndarray,
    R: np.ndarray,
    gamma: float,
    *,
    history: Optional[Tensor] = None,
    D: Optional[np.ndarray] = None,
    **kwargs: object,
) -> dict[str, object]:
    r"""Run the H-infinity min-max loop on a sequence ``PITNN`` (the capstone wiring).

    Thin convenience wrapper that builds a one-step ``f(x, u)`` adapter from
    ``pitnn`` via :func:`pitnn_one_step` (fixed operating-point ``history``;
    varies the current ``[q, p]`` state and control) and forwards it to the
    EXISTING :func:`hinf_minmax_from_dynamics`. The min-max logic is NOT
    duplicated -- this only collapses the sequence model into the one-step
    contract that the bridge already expects.

    See :func:`pitnn_one_step` for the operating-point / history convention and
    the state dimension (``state_dim = 2 * n_q == output_dim``).

    Args:
        pitnn: a constructed :class:`~pits_mras.models.pitnn.PITNN`.
        x0: operating-point state ``[2*n_q]`` (the canonical ``[q, p]`` block).
        u0: operating-point control ``[control_dim]`` (``= n_q`` by wiring).
        Q, R: cost matrices on the (linearized) state ``[2*n_q]`` and control.
        gamma: H-infinity attenuation level (must be GARE-feasible at the
            linearized ``(A, B)``).
        history: optional FIXED operating-point window ``[T, input_dim]`` passed
            to :func:`pitnn_one_step`.
        D: disturbance input matrix ``[2*n_q, n_w]``; defaults to ``B``.
        **kwargs: forwarded verbatim to :func:`hinf_minmax_train` via
            :func:`hinf_minmax_from_dynamics` (e.g. ``n_iters``, ``seed``).

    Returns:
        The :func:`hinf_minmax_from_dynamics` metrics dict, including the
        extracted linear model under keys ``"A"`` / ``"B"``.

    Note:
        A learned PITNN is nonlinear; the linearization is first-order about
        ``(x0, u0, e=0)`` and oracle (GARE ``P*``) recovery is NOT expected (an
        untrained / non-LTI model). Finiteness, shapes, and end-to-end
        runnability are the honest verification.
    """
    f = pitnn_one_step(pitnn, history=history)
    # The port-Hamiltonian decoder differentiates a learned Hamiltonian via
    # torch.autograd.grad inside its forward pass, which torch.func.jacrev
    # forbids; the "autograd" backend (classic double-backward) composes with it.
    return hinf_minmax_from_dynamics(
        f, x0, u0, Q, R, gamma, D=D, linearize_backend="autograd", **kwargs
    )
