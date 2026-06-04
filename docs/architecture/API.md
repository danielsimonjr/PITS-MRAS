# PITS-MRAS Public API Reference

PITS-MRAS (Physics-Informed Time-Series Model-Reference Adaptive Systems) is a
unified framework merging Physics-Informed Neural Networks (PINNs), time-series
deep learning, and Model-Reference Adaptive Control (MRAS).

A single `import pits_mras` exposes a **flat public API** of 17 top-level
symbols (the package's `__all__`). All are re-exported at the package root, so
`pits_mras.PITNN`, `pits_mras.MRASController`, etc. work directly. The 17
symbols, grouped by area:

| Area | Symbols |
|------|---------|
| **Models** | `PITNN`, `QuadraticCritic`, `LagrangianMultiplierHead` |
| **Controllers** | `MRASController`, `LinearReferenceModel`, `CLFCBFSafetyFilter` |
| **Constraints / PCML** | `PhysicsConstraints`, `ConstraintSpec`, `MechanicalDAE`, `HeatConductionDAE`, `SoftPCMLLoss`, `TaylorNeighborhoodApproximation`, `KKTProjectionLayer`, `PCMLModule` |
| **Training** | `pretrain_pitnn`, `cotraining_loop` |
| **Inference** | `RealtimeInferenceEngine` |

`__version__` is `"0.4.0"`.

> Note on the configuration layer: `PITSMRASConfig` and its seven sub-config
> dataclasses (`config.py`) are not in the package-level `__all__`, but they are
> the canonical way to parameterize the models and training loops and are
> documented below. Import them from `pits_mras.config`. The third training
> function, `train_irl_critic`, lives in `pits_mras.training` (re-exported from
> the subpackage, not the package root) and is documented in the Training
> section.

All tensor shapes below use `batch` for the leading batch dimension. Unless
noted, classes derive from `torch.nn.Module` and the documented `forward`
method is invoked by calling the instance.

---

## Models

### `PITNN`

`pits_mras.models.pitnn.PITNN` — Physics-Informed Temporal Neural Network, the
core dynamics model. Takes a sliding window of `(state, control)` history plus
the current tracking error and outputs a port-Hamiltonian dynamics prediction
`f_hat`. Pipeline: input normalization + embedding → causal (forward-only) LSTM
encoder → `PhysicsInformedAttention` → `PortHamiltonianDecoder`.

```python
PITNN(
    net_cfg: NetworkConfig,
    phys_cfg: PhysicsConfig,
    lagrangian_head: Optional[nn.Module] = None,
)
```

`output_dim` is `net_cfg.output_dim` and `n_q` (generalized coords) is
`phys_cfg.n_generalized_coords`; the canonical system assumes
`output_dim == 2 * n_q`. The optional `lagrangian_head` (a
`LagrangianMultiplierHead`) makes the forward pass also emit `lam_hat` (KKT
warm-start multipliers).

**Key methods:**

```python
forward(
    x_hist: Tensor,    # [batch, T, input_dim]  plant-state history
    u_hist: Tensor,    # [batch, T, input_dim]  control history
    x_p_curr: Tensor,  # [batch, input_dim]     current plant state
    u_curr: Tensor,    # [batch, control_dim]   current control
    e_curr: Tensor,    # [batch, e_dim]         current tracking error
    e_hist: Tensor,    # [batch, T, e_dim]      error history
) -> Dict[str, Tensor]
```

Returns a dict with keys:
`f_hat`, `H_val`, `context`, `alpha`, `h_enc`, `P_diss`, `energy_loss`,
`attn_reg_loss` (monitoring keys). When a `lagrangian_head` is attached, an
additional `lam_hat` key is included. (The redundant `f`/`H` aliases of
`f_hat`/`H_val` were removed in v0.3.1.)

```python
update_normalization(x_data: Tensor) -> None   # update running mean/std buffers from a data batch
normalize(x: Tensor) -> Tensor                  # apply running-statistic normalization
```

### `PortHamiltonianDecoder`

`pits_mras.models.decoders.PortHamiltonianDecoder` — full port-Hamiltonian
decoder enforcing `f_hat = J(q) grad_H - [0; R_theta(q)(dH/dp)] + B(x_p) u +
W_corr c_t`, with `J = -Jᵀ` (skew-symmetric), `R_theta ⪰ 0`, `H_theta > 0`.
(Used internally by `PITNN`; exported from `pits_mras.models` but not in the
package-level `__all__`.)

```python
PortHamiltonianDecoder(
    n_q: int,                            # generalized coordinate dimension
    context_dim: int,                    # dimension of attention context c_t
    output_dim: int,                     # full output dim (== 2 * n_q for [q, p])
    hamiltonian_hidden: int = 64,
    dissipation_hidden: int = 32,
    use_position_dependent_J: bool = False,
)
```

**Key methods:**

```python
forward(
    q: Tensor,      # [batch, n_q]        generalized positions
    p: Tensor,      # [batch, n_q]        generalized momenta
    q_dot: Tensor,  # [batch, n_q]        velocity (kept for compat; unused in dissipation)
    u: Tensor,      # [batch, control_dim] control input
    c_t: Tensor,    # [batch, context_dim] attention context
) -> tuple[Tensor, Tensor, Tensor, Tensor]
```

Returns `(f_hat, H_val, P_diss, energy_loss)`:
`f_hat` `[batch, 2*n_q]`, `H_val` `[batch, 1]`, `P_diss` `[batch]` (dissipated
power ≥ 0), `energy_loss` (scalar port-Hamiltonian energy residual).

```python
get_J(q: Tensor) -> Tensor   # batched interconnection matrix [batch, 2*n_q, 2*n_q], always skew-symmetric
```

### `QuadraticCritic`

`pits_mras.models.critic.QuadraticCritic` — linear-in-parameters quadratic
value-function approximator `V̂(e) = W_cᵀ φ(e)` over the upper-triangular
Kronecker basis, so `V̂(e) = eᵀP̂e` with `P̂` symmetric by construction (the LQR
limit `P̂ → P_CARE` is exactly representable). An optional MLP residual extends it
to the nonlinear regime.

```python
QuadraticCritic(
    state_dim: int,
    nonlinear_residual: bool = False,
    residual_hidden: int = 32,
)
```

**Key methods:**

```python
forward(e: Tensor) -> Tensor                 # V̂(e), shape [batch]
gradient(e: Tensor) -> Tensor                # ∇_e V̂ via autograd (= the costate), shape [batch, state_dim]
extract_P() -> Tensor                        # reconstruct symmetric P̂, shape [state_dim, state_dim]
set_P(P: Tensor) -> None                     # write a symmetric P̂ into W_c (LQR/CARE warm-start)
positivity_loss() -> Tensor                  # ReLU(-λ_min(P̂)), scalar
```

### `CostateHead`

`pits_mras.models.critic.CostateHead` — costate / optimal-control head enforcing
Identity 2 (PMP costate = critic gradient) by construction. Implements
`λ̂ = ∂V̂/∂e` and `u* = -½ R⁻¹Bᵀλ̂`. (Used internally by `MRASController`;
exported from `pits_mras.models`, not in the package-level `__all__`.)

```python
CostateHead(
    critic: QuadraticCritic,
    R_inv: Tensor,        # [control_dim, control_dim]
    B: Tensor,            # [state_dim, control_dim]
    half_grad: bool = True,
)
```

`half_grad=True` (default) applies the ½ so `u_opt` recovers `-Ke`; `half_grad=False`
omits it (literal §3.3 text).

**Key method:**

```python
forward(e: Tensor) -> tuple[Tensor, Tensor]
```

Returns `(lambda_hat, u_optimal)`: `lambda_hat` `[batch, state_dim]` (the costate
`∇V̂`, always un-scaled), `u_optimal` `[batch, control_dim]`.

### `LagrangianMultiplierHead`

`pits_mras.models.lagrangian_head.LagrangianMultiplierHead` — predicts the KKT
warm-start multipliers `lambda_hat` from a context vector (DAE-HardNet
`Y_hat = [y_hat, lambda_hat, d_hat]`). Equality/differential multipliers are
unconstrained; inequality multipliers are passed through `Softplus` for
non-negativity (KKT dual feasibility).

```python
LagrangianMultiplierHead(
    context_dim: int,      # dimension of the input context c_t
    n_lambda_eq: int,      # equality + differential multipliers (any sign)
    n_lambda_ineq: int,    # inequality multipliers (non-negative)
    hidden_dim: int = 32,
)
```

**Key method:**

```python
forward(context: Tensor) -> Tensor   # lambda_hat, shape [batch, n_lambda_eq + n_lambda_ineq]
```

---

## Controllers

### `LinearReferenceModel`

`pits_mras.controllers.reference_models.LinearReferenceModel` — Hurwitz linear
reference model `ẋ_m = A_m x_m + B_m r`, `y_m = C_m x_m`. On construction it
asserts `A_m` is Hurwitz, solves the Lyapunov equation `A_mᵀP + P A_m = -Q` for
the CLF/value matrix `P`, and runs Kleinman policy iteration for the LQR optimum
`(P_opt, K_opt)`. All matrices are stored as float32 buffers.

```python
LinearReferenceModel(
    A_m: np.ndarray,   # [n, n] reference dynamics (must be Hurwitz)
    B_m: np.ndarray,   # [n, m] control input matrix
    C_m: np.ndarray,   # [p, n] output matrix
    Q: np.ndarray,     # [n, n] state-cost / Lyapunov RHS (PD)
    R: np.ndarray,     # [m, m] control-cost (PD)
)
```

Stored buffers include: `A_m`, `B_m`, `C_m`, `Q`, `R`, `R_inv`, `P`, `P_opt`,
`K_opt`.

**Key methods:**

```python
reset(batch: int = 1) -> Tensor                       # zero reference state x_m = 0, shape [batch, n]
step(x_m: Tensor, r: Tensor, dt: float) -> Tensor     # forward-Euler step; x_m [batch, n], r [batch, m] -> [batch, n]
output(x_m: Tensor) -> Tensor                         # y_m = C_m x_m, shape [batch, p]
```

### `CLFCBFSafetyFilter`

`pits_mras.controllers.safety.CLFCBFSafetyFilter` — closed-form CLF-CBF safety
filter (Identity 3). The same `P` certifies both stability (CLF `V = eᵀPe`) and
safety (CBF `h = c - eᵀPe`, safe set `eᵀPe ≤ c`). Uses a closed-form
minimum-norm projection (no QP solver).

```python
CLFCBFSafetyFilter(
    P: Tensor,                  # [n, n] Lyapunov/CBF matrix
    A_m: Tensor,                # [n, n] reference dynamics
    B_ctrl: Tensor,             # [n, m] control input matrix
    safety_margin: float = 10.0,  # c: ellipsoid size
    decay_rate: float = 1.0,      # gamma: class-K rate
)
```

**Key methods:**

```python
forward(e: Tensor, u_nom: Tensor) -> Tuple[Tensor, Tensor, Tensor]
```

Args: `e` `[batch, n]` tracking error, `u_nom` `[batch, m]` nominal control.
Returns `(u_safe, h_e, slack)`: `u_safe` `[batch, m]` filtered control, `h_e`
`[batch]` CBF value (>0 = safe), `slack` `[batch]` correction magnitude (0 =
filter inactive).

```python
cbf_constraint_loss(e: Tensor, u: Tensor) -> Tensor   # mean(ReLU(-h(e))), scalar training penalty
```

### `MRASController`

`pits_mras.controllers.mras.MRASController` — adaptive controller fusing
classical MRAS with the actor-critic upgrade. Control law
`u = u_fb(e) + K_ff r + compensator(x_plant)`, where `u_fb` is the costate-head
optimal control `-½R⁻¹Bᵀ∇V̂ = -R⁻¹BᵀP̂e` (Identity 2). The critic is warm-started
to `reference_model.P_opt` at construction, so `u_fb` equals the LQR gain `-K_opt e`
initially and adapts thereafter.

```python
MRASController(
    reference_model: LinearReferenceModel,
    state_dim: int,
    control_dim: int,
    ref_dim: int,
    plant_dim: int,
    use_safety_filter: bool = True,
)
```

**Key methods:**

```python
setup_safety_filter(safety_margin: float = 10.0, decay_rate: float = 1.0) -> None
```
Instantiate the CBF filter from the critic's current `P` (must be called before
the safety filter activates in `forward`).

```python
forward(
    e: Tensor,
    r: Tensor,
    x_plant: Tensor,
    apply_safety: bool = True,
) -> Dict[str, Tensor]
```
Returns a dict with `u_nom`, `u`, `lambda_hat` (costate `∇V̂`), and `v_hat`
(value `V̂(e)`); when the safety filter is active, also `h_cbf` and `slack`.
The costate uses `torch.autograd.grad` internally, so callers under
`torch.no_grad()` must wrap this in `torch.enable_grad()`.

```python
lqr_warm_start(Q: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]
```
Solve the CARE so `P̂ = P_opt`; sets `K_fb` and aligns the critic's `P` via
`critic.set_P`. Returns `(P, K)` as float32 tensors.

```python
mras_regressor(e: Tensor, r: Tensor, x_plant: Tensor) -> Tensor
```
Classical MRAS regressor `φ_c = [eᵀ, rᵀ, x_pᵀ]ᵀ`, shape
`[batch, state_dim + ref_dim + plant_dim]`.

```python
dpg_action_value_gradient(e: Tensor, u: Tensor) -> Tensor
```
Action-value gradient `∇_a Q̂(e, u) = R u + BᵀP̂e`, shape `[batch, control_dim]`.

```python
dpg_actor_step(e: Tensor, r: Tensor, x_plant: Tensor, gamma_c: float = 0.1) -> Tensor
```
Deterministic-policy-gradient actor update (Identity 4). Adds the DPG term to
the gradients of the actor params (`K_ff` and `compensator`) via a surrogate
that is back-propagated internally; the caller's optimizer applies the step
afterward. Returns the scalar surrogate (already back-propagated, detached).

---

## Constraints / PCML

### `ConstraintSpec`

`pits_mras.constraints.base.ConstraintSpec` — dataclass of residual-count
metadata for a constraint system.

```python
@dataclass
class ConstraintSpec:
    n_differential: int = 0   # |N_D| differential-equation residuals
    n_equality: int = 0       # |N_E| algebraic equality residuals
    n_inequality: int = 0     # |N_I| inequality residuals
    n_outputs: int = 0        # |N_y| output variables coupled via Taylor
```

### `PhysicsConstraints`

`pits_mras.constraints.base.PhysicsConstraints` — abstract base (`ABC`) for a
plant's governing DAE system. Subclasses implement `differential`, `equality`,
`inequality`; the base provides the shared `violation` aggregate. All methods
operate batch-first.

```python
@property
@abstractmethod
spec -> ConstraintSpec

@abstractmethod
differential(x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor   # D = 0, [batch, n_differential]
@abstractmethod
equality(x: Tensor, t: Tensor, y: Tensor) -> Tensor                  # h = 0, [batch, n_equality]
@abstractmethod
inequality(x: Tensor, t: Tensor, y: Tensor) -> Tensor                # g (<= 0 feasible), [batch, n_inequality]

violation(x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor      # count-weighted mean-abs violation (scalar)
```

### `MechanicalDAE`

`pits_mras.constraints.mechanical.MechanicalDAE` — Euler-Lagrange DAE
constraints with optional holonomic constraints. State `y = [q, q_dot]`;
derivative variables `d = [q_dot, q_ddot]` (or `[q_dot, q_ddot, lambda]` when
holonomic). Subclass of `PhysicsConstraints`.

```python
MechanicalDAE(
    n_joints: int,
    n_holonomic: int,
    inertia_fn: Callable[[Tensor], Tensor],            # q -> [batch, n, n] M(q)
    coriolis_fn: Callable[[Tensor, Tensor], Tensor],   # (q, q_dot) -> [batch, n]
    gravity_fn: Callable[[Tensor], Tensor],            # q -> [batch, n] G(q)
    actuator_fn: Callable[[Tensor], Tensor],           # q -> [batch, n, m_ctrl] B(q)
    constraint_fn: Optional[Callable[[Tensor], Tensor]] = None,  # q -> [batch, m, n] J(q)
    q_bounds: Optional[Tuple[Tensor, Tensor]] = None,  # (q_min, q_max), each [n]
    u_bounds: Optional[Tuple[Tensor, Tensor]] = None,  # (u_min, u_max)
)
```

Implements:
- `differential(x, t, y, d)` → `[batch, n_differential]` — EOM `M q̈ + C + G`,
  plus holonomic `Ψ = J q` and `J q̇` blocks (and `-Jᵀλ` in the EOM) when
  constrained.
- `equality(x, t, y)` → `[batch, 0]` (no separate algebraic equality residual).
- `inequality(x, t, y)` → `[batch, 2n]` joint limits `[q_min - q, q - q_max]`
  (or `[batch, 0]` if `q_bounds is None`).

### `HeatConductionDAE`

`pits_mras.constraints.thermal.HeatConductionDAE` — 1-D transient heat
conduction `dT/dt = alpha · d²T/dx²` with operational temperature bounds.
Derivative-variable layout `d = [dT_dx, dT_dt, d2T_dx2, d2T_dt2]`. Subclass of
`PhysicsConstraints`.

```python
HeatConductionDAE(
    alpha: float,
    T_min: float = 15.0,
    T_max: float = 35.0,
)
```

Implements:
- `differential(x, t, y, d)` → `[batch, 1]` — `dT/dt - alpha · d2T/dx2`.
- `equality(x, t, y)` → `[batch, 0]`.
- `inequality(x, t, y)` → `[batch, 2]` — `[T_min - T, T - T_max]`.

### `SoftPCMLLoss`

`pits_mras.models.pcml.SoftPCMLLoss` — soft physics-constraint loss (Patel et
al. 2022):
`L = λ_diff·||D||² + λ_eq·||h||² + λ_ineq·||ReLU(g)||²`.

```python
SoftPCMLLoss(
    constraints: PhysicsConstraints,
    lambda_diff: float = 1.0,
    lambda_eq: float = 1.0,
    lambda_ineq: float = 0.5,
)
```

**Key method:**

```python
forward(x: Tensor, t: Tensor, y_pred: Tensor, d_pred: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]
```
Returns `(total, breakdown)` where `breakdown` has keys `diff`, `eq`, `ineq`,
`violation`.

### `TaylorNeighborhoodApproximation`

`pits_mras.models.pcml.TaylorNeighborhoodApproximation` — multi-point
neighborhood approximation (DAE-HardNet §3) converting differential operators
into algebraic derivative variables `d` for the KKT projection.

```python
TaylorNeighborhoodApproximation(
    backbone: nn.Module,
    input_dim: int,
    delta: float = 0.01,    # recommended 1e-3 .. 0.1
    order: int = 1,
)
```

**Key method:**

```python
forward(inputs: Tensor, derivatives: Tensor) -> Tensor   # [batch, output_dim]
```
`inputs` `[batch, input_dim]` is the `(x, t)` point; `derivatives` holds
first-order `d_i` in the first `input_dim` columns and (if `order >= 2`)
second-order `d_ii` in the next `input_dim` columns.

### `KKTProjectionLayer`

`pits_mras.models.pcml.KKTProjectionLayer` — differentiable KKT projection onto
the DAE constraint manifold (DAE-HardNet §3.1). Solves the minimum-distance
problem by Newton iteration on the KKT system with Fischer-Burmeister
complementarity; gradients flow via a single implicit-function-theorem step.

```python
KKTProjectionLayer(
    constraints: PhysicsConstraints,
    n_output: int,
    n_deriv: int,
    newton_step: float = 1.0,
    max_newton_iter: int = 10,
    newton_tol: float = 1e-6,
    reg: float = 1e-8,
)
```

**Key method:**

```python
forward(x: Tensor, t: Tensor, y_hat: Tensor, d_hat: Tensor, lam_hat: Tensor) -> Tuple[Tensor, Tensor, Tensor]
```
Projects `(y_hat, d_hat)` onto the manifold; returns `(y_tilde, d_tilde, lam_tilde)`.

### `PCMLModule`

`pits_mras.models.pcml.PCMLModule` — unified PCML wrapper managing the soft and
hard modes (PCML Addendum §2.2). Soft mode (pre-training) returns the
unconstrained prediction and `SoftPCMLLoss`; once the backbone data loss drops
below `eta`, `update_activation` flips to hard mode, where the prediction is
projected onto the constraint manifold.

```python
PCMLModule(
    constraints: PhysicsConstraints,
    backbone: nn.Module,
    input_dim: int,
    n_output: int,
    n_deriv: int,
    n_lambda: int,
    lambda_soft: float = 1.0,
    omega: float = 1.0,
    delta: float = 0.01,
    taylor_order: int = 1,
    eta: float = 0.01,
    newton_step: float = 1.0,
    max_newton_iter: int = 10,
)
```

**Key methods and properties:**

```python
update_activation(current_data_loss: float) -> bool   # True exactly on the call that flips to hard mode
mode -> str          # property: "hard" or "soft"
n_deriv -> int       # property: projection.n_d

forward(
    x: Tensor,
    t: Tensor,
    y_hat: Tensor,
    d_hat: Tensor,
    lam_hat: Tensor,
    y_true: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]
```
Returns `(y_pcml, pcml_loss, info)` for the active mode. In hard mode `info` has
keys `mode` (`"hard"`), `violation`, `data`, `deriv`; in soft mode `mode`
(`"soft"`) plus the `SoftPCMLLoss` breakdown (`diff`, `eq`, `ineq`,
`violation`).

---

## Training

### `pretrain_pitnn`

`pits_mras.training.pretrain.pretrain_pitnn` — three-stage physics-informed
pre-training curriculum (Algorithm 2). Updates the `PITNN` in place.

```python
pretrain_pitnn(
    pitnn: PITNN,
    cfg: PITSMRASConfig,
    *,
    epochs: int | None = None,        # default cfg.training.pretrain_epochs
    batch_size: int | None = None,    # default cfg.training.pretrain_batch_size
    lr: float | None = None,          # default cfg.training.pretrain_lr
    f_target_fn: FTargetFn | None = None,   # regression target f(x, u); default stable linear surrogate
    epsilon_tol: float = 1e3,
    history_length: int = 8,
    seed: int | None = None,          # default cfg.training.seed
) -> dict[str, list[float]]
```

`FTargetFn = Callable[[Tensor, Tensor], Tensor]`. Returns a history dict mapping
metric names to per-epoch lists: `total_loss`, `physics_loss`, `data_loss`,
`temporal_loss`, `lambda_data`, `lambda_temp`.

Two helper schedule functions are also exported from the module (not at the
package root): `data_weight_schedule(epoch, stage1_epochs, stage2_epochs) -> float`
and `temporal_weight_schedule(epoch, stage2_epochs, lambda_temp_final, stage1_epochs=1000) -> float`.

### `cotraining_loop`

`pits_mras.training.cotrain.cotraining_loop` — closed-loop actor-critic
co-training (Algorithm 3 extended, §8.2). Trains the `PITNN` and the
controller's critic in tandem on synthetic trajectories.

```python
cotraining_loop(
    pitnn: PITNN,
    controller: MRASController,
    ref_model: LinearReferenceModel,
    cfg: PITSMRASConfig,
    *,
    n_episodes: int | None = None,    # default cfg.training.n_episodes
    n_steps: int | None = None,       # default int(sim_duration / dt)
    batch_size: int = 8,
    pitnn_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    irl_window: int = 4,
    history_length: int = 8,
    seed: int | None = None,
    pcml_module: PCMLModule | None = None,
) -> dict[str, list[float]]
```

Returns a metrics dict with per-step lists: `irl_loss`, `hjb_loss`,
`positivity_loss`, `cbf_loss`, `total_loss`, `running_cost`
(and `pcml_loss` when a `pcml_module` is supplied). Sets up the controller's CBF
filter from the critic if requested and not yet attached.

### `train_irl_critic`

`pits_mras.training.irl_trainer.train_irl_critic` — offline batch least-squares
IRL critic trainer (§8.3). Fits the critic's `P` from a fixed batch of synthetic
optimal-closed-loop trajectories using the integral-RL Bellman identity. Updates
the critic in place via `set_P`. Re-exported from `pits_mras.training`.

```python
train_irl_critic(
    critic: QuadraticCritic,
    ref_model: LinearReferenceModel,
    *,
    n_trajectories: int = 64,
    traj_len: int = 40,
    window_size: int = 5,
    dt: float = 0.01,
    max_iters: int = 50,
    tol: float = 0.01,             # relative-error stop ||P_hat - P_opt|| / ||P_opt||
    seed: int = 0,
) -> tuple[Tensor, bool, int]
```

Returns `(P_hat, converged, n_iters)`.

---

## Inference

### `RealtimeInferenceEngine`

`pits_mras.inference.realtime.RealtimeInferenceEngine` — thread-safe real-time
closed-loop control engine (§9.1). Wraps a `PITNN`, an `MRASController`, and a
`LinearReferenceModel`; the `step` method runs one control cycle (measure →
reference step → tracking error → controller → CBF filter → history update). It
puts both wrapped modules into `eval()` mode at construction and guards `step`
with a `threading.Lock` so a fast control thread can call it while a slower
adaptation thread mutates shared parameters.

```python
RealtimeInferenceEngine(
    pitnn: PITNN,
    controller: MRASController,
    ref_model: LinearReferenceModel,
    horizon: int = 50,
    device: str = "cpu",
    pcml_module: Optional[PCMLModule] = None,
    pcml_projection_tolerance: float = 1e-5,
)
```

**Key method:**

```python
@torch.no_grad()
step(x_p: Tensor, r: Tensor, dt: float = 0.01) -> Dict[str, Any]
```
Args: `x_p` `[state_dim]` current plant state (no batch dim), `r` `[ref_dim]`
reference command, `dt` integration timestep. Returns a dict with `u_safe`
`[control_dim]`, `e` `[state_dim]`, `v_hat` (scalar), `h_cbf` (scalar), `f_hat`
`[output_dim]`, `cbf_active` (`bool`), and `pcml_violation` (`float`).

---

## Configuration

`pits_mras.config.PITSMRASConfig` is the master configuration — the single
object passed to the models and training loops. It aggregates seven sub-config
dataclasses, each constructed by default via `field(default_factory=...)`:

```python
@dataclass
class PITSMRASConfig:
    network: NetworkConfig
    physics: PhysicsConfig
    mras: MRASConfig
    safety: SafetyConfig
    losses: LossConfig
    training: TrainingConfig
    pcml: PCMLConfig

    @classmethod
    def from_yaml(cls, path: str) -> "PITSMRASConfig"   # overlay nested fields on defaults
    def to_yaml(self, path: str) -> None                # serialize all nested fields to YAML
```

### `NetworkConfig` — PITNN architecture

| Field | Default | Meaning |
|-------|---------|---------|
| `input_dim` | `10` | state + control dimension |
| `hidden_dim` | `128` | LSTM hidden size |
| `output_dim` | `4` | dynamics prediction dimension |
| `lstm_layers` | `2` | causal LSTM depth |
| `attention_heads` | `4` | physics-informed attention heads |
| `memory_horizon` | `50` | `T`: history steps retained |
| `embedding_dim` | `64` | state/control embedding width |

### `PhysicsConfig` — port-Hamiltonian decoder dims

| Field | Default | Meaning |
|-------|---------|---------|
| `n_generalized_coords` | `2` | `n_q` (positions) |
| `hamiltonian_hidden` | `64` | width of `H_θ` network |
| `dissipation_hidden` | `32` | width of `L_θ` network |
| `use_position_dependent_J` | `False` | set `True` for nonholonomic systems |

### `MRASConfig` — classical MRAS + IRL/actor-critic

| Field | Default | Meaning |
|-------|---------|---------|
| `state_dim` | `4` | dim of tracking error `e` |
| `control_dim` | `2` | dim of control `u` |
| `A_m`, `B_m`, `C_m` | `None` | reference-model matrices (nested lists) |
| `Q_cost`, `R_cost` | `None` | LQR cost matrices (nested lists) |
| `gamma_mras` | `0.1` | classical MRAS adaptation rate |
| `adapt_rate_theta` | `1e-4` | plant-model learning rate |
| `adapt_rate_controller` | `1e-3` | controller learning rate |
| `irl_window_size` | `50` | `T` for IRL Bellman integral window |
| `use_irl_critic` | `True` | enable integral-RL critic update |

### `SafetyConfig` — CLF-CBF-QP filter

| Field | Default | Meaning |
|-------|---------|---------|
| `enable_cbf` | `True` | enable the CBF filter |
| `safety_margin` | `10.0` | `c` in `h(e) = c - eᵀPe` |
| `cbf_decay_rate` | `1.0` | `γ` in the CBF constraint |

### `LossConfig` — unified total-loss weights

Main weights: `lambda_physics=1.0`, `lambda_temporal=0.5`, `lambda_stability=2.0`,
`lambda_data=1.0`, `lambda_irl=1.0`, `lambda_hjb=0.0` (opt-in; `>0` enables the
HJB critic regularizer), `lambda_pcml=1.0`.
Physics sub-weights: `lambda_energy=1.0`, `lambda_pde=1.0`, `lambda_bc=0.5`,
`lambda_sym=0.2`.
(The orphaned `lambda_adjoint` / `alpha_attn` / `alpha_smooth` / `mu_lyap` /
`beta_param` / `lambda_delta_u` weights were removed in v0.4.1 — they were
unconsumed; the corresponding sub-loss classes carry their own weights.)

### `TrainingConfig` — schedule (Algorithms 2 & 3)

| Field | Default | Meaning |
|-------|---------|---------|
| `pretrain_epochs` | `5000` | pre-training epochs |
| `pretrain_batch_size` | `64` | collocation batch size |
| `pretrain_lr` | `1e-3` | pre-train Adam LR |
| `stage1_epochs` | `1000` | physics-only stage |
| `stage2_epochs` | `2000` | data-physics balance stage |
| `n_episodes` | `1000` | co-training episodes |
| `sim_duration` | `10.0` | `T_sim` (seconds) |
| `dt` | `0.01` | `Δt` (seconds) |
| `device` | `"cuda"` if available else `"cpu"` | torch device |
| `seed` | `42` | RNG seed |
| `log_every` | `100` | logging interval |
| `checkpoint_every` | `500` | checkpoint interval |

### `PCMLConfig` — Physics-Constrained ML module

Soft mode: `lambda_soft_diff=1.0`, `lambda_soft_eq=1.0`, `lambda_soft_ineq=0.5`.
Hard mode (DAE-HardNet): `omega=1.0`, `eta=0.01`, `delta=0.01`, `taylor_order=1`,
`newton_step=1.0`, `max_newton_iter=10`, `pcml_projection_tolerance=1e-5`.
Constraint selection: `constraint_type="mechanical"` (`"mechanical" | "thermal"`),
`n_joints=2`, `n_holonomic=0`, `q_bounds=None`, `thermal_alpha=1.0`, `T_min=15.0`,
`T_max=35.0`.

---

## End-to-End Usage

A minimal pipeline (pretrain → controller → cotrain → realtime), consistent with
the real signatures above:

```python
import numpy as np
import torch

from pits_mras import (
    PITNN, MRASController, LinearReferenceModel,
    pretrain_pitnn, cotraining_loop, RealtimeInferenceEngine,
)
from pits_mras.config import PITSMRASConfig

# 1. Configuration (defaults are fine for a smoke run).
cfg = PITSMRASConfig()

# 2. Build the dynamics model from the network + physics sub-configs.
pitnn = PITNN(cfg.network, cfg.physics)

# 3. Physics-informed pre-training (Algorithm 2). Returns a history dict.
history = pretrain_pitnn(pitnn, cfg, epochs=2, batch_size=8)

# 4. Hurwitz reference model (state_dim = 4, control_dim = 2 to match defaults).
n, m = cfg.mras.state_dim, cfg.mras.control_dim
A_m = -np.eye(n)                       # Hurwitz
B_m = np.zeros((n, m)); B_m[:m, :] = np.eye(m)
C_m = np.eye(n)
Q = np.eye(n)
R = np.eye(m)
ref_model = LinearReferenceModel(A_m, B_m, C_m, Q, R)

# 5. MRAS controller (critic warm-started to P_opt). plant_dim == input_dim here.
controller = MRASController(
    reference_model=ref_model,
    state_dim=n,
    control_dim=m,
    ref_dim=m,
    plant_dim=cfg.network.input_dim,
)

# 6. Closed-loop co-training (Algorithm 3 extended). Returns a metrics dict.
metrics = cotraining_loop(pitnn, controller, ref_model, cfg,
                          n_episodes=1, n_steps=5, batch_size=4)

# 7. Real-time deployment loop.
engine = RealtimeInferenceEngine(pitnn, controller, ref_model, horizon=8)
x_p = torch.zeros(cfg.network.input_dim)   # current plant state [state_dim]
r = torch.zeros(m)                         # reference command [ref_dim]
out = engine.step(x_p, r, dt=cfg.training.dt)
print(out["u_safe"], out["v_hat"], out["cbf_active"])
```
