# PITS-MRAS: PCML Component Addendum

> **Read alongside:** `PITS_MRAS_Implementation_Plan.md`
> **Seed documents (required reading before implementation):**
> 
> - **Patel et al. (IFAC 2022):** *Physics Constrained Learning in Neural Network based Modeling* — introduces the soft augmentation approach: `MSE_total = MSE_data + λ × MSE_physics`.
> - **Golder, Roy & Hasan (arXiv 2025, DAEHardNet):** *A Physics Constrained Neural Network Enforcing Differential-Algebraic Hard Constraints* — introduces the KKT-projection hard enforcement layer; code at `https://github.com/SOULS-TAMU/DAE-HardNet`.
> 
> **Claude Code: Read both PDFs in `/mnt/project/` before implementing any file in this addendum.**

-----

## §0 · Why PCML and How It Changes the Architecture

The existing plan enforces physics *softly*: the port-Hamiltonian decoder shapes the network output through structural inductive biases (skew-symmetric J, PSD dissipation R=LᵀL), and the L_physics loss penalizes violations of energy conservation, PDE residuals, and boundary conditions. This is the PINNs paradigm of Raissi et al. (2019).

**The fundamental problem,** documented in both seed papers:

1. Soft penalties trade prediction accuracy against physical fidelity — they do not *guarantee* satisfaction.
1. Constraint violations accumulate when surrogate models are chained (e.g., the PITNN feeds into the MRAS controller, which feeds into the IRL critic).
1. Safety-critical applications (aerospace, nuclear, chemical processes) require *provable* first-principles satisfaction.
1. Soft physics losses make the optimization landscape nonconvex and unstable (spectral bias, gradient pathologies).

**The PCML solution** is a two-layer upgrade:

|Mode         |Source           |Mechanism                                                            |Guarantee                                |
|-------------|-----------------|---------------------------------------------------------------------|-----------------------------------------|
|**Soft PCML**|Patel et al. 2022|Augment loss with explicit constraint residuals `λ × MSE_constraints`|Reduced violation, no exact guarantee    |
|**Hard PCML**|DAEHardNet 2025  |KKT projection layer maps f̂ onto the DAE constraint manifold         |**Point-wise exact satisfaction** of DAEs|

Both modes coexist in the implementation. Soft PCML is used during pre-training (Phase 1). Hard PCML activates dynamically once the backbone data loss drops below a threshold η (mimicking DAEHardNet’s `eta` parameter) and is the primary mode during co-training and inference.

### Updated Architecture (PITNN → PCML → Controller)

```
┌─────────────────────────────────────────────────────────────┐
│                     PITNN Forward Pass                       │
│  History → LSTM → PhysicsInformedAttention → c_t            │
│                           ↓                                  │
│             Port-Hamiltonian Decoder                         │
│         [H_θ(q,p), R_θ=LᵀL, J_skew, B_ctrl]               │
│                           ↓                                  │
│          f̂_θ  (soft physics; may violate DAEs)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            PCML Module  (NEW — this addendum)                │
│                                                              │
│  SOFT PATH (pre-training):                                   │
│    L_pcml_soft = λ·‖D(f̂,x,t)‖² + μ·‖h(x,f̂)‖² + ν·ReLU(g) │
│                                                              │
│  HARD PATH (co-training + inference):                        │
│    KKT Projection Layer:                                     │
│      ỹ,d̃ = argmin ½‖y−f̂‖² s.t. D=0, h=0, g≤0, y=M(d)    │
│    Derivative Loss:                                          │
│      L_pcml_hard = MSE(ỹ, f_true) + ω·MSE(d̃, AD(∂ỹ))      │
│                           ↓                                  │
│          f̂_pcml  (hard-constrained, point-wise exact)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│   MRAS Controller (IRL Critic + Costate Head + CBF Filter)  │
│   Uses f̂_pcml instead of f̂_θ for all stability reasoning   │
└─────────────────────────────────────────────────────────────┘
```

-----

## §1 · Mathematical Framework

### 1.1 DAE Problem Setup (from DAEHardNet §2)

For the PITS-MRAS controlled plant with state y(x, t) and known governing physics:

```
D(x, t, y, ∂) = 0      ∀(x,t) ∈ Ω × [0,T]     (differential equations)
h(x, t, y) = 0          ∀(x,t) ∈ Ω × [0,T]     (algebraic equality constraints)
g(x, t, y) ≤ 0          ∀(x,t) ∈ Ω × [0,T]     (inequality constraints, e.g. state bounds)
y(x, t₀) = G_IC(x)                               (initial conditions)
y(x, t)  = G_BC(t)      x ∈ ∂Ω                  (boundary conditions)
```

**In PITS-MRAS specifically**, y = [q; q̇; λ] (generalized coordinates, velocities, constraint forces), and the DAEs are the equations of motion. For a mechanical system with holonomic constraints:

```
D₁: M(q)q̈ + C(q,q̇)q̇ + G(q) − Bᵤ(q)u − Jᵀ(q)λ = 0     (Newton-Euler)
D₂: Ψ(q) = 0                                               (holonomic constraint)
D₃: J(q)q̇ = 0                                             (velocity constraint, derivative of D₂)
h:  λ − [J M⁻¹ Jᵀ]⁻¹[J M⁻¹(Bᵤu − Cq̇ − G) + J̇q̇] = 0  (constraint force — algebraic)
g:  q_min ≤ q ≤ q_max                                      (joint limits)
```

The PITNN currently predicts f̂_θ ≈ [q̈; λ] without guaranteeing D₂=0 or D₃=0. The PCML hard layer projects f̂_θ onto {y | D=0, h=0, g≤0} with minimum Euclidean distance.

### 1.2 Soft PCML Loss (Patel et al. 2022)

From Equations (3)–(4) of the seed paper, the augmented loss is:

```
L_total = MSE_data + λ_soft × MSE_physics
```

where the physics term penalizes constraint residuals at N_f collocation points:

```
MSE_physics = (1/N_f) Σᵢ [|D(ŷᵢ, xᵢ)|² + |h(xᵢ, ŷᵢ)|² + ReLU(g(xᵢ, ŷᵢ))²]
```

The penalty parameter λ_soft is a hyperparameter tuned to trade off between data fit and physics compliance (Figure 4 of the seed paper shows this trade-off quantitatively).

### 1.3 Multiple-Point Neighborhood Approximation (DAEHardNet §3)

The key innovation that makes differential constraints tractable in the KKT projection is expressing the function value at a reference point as a weighted sum of its values at neighboring points, via Taylor expansion:

```
y(x, t) ≈ M(x, t, ∂) = (1/|X|) Σᵢ∈X [ y([x,t] + Δᵢ) − Δ·∂ᵢ − ½Δ²·∂ᵢᵢ ]
```

where X = {x₁, x₂, …, xₙ, t} is the set of all independent variables and Δᵢ ∈ ℝ^|X| is the step vector in dimension i. In PITS-MRAS context with state x_p = [q, q̇] and time t:

```
y(q₀, q̇₀, t₀) ≈ (1/3) [ y(q₀+Δ, q̇₀, t₀) − Δ ∂_q
                         + y(q₀, q̇₀+Δ, t₀) − Δ ∂_{q̇}
                         + y(q₀, q̇₀, t₀+Δ) − Δ ∂_t
                         − ½Δ²(∂_qq + ∂_{q̇q̇} + ∂_tt) ]
```

This allows the differential operators ∂ to be treated as **independent algebraic variables d** in the projection, converting the infinite-dimensional ODE/PDE constraint to a finite NLP.

### 1.4 Hard PCML: KKT Projection (DAEHardNet §3.1)

The projection finds the minimum-distance point on the constraint manifold (Equation 12 of DAEHardNet):

```
ỹ, d̃ = argmin_{y,d}  ½ ‖y − f̂_θ‖²
         s.t.
             Uᵢ(x, t, y, d) = 0,     ∀i ∈ N_D    (differential eqs as algebraic, via Taylor)
             hⱼ(x, t, y) = 0,         ∀j ∈ N_E    (algebraic equality)
             gₖ(x, t, y) + sₖ = 0,   ∀k ∈ N_I    (inequality with slacks)
             yₚ = Mₚ(x, t, d),        ∀p ∈ N_y    (Taylor neighborhood coupling)
```

The KKT stationarity condition (Equation 13 of DAEHardNet) gives a square nonlinear system:

```
y − f̂ + ΣᵢλᵢᴰΓ∇_{y,d}Uᵢ + ΣⱼλⱼᴱΓ∇_{y,d}hⱼ + ΣₖλₖᴵΓ∇_{y,d}gₖ + Σₚλₚ∇_{y,d}[yₚ − Mₚ] = 0
Uᵢ = 0 ∀i ∈ N_D
hⱼ = 0 ∀j ∈ N_E
gₖ + sₖ = 0 ∀k ∈ N_I
yₚ − Mₚ = 0 ∀p ∈ N_y
√[(λₖᴵ)² + sₖ²] − λₖᴵ − sₖ = 0 ∀k ∈ N_I    (Fischer-Burmeister complementarity)
```

This is solved by Newton’s method (differentiable through the implicit function theorem, enabling gradient backpropagation).

### 1.5 Hard PCML Loss Function (DAEHardNet Equation 15)

```
L_pcml_hard = MSE(ỹ, ȳ) + ω × MSE(d̃, AD(∂ỹ))
```

where:

- `ỹ` = projected (hard-constrained) PITNN output
- `ȳ` = target/measurement
- `d̃` = projected derivatives (from KKT solution)
- `AD(∂ỹ)` = autograd derivatives of `ỹ` with respect to inputs
- `ω` = derivative loss weight (default 1.0 = spherical norm ball; see §3.2 of DAEHardNet)

**Key insight from DAEHardNet §3.2:** The loss landscape of L_pcml_hard is an *ellipsoidal norm ball* (near-spherical when ω≈1), dramatically more stable than the multi-objective landscape of PINNs. This resolves the nonconvexity that plagues the existing L_physics term. Benchmark results: constraint violations reduced from O(10⁻¹) (PINN) to O(10⁻⁷ to 10⁻¹¹) (DAEHardNet).

-----

## §2 · New Files and Specifications

### 2.1 Constraint Library

Create `src/pits_mras/constraints/__init__.py` and the following files.

#### `src/pits_mras/constraints/base.py`

```python
"""
Base class for all physics constraint specifications.

Each constraint system must define:
  - D(x, t, y, d): differential equations (as algebraic, with derivatives d)
  - h(x, t, y):    algebraic equality constraints
  - g(x, t, y):    inequality constraints
  - jacobian_y(x, t, y, d): ∇_y of the constraint system (for KKT)
  - jacobian_d(x, t, y, d): ∇_d of the constraint system (for KKT)

This structure mirrors DAEHardNet's problem setup (Equation 1).
"""
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class ConstraintSpec:
    """Metadata about the constraint system — how many of each type."""
    n_differential: int = 0    # |N_D|
    n_equality: int = 0        # |N_E|
    n_inequality: int = 0      # |N_I|
    n_outputs: int = 0         # |N_y|  — number of output variables to couple via Taylor


class PhysicsConstraints(ABC):
    """
    Abstract base for physics constraint systems.

    The implementing class defines the governing equations of the controlled plant.
    All methods operate in batch mode: first dim is batch.
    """

    @property
    @abstractmethod
    def spec(self) -> ConstraintSpec:
        """Return the constraint specification metadata."""
        ...

    @abstractmethod
    def differential(self, x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor:
        """
        Evaluate D(x, t, y, d) = 0.
        d are the derivative variables (first and second order).
        Returns residual of shape [batch, n_differential].
        """
        ...

    @abstractmethod
    def equality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate h(x, t, y) = 0.
        Returns residual of shape [batch, n_equality].
        """
        ...

    @abstractmethod
    def inequality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate g(x, t, y) ≤ 0.
        Returns value of shape [batch, n_inequality].
        A positive value means constraint is violated.
        """
        ...

    def violation(self, x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor:
        """
        Compute the mean absolute constraint violation (evaluation metric).
        Mirrors DAEHardNet's Violation metric formula.
        """
        diff_viol  = self.differential(x, t, y, d).abs().mean()
        eq_viol    = self.equality(x, t, y).abs().mean()
        ineq_viol  = torch.relu(self.inequality(x, t, y)).mean()
        n_total = self.spec.n_differential + self.spec.n_equality + self.spec.n_inequality
        return (diff_viol * self.spec.n_differential +
                eq_viol  * self.spec.n_equality +
                ineq_viol * self.spec.n_inequality) / max(n_total, 1)
```

#### `src/pits_mras/constraints/mechanical.py`

```python
"""
Mechanical system constraints: Euler-Lagrange equations with holonomic constraints.

The DAE system for a robot with n joints and m holonomic constraints is:
  D₁: M(q)q̈ + C(q,q̇)q̇ + G(q) − B(q)u − Jᵀ(q)λ = 0   (equations of motion)
  D₂: Ψ(q) = 0                                            (holonomic position constraint)
  D₃: J(q)q̇  = 0                                         (velocity constraint)
  h:  λ − Λ(q)[J(q)M⁻¹(q)(B(q)u − C(q,q̇)q̇ − G(q)) + J̇(q)q̇] = 0  (constraint forces)
  g:  q_min ≤ q ≤ q_max                                   (joint limits)
  g:  τ_min ≤ u ≤ τ_max                                   (torque limits)

where Λ(q) = [J(q)M⁻¹(q)Jᵀ(q)]⁻¹ is the constraint inertia matrix.

For a system without holonomic constraints (m=0), only D₁ applies.
"""
import torch
from torch import Tensor
from typing import Callable, Optional
from pits_mras.constraints.base import PhysicsConstraints, ConstraintSpec


class MechanicalDAE(PhysicsConstraints):
    """
    DAE constraints for mechanical systems with optional holonomic constraints.

    Parameters:
        inertia_fn:    M(q) → [batch, n, n]      mass/inertia matrix
        coriolis_fn:   (q, q_dot) → [batch, n]   Coriolis/centrifugal forces
        gravity_fn:    q → [batch, n]             gravity forces
        actuator_fn:   q → [batch, n, m]          actuator input matrix B(q)
        constraint_fn: q → [batch, m, n]          holonomic constraint Jacobian J(q)
        q_bounds:      (q_min, q_max) or None      joint limits [n]
        u_bounds:      (u_min, u_max) or None      torque limits [m_control]
    """

    def __init__(
        self,
        n_joints: int,
        n_holonomic: int,
        inertia_fn: Callable,
        coriolis_fn: Callable,
        gravity_fn: Callable,
        actuator_fn: Callable,
        constraint_fn: Optional[Callable] = None,
        q_bounds: Optional[tuple] = None,
        u_bounds: Optional[tuple] = None,
    ):
        self.n_joints = n_joints
        self.n_holonomic = n_holonomic
        self.inertia_fn = inertia_fn
        self.coriolis_fn = coriolis_fn
        self.gravity_fn = gravity_fn
        self.actuator_fn = actuator_fn
        self.constraint_fn = constraint_fn
        self.q_bounds = q_bounds
        self.u_bounds = u_bounds

        n_diff = 1 + (2 if n_holonomic > 0 else 0)  # EOM + (Ψ=0, Jq̇=0 if constrained)
        n_eq   = n_holonomic if n_holonomic > 0 else 0   # λ equation
        n_ineq = (2 * n_joints if q_bounds else 0) + (2 * n_joints if u_bounds else 0)
        self._spec = ConstraintSpec(n_diff, n_eq, n_ineq, n_joints)

    @property
    def spec(self) -> ConstraintSpec:
        return self._spec

    def differential(self, q: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor:
        """
        y = [q, q_dot],  d = [q_dot (from Taylor), q_ddot, lambda]
        The EOM residual: M(q)q̈ + C(q,q̇)q̇ + G(q) − B(q)u − Jᵀ(q)λ = 0
        """
        q     = y[:, :self.n_joints]
        q_dot = y[:, self.n_joints:2*self.n_joints]
        q_ddot = d[:, self.n_joints:2*self.n_joints]     # second derivatives
        lam   = d[:, 2*self.n_joints:] if self.n_holonomic > 0 else None

        M   = self.inertia_fn(q)                         # [batch, n, n]
        C   = self.coriolis_fn(q, q_dot)                 # [batch, n]
        G   = self.gravity_fn(q)                         # [batch, n]
        Bu  = self.actuator_fn(q)                        # [batch, n, m] — u injected externally

        # EOM: M q̈ + C q̇ + G − B u − Jᵀ λ = 0
        eom = (M @ q_ddot.unsqueeze(-1)).squeeze(-1) + C + G
        if lam is not None and self.constraint_fn is not None:
            J = self.constraint_fn(q)                    # [batch, m, n]
            eom = eom - (J.transpose(-1, -2) @ lam.unsqueeze(-1)).squeeze(-1)

        residuals = [eom]

        if self.n_holonomic > 0 and self.constraint_fn is not None:
            J    = self.constraint_fn(q)
            psi  = (J @ q.unsqueeze(-1)).squeeze(-1)         # Ψ(q) = J(q)·q (linear holonomic)
            Jqdot = (J @ q_dot.unsqueeze(-1)).squeeze(-1)    # J(q)·q̇ = 0
            residuals += [psi, Jqdot]

        return torch.cat(residuals, dim=-1)  # [batch, n_differential]

    def equality(self, q: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """Constraint force algebraic equation (λ determination)."""
        if self.n_holonomic == 0 or self.constraint_fn is None:
            return torch.zeros(q.shape[0], 0, device=q.device)
        # λ = Λ(q)[J M⁻¹(B u − C q̇ − G) + J̇ q̇]  — simplified version
        q = y[:, :self.n_joints]
        q_dot = y[:, self.n_joints:2*self.n_joints]
        J = self.constraint_fn(q)
        M = self.inertia_fn(q)
        M_inv = torch.linalg.inv(M)
        Lambda = torch.linalg.inv(J @ M_inv @ J.transpose(-1, -2))
        C = self.coriolis_fn(q, q_dot)
        G = self.gravity_fn(q)
        lam_pred = (Lambda @ (J @ M_inv @ (-C - G).unsqueeze(-1))).squeeze(-1)
        # The algebraic constraint says the predicted λ must equal lam_pred
        # This is enforced as h(y) = 0 by matching the two estimates
        return lam_pred  # [batch, n_holonomic] — residual form

    def inequality(self, q: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """Joint and torque limit constraints g(y) ≤ 0."""
        q_state = y[:, :self.n_joints]
        violations = []
        if self.q_bounds is not None:
            q_min, q_max = self.q_bounds
            violations += [q_min - q_state, q_state - q_max]
        return torch.cat(violations, dim=-1) if violations else torch.zeros(q.shape[0], 0, device=q.device)
```

#### `src/pits_mras/constraints/thermal.py`

```python
"""
Thermal system constraints for HVAC and heat conduction.

Governing PDE: ∂T/∂t = α ∂²T/∂x²  (heat diffusion)
Steady-state:  ∂²T/∂x² + Q(x) / k = 0  (Poisson equation with source Q)
Energy balance: dU/dt = Q_in − Q_out (conservation of thermal energy)

From DAEHardNet Example 6 (1-D transient heat conduction).
"""
import torch
from torch import Tensor
from pits_mras.constraints.base import PhysicsConstraints, ConstraintSpec


class HeatConductionDAE(PhysicsConstraints):
    """
    1-D transient heat conduction: ∂T/∂t = α ∂²T/∂x²
    Mirrors DAEHardNet Example 6 exactly.

    For PITS-MRAS HVAC application: x = spatial coordinate, t = time,
    y = T(x,t) = temperature field, α = thermal diffusivity.
    """

    def __init__(self, alpha: float, T_min: float = 15.0, T_max: float = 35.0):
        self.alpha = alpha
        self.T_min = T_min
        self.T_max = T_max
        self._spec = ConstraintSpec(
            n_differential=1,   # heat equation residual
            n_equality=0,
            n_inequality=2,     # T_min ≤ T ≤ T_max
            n_outputs=1,
        )

    @property
    def spec(self) -> ConstraintSpec:
        return self._spec

    def differential(self, x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor:
        """
        D: ∂T/∂t − α ∂²T/∂x² = 0
        d = [dT_dx, dT_dt, d²T_dx², d²T_dt²] (derivative variables)
        Returns residual [batch, 1].
        """
        dT_dt   = d[:, 1:2]     # time derivative
        d2T_dx2 = d[:, 2:3]     # second spatial derivative
        residual = dT_dt - self.alpha * d2T_dx2
        return residual          # [batch, 1]

    def equality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], 0, device=x.device)

    def inequality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        T = y[:, 0:1]
        lower = self.T_min - T   # ≤ 0 when T ≥ T_min
        upper = T - self.T_max   # ≤ 0 when T ≤ T_max
        return torch.cat([lower, upper], dim=-1)  # [batch, 2]
```

### 2.2 PCML Core Module

Create `src/pits_mras/models/pcml.py`:

```python
"""
Physics-Constrained Machine Learning (PCML) Module for PITS-MRAS.

Implements two constraint enforcement modes:
  1. Soft PCML (Patel et al. 2022): Augmented loss function approach.
     MSE_total = MSE_data + λ × MSE_physics
     Reduces constraint violation probabilistically; no hard guarantee.
     Used during early training phases.

  2. Hard PCML (DAEHardNet, Golder et al. 2025): KKT projection layer.
     Projects neural network outputs onto the DAE constraint manifold.
     Achieves point-wise constraint satisfaction to machine precision.
     Activated dynamically when backbone loss < η threshold.

The PCML module wraps the existing PortHamiltonianDecoder output f̂_θ.
During hard mode, f̂_θ is projected to f̂_pcml with guaranteed constraint
satisfaction, which is then used by the MRAS controller.

References:
  - Patel et al. (IFAC PapersOnLine 55-7, 2022) - Soft PCML
  - Golder, Roy & Hasan (arXiv:2512.05881, 2025) - DAEHardNet hard PCML
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from pits_mras.constraints.base import PhysicsConstraints, ConstraintSpec


class SoftPCMLLoss(nn.Module):
    """
    Soft physics constraint loss augmentation (Patel et al. 2022, Equations 3-4).

    Computes: L_pcml_soft = λ_diff × ‖D(ŷ,x,t)‖² + λ_eq × ‖h(x,ŷ)‖² + λ_ineq × ‖ReLU(g)‖²

    This is the GENERALIZATION of the existing L_physics loss in PITS-MRAS:
      - The existing L_energy = port-Hamiltonian energy conservation constraint
      - The existing L_PDE = PDE operator residual
      - The existing L_BC = boundary condition residual
    All of these are SPECIAL CASES of the differential/equality/BC constraints above.

    The SoftPCMLLoss should REPLACE L_physics during hard-PCML training phases,
    and CO-EXIST with L_physics during soft-PCML phases.
    """

    def __init__(
        self,
        constraints: PhysicsConstraints,
        lambda_diff: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 0.5,
    ):
        super().__init__()
        self.constraints = constraints
        self.lambda_diff = lambda_diff
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq

    def forward(
        self,
        x: Tensor,            # [batch, spatial_dim]
        t: Tensor,            # [batch, 1]
        y_pred: Tensor,       # [batch, n_y]  neural network prediction f̂_θ
        d_pred: Tensor,       # [batch, n_d]  derivative prediction (from autograd or Taylor)
    ) -> Tuple[Tensor, dict]:
        """
        Returns (total_soft_loss, breakdown_dict).
        breakdown_dict has keys 'diff', 'eq', 'ineq', 'violation'.
        """
        diff_res  = self.constraints.differential(x, t, y_pred, d_pred)
        eq_res    = self.constraints.equality(x, t, y_pred)
        ineq_val  = self.constraints.inequality(x, t, y_pred)

        L_diff  = (diff_res ** 2).mean() if diff_res.numel() > 0 else torch.tensor(0.0)
        L_eq    = (eq_res ** 2).mean() if eq_res.numel() > 0 else torch.tensor(0.0)
        L_ineq  = (F.relu(ineq_val) ** 2).mean() if ineq_val.numel() > 0 else torch.tensor(0.0)

        total = self.lambda_diff * L_diff + self.lambda_eq * L_eq + self.lambda_ineq * L_ineq
        violation = self.constraints.violation(x, t, y_pred, d_pred)

        return total, {'diff': L_diff, 'eq': L_eq, 'ineq': L_ineq, 'violation': violation}


class TaylorNeighborhoodApproximation(nn.Module):
    """
    Multiple-point neighborhood approximation (DAEHardNet §3, Equation 9).

    Evaluates y(x,t) at nearby points to express the function value as a
    weighted combination of its neighbors and their derivatives, enabling
    differential constraints to be treated as algebraic constraints in the KKT system.

    For input dimension |X| (= spatial_dim + 1 for time):
        y(x,t) ≈ (1/|X|) Σᵢ∈X [ y([x,t]+Δᵢ) − Δ·∂ᵢ − ½Δ²·∂ᵢᵢ ]

    The approximation error vanishes as Δ→0, but very small Δ causes numerical
    instability in Newton projection. Recommended range: Δ ∈ [0.001, 0.1].
    """

    def __init__(
        self,
        backbone: nn.Module,     # the neural network that is being approximated
        input_dim: int,          # spatial_dim + 1 (for time)
        delta: float = 0.01,     # Taylor offset Δ
        order: int = 1,          # 1 or 2 (DAEHardNet used order=1 for most examples)
    ):
        super().__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.delta = delta
        self.order = order

    def forward(self, inputs: Tensor, derivatives: Tensor) -> Tensor:
        """
        Compute M(x, t, d) — the neighborhood approximation.

        inputs:      [batch, input_dim]    (x, t)
        derivatives: [batch, n_derivs]     (∂ᵢ for each input dim, and ∂ᵢᵢ if order=2)

        Returns: y_approx [batch, output_dim] — the approximated function value.
        """
        batch = inputs.shape[0]
        neighbor_values = []

        for i in range(self.input_dim):
            # Step in dimension i
            delta_vec = torch.zeros_like(inputs)
            delta_vec[:, i] = self.delta
            neighbor_input = inputs + delta_vec

            with torch.enable_grad():
                y_neighbor = self.backbone(neighbor_input)

            # First-order correction: y_neighbor − Δ·∂ᵢ
            d_i = derivatives[:, i:i+1]   # [batch, 1] or [batch, output_dim]
            term = y_neighbor - self.delta * d_i.expand_as(y_neighbor)

            # Second-order correction: − ½Δ²·∂ᵢᵢ
            if self.order >= 2:
                d_ii = derivatives[:, self.input_dim + i:self.input_dim + i + 1]
                term = term - 0.5 * (self.delta ** 2) * d_ii.expand_as(y_neighbor)

            neighbor_values.append(term)

        # Average over all directions
        y_approx = torch.stack(neighbor_values, dim=0).mean(dim=0)  # [batch, output_dim]
        return y_approx


class KKTProjectionLayer(nn.Module):
    """
    KKT-based differentiable projection layer (DAEHardNet §3.1).

    Projects the unconstrained network output f̂_θ onto the DAE constraint
    manifold by solving the KKT conditions of the distance minimization problem:

        ỹ, d̃ = argmin ½‖y − f̂_θ‖²  s.t. D=0, h=0, g≤0, y=M(d)

    The KKT system is solved using Newton's method (differentiable via implicit
    differentiation — gradients flow through the projection for backpropagation).

    Implementation follows Algorithm 1 of DAEHardNet.
    The projection layer is activated dynamically when the backbone data loss
    drops below η (the 'eta' threshold), following DAEHardNet's dynamic activation.
    """

    def __init__(
        self,
        constraints: PhysicsConstraints,
        taylor_approx: TaylorNeighborhoodApproximation,
        n_output: int,           # number of output variables |N_y|
        n_deriv: int,            # number of derivative variables |d|
        newton_step: float = 1.0,
        max_newton_iter: int = 10,
        newton_tol: float = 1e-6,
    ):
        super().__init__()
        self.constraints = constraints
        self.taylor_approx = taylor_approx
        self.n_output = n_output
        self.n_deriv = n_deriv
        self.newton_step = newton_step
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol

        spec = constraints.spec
        self.n_eq_total = (spec.n_differential + spec.n_equality +
                           spec.n_inequality + spec.n_outputs)
        # Lagrangian multiplier sizes
        self.n_lambda = spec.n_differential + spec.n_equality + spec.n_inequality + spec.n_outputs
        self.n_slack = spec.n_inequality

    def _build_kkt_system(
        self,
        x: Tensor,
        t: Tensor,
        y_hat: Tensor,           # unconstrained prediction [batch, n_output]
        y_curr: Tensor,          # current y iterate [batch, n_output]
        d_curr: Tensor,          # current d iterate [batch, n_deriv]
        lam: Tensor,             # current Lagrangian multipliers [batch, n_lambda]
        slack: Tensor,           # current slack variables [batch, n_slack]
    ) -> Tensor:
        """
        Build the KKT residual vector F(y, d, λ, s) — the square system of equations
        whose root gives the projected (ỹ, d̃).

        Returns F of shape [batch, n_variables_total] where:
        n_variables_total = n_output + n_deriv + n_lambda + n_slack
        """
        spec = self.constraints.spec

        # 1. Stationarity: y − ŷ + Σ λᵢ ∇_{y,d} constraint_i = 0
        residuals = [y_curr - y_hat]   # [batch, n_output]

        # 2. Primal feasibility: D(y,d)=0, h(y)=0, g(y)+s=0, y-M(d)=0
        diff_res = self.constraints.differential(x, t, y_curr, d_curr)     # [batch, n_D]
        eq_res   = self.constraints.equality(x, t, y_curr)                  # [batch, n_E]
        ineq_res = self.constraints.inequality(x, t, y_curr)                # [batch, n_I]
        taylor_res = y_curr - self.taylor_approx(torch.cat([x, t], dim=-1), d_curr)  # [batch, n_y]

        if spec.n_differential > 0:
            residuals.append(diff_res)
        if spec.n_equality > 0:
            residuals.append(eq_res)
        if spec.n_inequality > 0:
            residuals.append(ineq_res + slack)    # g(y) + s = 0
        if spec.n_outputs > 0:
            residuals.append(taylor_res)

        # 3. Fischer-Burmeister complementarity: √(λ² + s²) − λ − s = 0 for each ineq
        if spec.n_inequality > 0:
            lam_ineq = lam[:, spec.n_differential + spec.n_equality:
                              spec.n_differential + spec.n_equality + spec.n_inequality]
            fb = torch.sqrt(lam_ineq ** 2 + slack ** 2 + 1e-8) - lam_ineq - slack
            residuals.append(fb)

        return torch.cat(residuals, dim=-1)  # [batch, n_total]

    def forward(
        self,
        x: Tensor,           # [batch, spatial_dim]
        t: Tensor,           # [batch, 1]
        y_hat: Tensor,       # [batch, n_output]  unconstrained PITNN output
        d_hat: Tensor,       # [batch, n_deriv]   autograd derivative of y_hat
        lam_hat: Tensor,     # [batch, n_lambda]  predicted Lagrangian multipliers from backbone
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run the KKT projection (Newton iterations).

        Returns:
            y_tilde:   [batch, n_output]   hard-constrained output
            d_tilde:   [batch, n_deriv]    hard-constrained derivatives
            lam_tilde: [batch, n_lambda]   projected multipliers
        """
        # Initialize Newton iterates from the backbone predictions
        y = y_hat.detach().clone().requires_grad_(False)
        d = d_hat.detach().clone().requires_grad_(False)
        lam = lam_hat.detach().clone().requires_grad_(False)
        slack = torch.ones(x.shape[0], self.n_slack, device=x.device) * 0.1

        for _ in range(self.max_newton_iter):
            F = self._build_kkt_system(x, t, y_hat, y, d, lam, slack)

            if F.abs().max() < self.newton_tol:
                break

            # Jacobian via autograd (enables backprop through projection)
            jac = torch.zeros(x.shape[0], F.shape[1], y.shape[1] + d.shape[1],
                              device=x.device)
            for k in range(F.shape[1]):
                grad_y = torch.autograd.grad(
                    F[:, k].sum(), [y, d],
                    create_graph=True, allow_unused=True
                )
                if grad_y[0] is not None:
                    jac[:, k, :y.shape[1]] = grad_y[0]
                if grad_y[1] is not None:
                    jac[:, k, y.shape[1]:] = grad_y[1]

            # Newton step: [y; d] ← [y; d] − α J⁺ F  (pseudoinverse for over/underdetermined)
            # For square system: use torch.linalg.solve
            yd = torch.cat([y, d], dim=-1)
            try:
                delta_yd = torch.linalg.lstsq(jac, F.unsqueeze(-1)).solution.squeeze(-1)
                yd_new = yd - self.newton_step * delta_yd
                y = yd_new[:, :y.shape[1]].detach().requires_grad_(True)
                d = yd_new[:, y.shape[1]:].detach().requires_grad_(True)
            except RuntimeError:
                break   # Degenerate Jacobian — skip this batch

        return y, d, lam

    def hard_constraint_loss(
        self,
        y_tilde: Tensor,     # projected output
        y_true: Tensor,      # ground truth
        d_tilde: Tensor,     # projected derivatives
        y_tilde_ad: Tensor,  # autograd derivative of y_tilde
        omega: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute DAEHardNet loss (Equation 15):
            L = MSE(ỹ, ȳ) + ω × MSE(d̃, AD(∂ỹ))

        This near-spherical loss landscape (vs the complex PINN landscape) is
        a key source of DAEHardNet's training stability advantage.

        Returns (total_loss, breakdown) where breakdown has 'data' and 'deriv' keys.
        """
        L_data  = F.mse_loss(y_tilde, y_true)
        L_deriv = F.mse_loss(d_tilde, y_tilde_ad)
        total   = L_data + omega * L_deriv
        return total, {'data': L_data, 'deriv': L_deriv}


class PCMLModule(nn.Module):
    """
    Unified PCML wrapper that manages both soft and hard modes.

    Integration point in the PITNN forward pass:
      f_hat, H_val, P_diss, energy_loss = decoder(q, p, q_dot, u, context)
      f_pcml, pcml_loss, pcml_info = pcml_module(x, t, f_hat, d_hat, lam_hat, f_true)

    The PCMLModule replaces/augments L_physics from the existing plan:
      - Soft mode:  replaces L_energy + L_PDE + L_BC with SoftPCMLLoss
      - Hard mode:  projects f_hat → f_pcml, then L_hard replaces L_physics entirely

    The module tracks backbone data loss and automatically activates the hard
    projection layer when loss drops below η (DAEHardNet dynamic activation).
    """

    def __init__(
        self,
        constraints: PhysicsConstraints,
        backbone: nn.Module,
        input_dim: int,
        n_output: int,
        n_deriv: int,
        n_lambda: int,
        lambda_soft: float = 1.0,
        omega: float = 1.0,          # derivative loss weight (DAEHardNet ω)
        delta: float = 0.01,         # Taylor offset Δ
        taylor_order: int = 1,
        eta: float = 0.01,           # activation threshold (DAEHardNet eta)
        newton_step: float = 1.0,
        max_newton_iter: int = 10,
    ):
        super().__init__()
        self.eta = eta
        self._hard_mode_active = False
        self._best_data_loss = float("inf")

        self.soft_loss = SoftPCMLLoss(constraints, lambda_diff=lambda_soft)
        self.taylor_approx = TaylorNeighborhoodApproximation(
            backbone, input_dim, delta=delta, order=taylor_order
        )
        self.projection = KKTProjectionLayer(
            constraints, self.taylor_approx,
            n_output, n_deriv,
            newton_step=newton_step,
            max_newton_iter=max_newton_iter,
        )
        self.constraints = constraints

    def update_activation(self, current_data_loss: float) -> bool:
        """
        Check if hard mode should be activated based on the current backbone data loss.
        Mirrors DAEHardNet's dynamic projection activation logic.

        Returns True if hard mode was just activated this call.
        """
        if not self._hard_mode_active and current_data_loss < self.eta:
            self._hard_mode_active = True
            return True
        return False

    @property
    def mode(self) -> str:
        return "hard" if self._hard_mode_active else "soft"

    def forward(
        self,
        x: Tensor,           # [batch, spatial_dim]
        t: Tensor,           # [batch, 1]
        y_hat: Tensor,       # [batch, n_output]  PITNN unconstrained prediction f̂_θ
        lam_hat: Tensor,     # [batch, n_lambda]  backbone-predicted Lagrangian multipliers
        y_true: Optional[Tensor] = None,   # [batch, n_output] or None (inference)
    ) -> Tuple[Tensor, Tensor, dict]:
        """
        Forward pass through the PCML module.

        Returns:
            y_pcml:    [batch, n_output]  constrained prediction
            pcml_loss: scalar             PCML loss (soft or hard depending on mode)
            info:      dict               breakdown of loss components and violation metric
        """
        # Always compute the autograd derivatives for monitoring/loss
        y_hat_with_grad = y_hat.requires_grad_(True)
        # Approximate d as the autograd gradient of y_hat w.r.t. inputs
        # (In full implementation, this is computed from the backbone's forward pass)
        d_hat = torch.autograd.grad(
            y_hat_with_grad.sum(), x, create_graph=True, allow_unused=True
        )[0]
        if d_hat is None:
            d_hat = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        if self._hard_mode_active:
            # ── HARD MODE: KKT projection ──
            y_tilde, d_tilde, lam_tilde = self.projection(x, t, y_hat, d_hat, lam_hat)

            # Autograd derivative of the projected output (for DAEHardNet derivative loss)
            y_tilde_ad = torch.autograd.grad(
                y_tilde.sum(), x, create_graph=True, allow_unused=True
            )[0]
            if y_tilde_ad is None:
                y_tilde_ad = d_hat

            if y_true is not None:
                loss, breakdown = self.projection.hard_constraint_loss(
                    y_tilde, y_true, d_tilde, y_tilde_ad
                )
            else:
                loss = torch.tensor(0.0, device=x.device)
                breakdown = {'data': 0.0, 'deriv': 0.0}

            violation = self.constraints.violation(x, t, y_tilde, d_tilde)
            info = {'mode': 'hard', 'violation': violation, **breakdown}
            return y_tilde, loss, info

        else:
            # ── SOFT MODE: augmented loss ──
            soft_loss, breakdown = self.soft_loss(x, t, y_hat, d_hat)
            info = {'mode': 'soft', **breakdown}
            return y_hat, soft_loss, info
```

### 2.3 Lagrangian Multiplier Head in PITNN

The KKT projection requires predicted Lagrangian multipliers λ̂ as warm-start values for Newton iteration. Add a dedicated head to the PITNN that produces λ̂:

Create `src/pits_mras/models/lagrangian_head.py`:

```python
"""
Lagrangian multiplier head for DAEHardNet-style KKT projection.

The backbone NN in DAEHardNet outputs Ŷ = [ŷ, λ̂, ∂ˆ]:
  - ŷ:  unconstrained state prediction
  - λ̂:  predicted Lagrangian multipliers (warm start for Newton)
  - ∂ˆ: auto-differentiation derivatives of ŷ

In PITS-MRAS, this head attaches to the PortHamiltonianDecoder's context
vector c_t to predict λ̂ for the KKT system.
"""
import torch.nn as nn
from torch import Tensor


class LagrangianMultiplierHead(nn.Module):
    """
    Predicts the Lagrangian multipliers λ̂ ∈ ℝ^n_lambda for the KKT projection.
    These serve as warm-start values for Newton's method in the projection layer.

    Architecture: 2-layer MLP from the attention context c_t.
    Multipliers for equality constraints: unconstrained (any sign).
    Multipliers for inequality constraints: must be ≥ 0 (Softplus activation).
    """

    def __init__(
        self,
        context_dim: int,
        n_lambda_eq: int,    # equality + differential constraint multipliers (any sign)
        n_lambda_ineq: int,  # inequality constraint multipliers (must be ≥ 0)
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.n_lambda_eq = n_lambda_eq
        self.n_lambda_ineq = n_lambda_ineq

        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_lambda_eq + n_lambda_ineq),
        )

    def forward(self, context: Tensor) -> Tensor:
        """
        Returns λ̂ of shape [batch, n_lambda_eq + n_lambda_ineq].
        Equality multipliers: free (any sign).
        Inequality multipliers: non-negative (Softplus applied).
        """
        raw = self.net(context)                              # [batch, n_total]
        if self.n_lambda_ineq > 0:
            lam_eq   = raw[:, :self.n_lambda_eq]
            lam_ineq = nn.functional.softplus(raw[:, self.n_lambda_eq:])
            return torch.cat([lam_eq, lam_ineq], dim=-1)
        return raw
```

### 2.4 Integration into PITNN (Modification to Existing §5.4)

Add the following to `src/pits_mras/models/pitnn.py`:

1. Import `PCMLModule`, `LagrangianMultiplierHead`.
1. Add `self.lagrangian_head = LagrangianMultiplierHead(net_cfg.hidden_dim, n_lambda_eq, n_lambda_ineq)` in `__init__`.
1. In `forward()`, after computing `f_hat` from the decoder, add:

```python
# Step 7: Predict Lagrangian multipliers for KKT warm start
lam_hat = self.lagrangian_head(context)   # [batch, n_lambda]

# Step 8: PCML projection (soft or hard depending on mode)
# NOTE: pcml_module is passed in from the training loop, not stored in PITNN,
# because it needs access to y_true during training but not during inference.
# Store it here only if needed for standalone inference.
```

The `pcml_module` is attached to the training loop (§3 below), not to the PITNN itself, following DAEHardNet’s separation of backbone (learnable) and projection (non-learnable, differentiable).

-----

## §3 · Updated Loss Function and Training Integration

### 3.1 PCML Loss in `TotalLoss` Aggregator

Update `src/pits_mras/losses/__init__.py` to add PCML terms. The updated total loss is:

```
L_total = L_physics         (or replaced by L_pcml_soft in soft mode)
        + λ_temporal L_temporal
        + λ_stab L_stability
        + L_data
        + L_irl              (Connection 1 — IRL Bellman)
        + λ_hjb L_hjb        (Connection 8 — HJB residual)
        + λ_costate L_costate (Connection 3 — costate consistency)
        + λ_pcml L_pcml_soft  (NEW — soft mode, replaces L_physics when active)
        + L_pcml_hard         (NEW — hard mode, replaces L_pcml_soft when η reached)
```

**Mode switching logic:**

```python
# In the co-training loop (cotrain.py):
if pcml_module.update_activation(current_data_loss):
    print(f"[PCML] Hard mode activated at epoch {epoch}, data_loss={current_data_loss:.4e}")
    # Disable the existing L_physics terms (now handled by hard PCML)
    cfg.losses.lambda_physics = 0.0

if pcml_module.mode == "soft":
    f_pcml = f_hat                         # soft mode: use unconstrained prediction
    pcml_loss = soft_pcml_loss(x, t, f_hat, d_hat)
else:
    f_pcml, pcml_loss, pcml_info = pcml_module(x, t, f_hat, lam_hat, f_true)
    # f_pcml is now hard-constrained and used downstream by the controller
```

### 3.2 Derivative Loss Connection (Synergistic Effect)

DAEHardNet §4.4 demonstrates a **synergistic effect**: minimizing the derivative loss improves the backbone’s prediction, and better backbone predictions improve the derivative loss. In PITS-MRAS, this synergy reinforces the costate identity (Identity 2): the derivative loss `MSE(d̃, AD(∂ỹ))` ensures the projected gradients match the autograd gradients, which are the same objects as the costate `λ = ∇V̂`. This creates a virtuous cycle:

```
Better λ̂  →  Better KKT warm start  →  Better projection  →  Better d̃
     ↑                                                              ↓
Smaller L_costate  ←  Closer alignment between d̃ and λ̂  ←  Synergistic effect
```

To implement this explicitly, add the following cross-term to `TotalLoss`:

```python
# PCML-costate synergy: project λ̂ into the same space as d̃ and enforce alignment
# This connects the IRL costate head to the PCML derivative variables
if pcml_module.mode == "hard":
    # lambda_hat (from costate head, shape [batch, state_dim]) should align
    # with the projected first-order temporal derivative d̃_t (∂f/∂t)
    d_tilde_t = d_tilde[:, -1:]   # last derivative = temporal component
    L_costate_pcml = F.mse_loss(lambda_hat[:, :1], d_tilde_t)
    L_total += cfg.losses.lambda_costate * L_costate_pcml
```

### 3.3 Inference Optimization (DAEHardNet §4.8)

DAEHardNet shows that after training, the difference between unconstrained backbone output and projected output is O(10⁻⁶). At this point, the projection can be bypassed for faster inference, reducing compute to near-MLP speed.

Add to `src/pits_mras/inference/realtime.py`:

```python
# In RealtimeInferenceEngine.step():

# Check if backbone output is already near the constraint manifold
# (DAEHardNet §4.8: projection difference < tolerance → skip projection)
if self.pcml_module is not None and self.pcml_module.mode == "hard":
    violation = self.pcml_module.constraints.violation(x, t, f_hat, d_hat)
    if violation.item() < self.pcml_projection_tolerance:
        f_pcml = f_hat    # skip projection for speed
    else:
        f_pcml, _, _ = self.pcml_module(x, t, f_hat, lam_hat)
else:
    f_pcml = f_hat

# Add 'violation' to the returned dict for monitoring
out["pcml_violation"] = violation.item() if self.pcml_module else 0.0
```

-----

## §4 · Updated Configuration

Add to `PITSMRASConfig` in `src/pits_mras/config.py`:

```python
@dataclass
class PCMLConfig:
    """Configuration for the PCML physics constraint module."""
    # Soft mode parameters (Patel et al. 2022)
    lambda_soft_diff: float = 1.0      # weight on differential constraint residuals
    lambda_soft_eq: float = 1.0        # weight on algebraic equality residuals
    lambda_soft_ineq: float = 0.5      # weight on inequality constraint violations

    # Hard mode parameters (DAEHardNet 2025)
    omega: float = 1.0                 # derivative loss weight (ω in Eq. 15 of DAEHardNet)
    eta: float = 0.01                  # activation threshold for hard mode (data loss < η)
    delta: float = 0.01                # Taylor offset Δ (recommended range: 0.001–0.1)
    taylor_order: int = 1              # Taylor approximation order (1 or 2)
    newton_step: float = 1.0           # Newton step length for KKT projection
    max_newton_iter: int = 10          # Maximum Newton iterations per forward pass
    pcml_projection_tolerance: float = 1e-5   # Skip projection if violation < this (inference)

    # Constraint system selection
    constraint_type: str = "mechanical"   # "mechanical", "thermal", or "custom"
    # For mechanical: system parameters
    n_joints: int = 2
    n_holonomic: int = 0               # 0 = no holonomic constraints
    q_bounds: Optional[tuple] = None
    # For thermal: diffusivity and temperature bounds
    thermal_alpha: float = 1.0
    T_min: float = 15.0
    T_max: float = 35.0
```

Update `PITSMRASConfig` to include `pcml: PCMLConfig = field(default_factory=PCMLConfig)`.

-----

## §5 · Updated Dependencies

Add to `requirements.txt` (needed for the Newton solver in the KKT projection):

```text
# Required for PCML KKT projection layer
scipy>=1.10.0    # already present; used for KKT Jacobian regularization
```

No additional packages are strictly required. The KKT Newton solver is implemented in pure PyTorch (`torch.linalg.lstsq`), keeping the dependency footprint small. If the user wants to compare against CVXPY-based projection:

```text
# Optional: for multi-constraint QP projection benchmarking
# cvxpy>=1.3.0
```

-----

## §6 · New Test Files

### `tests/test_pcml_soft.py`

```python
"""
Tests for soft PCML (Patel et al. 2022).

Test 1: Constraint violation decreases with higher λ_soft (Figure 4 of seed paper).
Test 2: A NN trained with λ_soft=1.0 should satisfy constraints better than λ_soft=0.
Test 3: Soft PCML loss for a known analytical constraint (y1 + y2 = 1, matching Case 1
        of the seed paper) should be zero when the constraint is satisfied.
"""
def test_violation_decreases_with_penalty():
    """Verify that increasing λ_soft reduces mean absolute constraint violation."""
    ...

def test_soft_pcml_zero_on_exact_constraint():
    """SoftPCMLLoss should return 0 when prediction exactly satisfies D=0 and h=0."""
    ...

def test_partial_physics_knowledge():
    """
    Seed paper Case 2 (distillation): Train with only partial physics (mass balance,
    not component balance). The PCML model should still satisfy the unknown constraint
    better than the data-only model.
    """
    ...
```

### `tests/test_pcml_hard.py`

```python
"""
Tests for hard PCML (DAEHardNet).

Test 1: KKT projection should reduce constraint violation to near machine precision
        for a simple linear DAE system (analogous to DAEHardNet Example 1).
Test 2: After projection, ‖y_tilde - y_hat‖ should be very small (< 1e-4 for a
        well-trained backbone), validating the inference bypass criterion.
Test 3: Hard PCML loss landscape should be more stable than PINN-style soft loss
        (verify that ω=1 gives spherical norm ball shape).
Test 4: Taylor neighborhood approximation with delta→0 should converge to autograd
        derivative (numerical validation of Equation 9 of DAEHardNet).
Test 5: Dynamic activation — hard mode activates when data_loss < eta.
        Before activation: mode == "soft". After: mode == "hard".
"""
def test_kkt_projection_reduces_violation():
    """For a simple DAE (e.g., Lotka-Volterra), KKT projection should achieve violation < 1e-6."""
    ...

def test_inference_bypass_valid():
    """After projection, gap ‖ỹ - ŷ‖ should be < pcml_projection_tolerance."""
    ...

def test_dynamic_activation():
    """update_activation() should switch mode from 'soft' to 'hard' at correct threshold."""
    ...

def test_derivative_loss_synergy():
    """
    Train a tiny MLP on a 1-D ODE for 100 steps. Verify that:
    (a) derivative loss decreases over epochs
    (b) data loss also decreases (synergistic effect, DAEHardNet §3.2)
    """
    ...
```

### `tests/test_pcml_integration.py`

```python
"""
Integration tests: PCML module connected to PITNN + MRAS controller.

Test 1: Full forward pass through PITNN → PCML → MRAS does not crash.
Test 2: In hard mode, f_pcml satisfies the mechanical constraints (D₁, D₂, D₃, h, g).
Test 3: Constraint violation reported in realtime engine output matches manual calculation.
Test 4: Lagrangian multiplier head outputs non-negative values for inequality multipliers.
"""
...
```

-----

## §7 · Updated Architecture Documentation in `docs/`

Update `docs/PITS-MRAS — Physics-Informed-Time-Series...md` with a new Section 2.7 titled “**Physics-Constrained Machine Learning (PCML) Layer**” that:

1. Explains the distinction between physics-*informed* (soft penalty) and physics-*constrained* (hard projection) neural networks.
1. Cites both seed papers explicitly with DOIs.
1. Shows the DAE problem formulation (Equation 1 of DAEHardNet).
1. Presents the updated total loss function including both soft and hard PCML terms.
1. Includes the updated architecture flowchart with the PCML projection block.
1. Explains the dynamic activation logic (η threshold) and inference bypass optimization.

Also add to `docs/PITS-MRAS_FINAL_SUMMARY.md`:

```markdown
## PCML Component (Added: vX.Y.Z)

**Source:** Patel et al. (IFAC 2022) + Golder et al. (DAEHardNet, arXiv:2512.05881)

**What it does:** Upgrades physics enforcement from soft penalties (PINNs)
to hard constraint satisfaction (KKT projection). Achieves constraint violations
at machine precision (O(10⁻⁷ to 10⁻¹¹)) vs. O(10⁻¹) for PINNs.

**Two modes:**
- Soft PCML (pre-training): augmented loss λ × ‖DAE residual‖²
- Hard PCML (co-training + inference): KKT projection → point-wise exact satisfaction

**Key identifiers:**
- `PCMLModule` in `src/pits_mras/models/pcml.py`
- `KKTProjectionLayer`: differentiable Newton projection
- `TaylorNeighborhoodApproximation`: converts differential to algebraic constraints
- `PhysicsConstraints` ABC: plug-in constraint system for any physical plant
```

-----

## §8 · Integration Map with the Main Implementation Plan

|Component from Main Plan        |PCML Interaction                                              |Change Required                                    |
|--------------------------------|--------------------------------------------------------------|---------------------------------------------------|
|`PortHamiltonianDecoder` (§5.2) |Feeds `f̂_θ` into `PCMLModule`                                 |Add `lam_hat` output via `LagrangianMultiplierHead`|
|`PhysicsLoss` (§6.1)            |Replaced by `SoftPCMLLoss` in soft mode, disabled in hard mode|Mode-dependent switching in `TotalLoss`            |
|`PITNN.forward()` (§5.4)        |Returns `lam_hat` in addition to existing outputs             |Add `lagrangian_head` call                         |
|Co-training loop (§8.2)         |PCML dynamic activation via `update_activation()`             |Add PCML forward call after decoder                |
|`RealtimeInferenceEngine` (§9.1)|PCML projection bypass for fast inference                     |Add `pcml_violation` check                         |
|`LossConfig` (§4.2)             |Add `lambda_pcml_soft`, `omega_pcml_hard`                     |Extend `LossConfig` dataclass                      |
|`TotalLoss.__init__`            |Instantiate `SoftPCMLLoss`                                    |Import and add to loss components                  |
|Smoke tests (§11.6)             |Add PCML to the smoke test forward pass                       |Import `PCMLModule` in test                        |

**Phase ordering update:** The main plan’s Phase 1–9 ordering is unchanged. PCML slots into Phase 3 (after Loss Functions) and Phase 5 (Training Pipelines). Specifically:

- PCML constraints (`src/pits_mras/constraints/`) → implement **between Phase 1 and Phase 2** (after config, before models).
- `pcml.py` and `lagrangian_head.py` → implement as part of **Phase 2** (Models).
- PCML loss integration → implement as part of **Phase 3** (Losses).
- Dynamic activation in training loop → implement as part of **Phase 5** (Training).

-----

## §9 · Key References (PCML-Specific)

- **Patel, R.S., Bhartiya, S., & Gudi, R.D.** “Physics Constrained Learning in Neural Network based Modeling.” *IFAC PapersOnLine* 55-7 (2022): 79–85. [Soft PCML seed document]
- **Golder, R., Roy, B.N., & Hasan, M.M.F.** “DAE-HardNet: A Physics Constrained Neural Network Enforcing Differential-Algebraic Hard Constraints.” arXiv:2512.05881 (Dec 2025). Code: `https://github.com/SOULS-TAMU/DAE-HardNet`. [Hard PCML seed document]
- **Iftakher, A., Golder, R., Roy, B.N., & Hasan, M.M.F.** “Physics-informed neural networks with hard nonlinear equality and inequality constraints.” *Computers & Chemical Engineering* (2025): 109418. [KKT-HardNet predecessor — algebraic-only version]
- **Lastrucci, G. & Schweidtmann, A.M.** “ENFORCE: Exact Nonlinear Constrained Learning with Adaptive-depth Neural Projection.” arXiv:2502.06774 (2025). [Alternative projection framework for nonlinear equality constraints]
- **Raissi, M., Perdikaris, P., & Karniadakis, G.E.** “Physics-informed neural networks.” *J. Computational Physics* 378 (2019): 686–707. [Original PINN baseline that PCML hardens]