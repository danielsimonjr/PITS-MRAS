"""Centralized configuration for PITS-MRAS (IP §4.2).

Owning phase: Phase 1 (Foundation Layer).

Dataclass-based config that maps directly to the hyperparameters in the
technical specification and the RL extensions. Six sub-config dataclasses
(``NetworkConfig``, ``PhysicsConfig``, ``MRASConfig``, ``SafetyConfig``,
``LossConfig``, ``TrainingConfig``) are aggregated into the master
``PITSMRASConfig`` with ``from_yaml`` / ``to_yaml``. stdlib :mod:`dataclasses`
is used (not pydantic), per the design docs.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import yaml


@dataclass
class NetworkConfig:
    """Architecture hyperparameters for PITNN."""

    input_dim: int = 10  # state + control dimension
    hidden_dim: int = 128
    output_dim: int = 4  # dynamics prediction dimension
    lstm_layers: int = 2
    attention_heads: int = 4
    memory_horizon: int = 50  # T: time steps of history to retain
    embedding_dim: int = 64


@dataclass
class PhysicsConfig:
    """Port-Hamiltonian decoder dimensions."""

    n_generalized_coords: int = 2  # n_q (positions)
    hamiltonian_hidden: int = 64  # width of H_θ network
    dissipation_hidden: int = 32  # width of L_θ network
    use_position_dependent_J: bool = False  # set True for nonholonomic systems


@dataclass
class MRASConfig:
    """Classical MRAS and new IRL/actor-critic parameters."""

    state_dim: int = 4  # dimension of tracking error e
    control_dim: int = 2  # dimension of control input u
    # Reference model: ẋ_m = A_m x_m + B_m r. Supply as nested lists.
    A_m: Optional[List[List[float]]] = None  # Hurwitz matrix [state_dim, state_dim]
    B_m: Optional[List[List[float]]] = None  # shape [state_dim, control_dim]
    C_m: Optional[List[List[float]]] = None  # shape [output_dim, state_dim]
    # LQR cost matrices
    Q_cost: Optional[List[List[float]]] = None  # tracking cost [state_dim, state_dim]
    R_cost: Optional[List[List[float]]] = None  # control cost [control_dim, control_dim]
    # Adaptation gains
    gamma_mras: float = 0.1  # classical MRAS adaptation rate
    adapt_rate_theta: float = 1e-4  # plant model learning rate
    adapt_rate_controller: float = 1e-3  # controller learning rate
    # IRL parameters (Connection 1)
    irl_window_size: int = 50  # T for IRL Bellman integral window
    use_irl_critic: bool = True  # enable Integral RL critic update


@dataclass
class SafetyConfig:
    """CLF-CBF-QP safety filter (Connection 3/6)."""

    enable_cbf: bool = True
    safety_margin: float = 10.0  # c in h(e) = c − eᵀPe
    cbf_decay_rate: float = 1.0  # γ in the CBF constraint


@dataclass
class LossConfig:
    """Loss weights for the unified total loss."""

    lambda_physics: float = 1.0
    lambda_temporal: float = 0.5
    lambda_stability: float = 2.0
    lambda_data: float = 1.0
    lambda_irl: float = 1.0  # IRL Bellman error weight
    lambda_hjb: float = 0.0  # HJB residual weight; >0 opts the critic into the
    # HJB regularizer (applied via the critic optimizer in cotraining_loop, §3.5)
    lambda_pcml: float = 1.0  # PCML constraint loss weight (soft or hard)
    # Physics sub-weights (consumed by PhysicsLoss)
    lambda_energy: float = 1.0
    lambda_pde: float = 1.0
    lambda_bc: float = 0.5
    lambda_sym: float = 0.2


@dataclass
class TrainingConfig:
    """Training schedule parameters matching Algorithm 2 and Algorithm 3."""

    # Pre-training (Algorithm 2)
    pretrain_epochs: int = 5000
    pretrain_batch_size: int = 64
    pretrain_lr: float = 1e-3
    stage1_epochs: int = 1000  # physics-only
    stage2_epochs: int = 2000  # data-physics balance
    # Co-training (Algorithm 3)
    n_episodes: int = 1000
    sim_duration: float = 10.0  # T_sim in seconds
    dt: float = 0.01  # Δt in seconds
    # General
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_every: int = 100
    checkpoint_every: int = 500


@dataclass
class PCMLConfig:
    """Physics-Constrained ML module config (PCML Addendum §4).

    Soft mode (Patel et al. 2022) and hard mode (DAE-HardNet) parameters, plus
    the constraint-system selection used to build the :class:`PhysicsConstraints`.
    """

    # Soft mode (Patel et al. 2022) residual weights.
    lambda_soft_diff: float = 1.0
    lambda_soft_eq: float = 1.0
    lambda_soft_ineq: float = 0.5
    # Hard mode (DAE-HardNet) parameters.
    omega: float = 1.0  # derivative-loss weight (Eq. 15)
    eta: float = 0.01  # data-loss threshold to activate hard mode
    delta: float = 0.01  # Taylor offset (recommended 1e-3 .. 0.1)
    taylor_order: int = 1  # Taylor approximation order (1 or 2)
    newton_step: float = 1.0  # KKT Newton step length
    max_newton_iter: int = 10  # max Newton iterations per projection
    pcml_projection_tolerance: float = 1e-5  # skip projection if violation < this
    # Constraint-system selection.
    constraint_type: str = "mechanical"  # "mechanical" | "thermal"
    n_joints: int = 2
    n_holonomic: int = 0
    q_bounds: Optional[List[List[float]]] = None  # [q_min, q_max]
    thermal_alpha: float = 1.0
    T_min: float = 15.0
    T_max: float = 35.0


@dataclass
class PITSMRASConfig:
    """Master configuration — the single object passed to all components."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    mras: MRASConfig = field(default_factory=MRASConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pcml: PCMLConfig = field(default_factory=PCMLConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PITSMRASConfig":
        """Build a config from a YAML file, overlaying nested fields on defaults.

        Validation is fail-loud: an unknown top-level section, or an unknown
        field within a known section, raises :class:`ValueError` naming the
        offending key rather than silently dropping it. Valid keys are taken
        from :func:`dataclasses.fields` of the master config and each
        sub-dataclass.
        """
        with open(path) as f:
            d = yaml.safe_load(f)
        # Recursively build nested dataclasses from dict.
        cfg = cls()
        if not d:
            return cfg
        valid_sections = {f.name for f in dataclasses.fields(cfg)}
        for key, val in d.items():
            if key not in valid_sections:
                raise ValueError(
                    f"Unknown config section '{key}' in {path}. "
                    f"Valid sections: {sorted(valid_sections)}."
                )
            sub = getattr(cfg, key)
            if not isinstance(val, dict):
                continue
            valid_fields = {f.name for f in dataclasses.fields(sub)}
            for k, v in val.items():
                if k not in valid_fields:
                    raise ValueError(
                        f"Unknown field '{k}' in config section '{key}' of {path}. "
                        f"Valid fields: {sorted(valid_fields)}."
                    )
                setattr(sub, k, v)
        return cfg

    def to_yaml(self, path: str) -> None:
        """Serialize the config (all nested fields) to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
