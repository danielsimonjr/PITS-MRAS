"""Phase-1 unit tests for ``pits_mras.config`` (IP §4.2).

Covers the six sub-config dataclasses + master ``PITSMRASConfig`` defaults and
the ``from_yaml`` / ``to_yaml`` round-trip.
"""

import dataclasses

from pits_mras.config import (
    LossConfig,
    MRASConfig,
    NetworkConfig,
    PhysicsConfig,
    PITSMRASConfig,
    SafetyConfig,
    TrainingConfig,
)


def test_network_config_defaults() -> None:
    """NetworkConfig defaults match IP §4.2."""
    c = NetworkConfig()
    assert c.input_dim == 10
    assert c.hidden_dim == 128
    assert c.output_dim == 4
    assert c.lstm_layers == 2
    assert c.attention_heads == 4
    assert c.memory_horizon == 50
    assert c.embedding_dim == 64


def test_physics_config_defaults() -> None:
    """PhysicsConfig defaults match IP §4.2."""
    c = PhysicsConfig()
    assert c.n_generalized_coords == 2
    assert c.hamiltonian_hidden == 64
    assert c.dissipation_hidden == 32
    assert c.use_position_dependent_J is False


def test_mras_config_defaults() -> None:
    """MRASConfig defaults match IP §4.2."""
    c = MRASConfig()
    assert c.state_dim == 4
    assert c.control_dim == 2
    assert c.A_m is None and c.B_m is None and c.C_m is None
    assert c.Q_cost is None and c.R_cost is None
    assert c.gamma_mras == 0.1
    assert c.adapt_rate_theta == 1e-4
    assert c.adapt_rate_controller == 1e-3
    assert c.irl_window_size == 50
    assert c.use_irl_critic is True


def test_safety_config_defaults() -> None:
    """SafetyConfig defaults match IP §4.2."""
    c = SafetyConfig()
    assert c.enable_cbf is True
    assert c.safety_margin == 10.0
    assert c.cbf_decay_rate == 1.0


def test_loss_config_defaults() -> None:
    """LossConfig defaults match IP §4.2."""
    c = LossConfig()
    assert c.lambda_physics == 1.0
    assert c.lambda_temporal == 0.5
    assert c.lambda_stability == 2.0
    assert c.lambda_data == 1.0
    assert c.lambda_irl == 1.0
    assert c.lambda_hjb == 0.01
    assert c.lambda_adjoint == 0.05
    assert c.lambda_energy == 1.0
    assert c.lambda_pde == 1.0
    assert c.lambda_bc == 0.5
    assert c.lambda_sym == 0.2
    assert c.alpha_attn == 0.1
    assert c.alpha_smooth == 0.05
    assert c.mu_lyap == 0.01
    assert c.beta_param == 1e-4
    assert c.lambda_delta_u == 0.01


def test_training_config_defaults() -> None:
    """TrainingConfig defaults match IP §4.2."""
    c = TrainingConfig()
    assert c.pretrain_epochs == 5000
    assert c.pretrain_batch_size == 64
    assert c.pretrain_lr == 1e-3
    assert c.stage1_epochs == 1000
    assert c.stage2_epochs == 2000
    assert c.n_episodes == 1000
    assert c.sim_duration == 10.0
    assert c.dt == 0.01
    assert c.device in ("cuda", "cpu")
    assert c.seed == 42
    assert c.log_every == 100
    assert c.checkpoint_every == 500


def test_master_config_composition() -> None:
    """PITSMRASConfig composes all six sub-configs via default_factory."""
    cfg = PITSMRASConfig()
    assert isinstance(cfg.network, NetworkConfig)
    assert isinstance(cfg.physics, PhysicsConfig)
    assert isinstance(cfg.mras, MRASConfig)
    assert isinstance(cfg.safety, SafetyConfig)
    assert isinstance(cfg.losses, LossConfig)
    assert isinstance(cfg.training, TrainingConfig)
    # Distinct instances per construction (no shared mutable default).
    cfg2 = PITSMRASConfig()
    assert cfg.network is not cfg2.network


def test_yaml_round_trip(tmp_path) -> None:
    """to_yaml then from_yaml preserves modified nested fields."""
    cfg = PITSMRASConfig()
    cfg.network.hidden_dim = 256
    cfg.mras.gamma_mras = 0.5
    cfg.safety.enable_cbf = False
    cfg.losses.lambda_irl = 2.5

    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(path))
    assert path.exists()

    loaded = PITSMRASConfig.from_yaml(str(path))
    assert loaded.network.hidden_dim == 256
    assert loaded.mras.gamma_mras == 0.5
    assert loaded.safety.enable_cbf is False
    assert loaded.losses.lambda_irl == 2.5
    # Untouched defaults survive.
    assert loaded.network.input_dim == 10
    assert loaded.training.pretrain_epochs == 5000


def test_to_yaml_serializes_all_sections(tmp_path) -> None:
    """The serialized dict contains every top-level section."""
    cfg = PITSMRASConfig()
    path = tmp_path / "full.yaml"
    cfg.to_yaml(str(path))
    import yaml

    with open(path) as f:
        d = yaml.safe_load(f)
    for section in ("network", "physics", "mras", "safety", "losses", "training"):
        assert section in d
    # asdict round-trips field structure.
    assert set(d.keys()) == {f.name for f in dataclasses.fields(cfg)}
