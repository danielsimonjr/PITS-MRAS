"""Soft Actor-Critic module tests (Connection 5 — max-entropy RL).

Covers :class:`pits_mras.models.sac.GaussianPolicy`,
:class:`pits_mras.models.sac.TwinQCritic`, and
:class:`pits_mras.training.sac.SACTrainer`.

Structural checks pin shapes / bounds / differentiability / soft-update. The
crux is the FALSIFIABLE LEARNING SANITY: on a single-step contextual-bandit
problem with reward ``-||a - target||^2``, the deterministic policy mean must
move substantially closer to ``target`` after training than at init. Written as
a loose ratio (final < 0.5 * initial) with a fixed seed to stay non-flaky.
"""

import torch

from pits_mras.models.sac import GaussianPolicy, TwinQCritic
from pits_mras.training.sac import SACTrainer


# --------------------------------------------------------------------------- #
# GaussianPolicy.
# --------------------------------------------------------------------------- #
def test_policy_sample_shapes() -> None:
    """sample returns action [B, A], log_prob [B, 1], mean_action [B, A]."""
    pol = GaussianPolicy(state_dim=3, action_dim=2, hidden=(16, 16))
    s = torch.randn(5, 3)
    action, log_prob, mean_action = pol.sample(s)
    assert action.shape == (5, 2)
    assert log_prob.shape == (5, 1)
    assert mean_action.shape == (5, 2)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()


def test_policy_actions_bounded_by_scale() -> None:
    """Tanh squashing keeps actions in [-scale, scale]."""
    scale = 2.5
    pol = GaussianPolicy(state_dim=4, action_dim=3, action_scale=scale)
    s = torch.randn(64, 4)
    action, _, mean_action = pol.sample(s)
    assert action.abs().max().item() <= scale + 1e-5
    assert mean_action.abs().max().item() <= scale + 1e-5


def test_policy_reparam_is_differentiable() -> None:
    """Reparameterized sample flows grad to the policy parameters."""
    pol = GaussianPolicy(state_dim=3, action_dim=2, hidden=(16, 16))
    s = torch.randn(8, 3)
    action, log_prob, _ = pol.sample(s)
    loss = action.pow(2).mean() + log_prob.mean()
    loss.backward()
    grads = [p.grad for p in pol.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0.0 for g in grads)


def test_policy_mean_path_shape_and_determinism() -> None:
    """mean() is deterministic and shaped [B, A]."""
    pol = GaussianPolicy(state_dim=3, action_dim=2)
    s = torch.randn(7, 3)
    m1 = pol.mean(s)
    m2 = pol.mean(s)
    assert m1.shape == (7, 2)
    assert torch.allclose(m1, m2)


# --------------------------------------------------------------------------- #
# TwinQCritic.
# --------------------------------------------------------------------------- #
def test_twin_q_shapes() -> None:
    """forward returns two [B, 1] Q-values; q_min is [B, 1]."""
    critic = TwinQCritic(state_dim=3, action_dim=2, hidden=(16, 16))
    s = torch.randn(5, 3)
    a = torch.randn(5, 2)
    q1, q2 = critic(s, a)
    assert q1.shape == (5, 1)
    assert q2.shape == (5, 1)
    assert critic.q_min(s, a).shape == (5, 1)


def test_twin_q_min_is_elementwise_min() -> None:
    """q_min equals elementwise min of the two heads."""
    critic = TwinQCritic(state_dim=3, action_dim=2)
    s = torch.randn(6, 3)
    a = torch.randn(6, 2)
    q1, q2 = critic(s, a)
    assert torch.allclose(critic.q_min(s, a), torch.min(q1, q2))


# --------------------------------------------------------------------------- #
# SACTrainer.update structural checks.
# --------------------------------------------------------------------------- #
def _synthetic_batch(state_dim: int, action_dim: int, B: int = 32) -> dict[str, torch.Tensor]:
    return {
        "state": torch.randn(B, state_dim),
        "action": torch.tanh(torch.randn(B, action_dim)),
        "reward": torch.randn(B, 1),
        "next_state": torch.randn(B, state_dim),
        "done": torch.randint(0, 2, (B, 1)).float(),
    }


def test_update_returns_finite_losses_and_positive_alpha() -> None:
    """One update yields finite losses and a positive alpha."""
    trainer = SACTrainer(state_dim=3, action_dim=2, hidden=(16, 16), seed=0)
    out = trainer.update(_synthetic_batch(3, 2))
    for key in ("critic_loss", "actor_loss", "alpha_loss", "alpha", "log_prob"):
        assert key in out
        assert torch.isfinite(torch.tensor(out[key]))
    assert out["alpha"] > 0.0


def test_update_changes_policy_and_critic_params() -> None:
    """A single step changes both policy and critic parameters."""
    trainer = SACTrainer(state_dim=3, action_dim=2, hidden=(16, 16), seed=1)
    pol_before = [p.detach().clone() for p in trainer.policy.parameters()]
    cri_before = [p.detach().clone() for p in trainer.critic.parameters()]
    trainer.update(_synthetic_batch(3, 2))
    pol_changed = any(
        not torch.allclose(b, a) for b, a in zip(pol_before, trainer.policy.parameters())
    )
    cri_changed = any(
        not torch.allclose(b, a) for b, a in zip(cri_before, trainer.critic.parameters())
    )
    assert pol_changed
    assert cri_changed


def test_log_alpha_has_gradient() -> None:
    """The temperature parameter receives a gradient on update."""
    trainer = SACTrainer(state_dim=3, action_dim=2, hidden=(16, 16), seed=2)
    trainer.update(_synthetic_batch(3, 2))
    assert trainer.log_alpha.grad is not None
    assert torch.isfinite(trainer.log_alpha.grad).all()


def test_soft_update_moves_target_toward_online() -> None:
    """After an update the target critic is a Polyak blend toward the online net."""
    tau = 0.1
    trainer = SACTrainer(state_dim=3, action_dim=2, hidden=(16, 16), tau=tau, seed=3)
    # Targets start equal to online; snapshot that initial value.
    init_target = [p.detach().clone() for p in trainer.critic_target.parameters()]
    for p, pt in zip(trainer.critic.parameters(), trainer.critic_target.parameters()):
        assert torch.allclose(p, pt)

    trainer.update(_synthetic_batch(3, 2))

    moved = False
    for online, tgt, t0 in zip(
        trainer.critic.parameters(), trainer.critic_target.parameters(), init_target
    ):
        # Online drifted away from the shared init.
        if not torch.allclose(online, t0):
            moved = True
        # Polyak: tgt == (1 - tau) * t0 + tau * online (exact, no grad on tgt).
        expected = (1.0 - tau) * t0 + tau * online.detach()
        assert torch.allclose(tgt, expected, atol=1e-6)
    assert moved


# --------------------------------------------------------------------------- #
# Falsifiable learning sanity: contextual bandit, reward = -||a - target||^2.
# --------------------------------------------------------------------------- #
def test_sac_learns_to_reach_target() -> None:
    """Deterministic policy mean moves substantially toward a fixed target.

    Single-step (done=1) bandit: reward = -||a - target||^2, target fixed and
    state-independent. SAC should drive the greedy action toward target. Loose
    ratio (final < 0.5 * initial distance) with a fixed seed.
    """
    torch.manual_seed(0)
    state_dim, action_dim = 2, 2
    target = torch.tensor([[0.6, -0.4]])
    trainer = SACTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=(64, 64),
        lr=3e-3,
        action_scale=1.0,
        seed=0,
    )

    eval_state = torch.zeros(1, state_dim)

    def _dist(state: torch.Tensor) -> float:
        with torch.no_grad():
            a = trainer.policy.mean(state)
        return float((a - target).norm())

    init_dist = _dist(eval_state)

    gen = torch.Generator().manual_seed(0)
    for _ in range(800):
        s = torch.randn(128, state_dim, generator=gen)
        # Sample actions from current policy to build off-policy transitions.
        with torch.no_grad():
            a, _, _ = trainer.policy.sample(s)
        r = -((a - target) ** 2).sum(dim=-1, keepdim=True)
        batch = {
            "state": s,
            "action": a,
            "reward": r,
            "next_state": torch.randn(128, state_dim, generator=gen),
            "done": torch.ones(128, 1),  # single-step bandit
        }
        trainer.update(batch)

    final_dist = _dist(eval_state)
    assert (
        final_dist < 0.5 * init_dist
    ), f"SAC did not learn: init={init_dist:.3f} final={final_dist:.3f}"
