"""Training subpackage: physics pretrain, IRL co-train, offline IRL trainer.

Owning phase: Phase 5 (Training Pipelines) per ROADMAP.md / ARCHITECTURE.md
§8. Re-exports the three public training entry points now that the pipelines
are implemented.
"""

from pits_mras.training.cotrain import cotraining_loop
from pits_mras.training.hinf_minmax import (
    hinf_minmax_from_dynamics,
    hinf_minmax_from_pitnn,
    hinf_minmax_train,
    hji_residual,
    pitnn_one_step,
)
from pits_mras.training.irl_trainer import train_irl_critic
from pits_mras.training.pretrain import pretrain_pitnn
from pits_mras.training.sac import SACTrainer
from pits_mras.training.tdmpc import tdmpc_update

__all__ = [
    "pretrain_pitnn",
    "cotraining_loop",
    "train_irl_critic",
    "hinf_minmax_train",
    "hinf_minmax_from_dynamics",
    "hinf_minmax_from_pitnn",
    "pitnn_one_step",
    "hji_residual",
    "SACTrainer",
    "tdmpc_update",
]
