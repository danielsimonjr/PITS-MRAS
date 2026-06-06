"""Data subpackage: reusable trajectory dataset, generator, and loader (Gap G7).

Factors the previously-inline synthetic-trajectory plumbing out of the training
pipelines into a small reusable, opt-in surface. Importing this package has no
effect on the existing training path unless a caller explicitly threads a
dataset / loader through ``training`` (see ``cotraining_loop(..., dataset=...)``).

Public API:

* :class:`TrajectoryDataset` -- windowed ``(state, control)`` Dataset.
* :func:`generate_synthetic_trajectories` -- seedable forward-Euler generator.
* :func:`make_dataloader` -- DataLoader convenience wrapper.
"""

from pits_mras.data.trajectory import (
    TrajectoryDataset,
    generate_synthetic_trajectories,
    make_dataloader,
)

__all__ = [
    "TrajectoryDataset",
    "generate_synthetic_trajectories",
    "make_dataloader",
]
