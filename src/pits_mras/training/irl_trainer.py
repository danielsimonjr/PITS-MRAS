r"""Standalone offline IRL critic trainer (IP §8.3).

Owning phase: Phase 5 (Training Pipelines).

ARCHITECTURE.md §2.1 / §6.3: offline batch least-squares critic pre-training from
a fixed dataset of trajectories; runs Kleinman-style batch least-squares on the
IRL Bellman equations and stops when
:math:`\|\hat P - P_{opt}\|_F / \|P_{opt}\|_F < 0.01`. The trajectory dataset
format/loader is unspecified in the sources (Gap G7).

TODO(phase-5): implement per docs/ARCHITECTURE.md §8.3.
"""
