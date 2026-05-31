"""IRL-extended co-training loop (IP §8.2, Algorithm 3). "Most critical file."

Owning phase: Phase 5 (Training Pipelines).

ARCHITECTURE.md §2 / §6.2 mandates a top-level export ``cotraining_loop`` that
extends Algorithm 3 with, in order: IRL critic update (separate
``critic_optimizer`` Adam lr=1e-3, grad-clip 1.0) + policy improvement
``K_new = R^-1 B^T P_hat``; HJB update (if ``lambda_hjb > 0``); costate
consistency ``L_costate``; critic positivity ``1e-3 * L_pos``; CBF constraint
``0.1 * L_cbf``. Two optimizers (``optimizer_pitnn`` Adam lr=1e-4). The base loop
body is prose-only in the sources (Gap G5) and the full signature is not given
(Gap G6).

TODO(phase-5): implement ``cotraining_loop`` per docs/ARCHITECTURE.md §8.2.
"""
