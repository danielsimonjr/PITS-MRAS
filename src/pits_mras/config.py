"""Configuration dataclasses for model / controller / training (IP §4.2).

Owning phase: Phase 1 (Foundation Layer).

ARCHITECTURE.md §2.1 / §4.2 specifies dataclass config objects -- ``NetworkConfig``,
``PhysicsConfig``, ``MRASConfig``, ``SafetyConfig``, ``LossConfig``,
``TrainingConfig`` aggregated into a master ``PITSMRASConfig`` with
``from_yaml`` / ``to_yaml``.

Dependency note (scaffold task): the scaffold task brief said the *plan comments*
mention "Pydantic configs" and asked that pydantic NOT be added (it is absent from
``requirements.txt``). In the actual design docs the realized choice is stdlib
:mod:`dataclasses` (ARCHITECTURE.md §2.1 literally says "dataclass config"), so
there is no real pydantic dependency to introduce here -- the concern is moot, and
this module uses :mod:`dataclasses` as the docs direct. Should anyone revisit a
pydantic migration, that is an ADR-level dependency decision deferred to the user.

TODO(phase-1): implement the six dataclasses + master ``PITSMRASConfig`` with
``from_yaml`` / ``to_yaml`` per docs/ARCHITECTURE.md §4.2. Default dims live in
§8.2 (e.g. input_dim=10, hidden_dim=128, output_dim=4, memory_horizon=50).
"""

from dataclasses import dataclass


@dataclass
class PITSMRASConfig:
    """Master configuration placeholder aggregating all sub-configs.

    Named in ARCHITECTURE.md §4.2. The six sub-config dataclasses and the
    ``from_yaml`` / ``to_yaml`` round-trip are added in Phase 1; fields are left
    unspecified here to avoid fabricating a schema the sources do not enumerate in
    full.
    """
