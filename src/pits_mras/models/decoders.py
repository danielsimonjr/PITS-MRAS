r"""Port-Hamiltonian decoders (IP §5.2).

Owning phase: Phase 2 (Neural Network Models).

Implements Connection 2 (port-Hamiltonian storage = value). ARCHITECTURE.md §2.1
names ``HamiltonianNet``, ``DissipationNet``, and ``PortHamiltonianDecoder``
realizing :math:`\hat f_\theta = J\nabla H - R_\theta\dot q + Bu + W_{corr}c_t +
b_{corr}` with :math:`R_\theta = L^\top L \succeq 0`, :math:`J = -J^\top`,
:math:`H_\theta > 0`.

TODO(phase-2): implement per docs/ARCHITECTURE.md §5.2. Concrete signatures live
in the technical spec (Gap G4); left as a documented placeholder.
"""
