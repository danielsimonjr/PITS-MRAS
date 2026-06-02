"""Base classes for physics constraint specifications (PCML Addendum §2.1).

Each constraint system defines the governing DAEs of the controlled plant as
three residual families, mirroring the DAE-HardNet problem setup (Eq. 1 of
Golder et al. 2025):

- ``differential(x, t, y, d)`` -- differential equations ``D = 0`` expressed
  algebraically, with the derivative terms supplied as independent variables
  ``d`` (the Taylor-neighborhood trick that converts ODE/PDE constraints to a
  finite algebraic system in the KKT projection).
- ``equality(x, t, y)`` -- algebraic equality constraints ``h = 0``.
- ``inequality(x, t, y)`` -- inequality constraints ``g <= 0`` (a positive
  value means the constraint is violated).

The same residuals feed both the soft PCML loss (square them) and the hard KKT
projection (constrain them to zero), so the projection layer can stay
constraint-agnostic. All methods operate in batch mode (first dim is batch).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass
class ConstraintSpec:
    """Metadata about a constraint system -- how many of each residual type.

    Attributes:
        n_differential: ``|N_D|`` number of differential-equation residuals.
        n_equality: ``|N_E|`` number of algebraic equality residuals.
        n_inequality: ``|N_I|`` number of inequality residuals.
        n_outputs: ``|N_y|`` number of output variables coupled via Taylor.
    """

    n_differential: int = 0
    n_equality: int = 0
    n_inequality: int = 0
    n_outputs: int = 0


class PhysicsConstraints(ABC):
    """Abstract base for a plant's governing DAE system.

    Concrete subclasses implement :meth:`differential`, :meth:`equality` and
    :meth:`inequality`; :meth:`violation` provides the shared aggregate metric.
    """

    @property
    @abstractmethod
    def spec(self) -> ConstraintSpec:
        """Return the :class:`ConstraintSpec` metadata for this system."""
        raise NotImplementedError

    @abstractmethod
    def differential(self, x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor:
        """Differential residual ``D(x, t, y, d) = 0`` -> ``[batch, n_differential]``.

        ``d`` holds the derivative variables (first and second order) treated as
        independent algebraic unknowns.
        """
        raise NotImplementedError

    @abstractmethod
    def equality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """Algebraic equality residual ``h(x, t, y) = 0`` -> ``[batch, n_equality]``."""
        raise NotImplementedError

    @abstractmethod
    def inequality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """Inequality value ``g(x, t, y)`` (``<= 0`` is feasible) -> ``[batch, n_inequality]``.

        A positive entry means the constraint is violated.
        """
        raise NotImplementedError

    def violation(self, x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor:
        """Mean absolute constraint violation (scalar evaluation metric).

        Weighted by the per-type residual counts so it mirrors DAE-HardNet's
        Violation metric; inequality contributes only its positive (violating)
        part via ``ReLU(g)``.
        """
        spec = self.spec
        diff = self.differential(x, t, y, d)
        eq = self.equality(x, t, y)
        ineq = self.inequality(x, t, y)
        diff_v = diff.abs().mean() if diff.numel() > 0 else x.new_zeros(())
        eq_v = eq.abs().mean() if eq.numel() > 0 else x.new_zeros(())
        ineq_v = ineq.clamp(min=0.0).mean() if ineq.numel() > 0 else x.new_zeros(())
        n_total = spec.n_differential + spec.n_equality + spec.n_inequality
        weighted = (
            diff_v * spec.n_differential
            + eq_v * spec.n_equality
            + ineq_v * spec.n_inequality
        )
        return weighted / max(n_total, 1)
