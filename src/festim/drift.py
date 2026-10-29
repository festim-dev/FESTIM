"""Drift terms: transport driven by a gradient other than the species' own.

A drift term adds a velocity to the species flux::

    J = -D grad(c) + c v

Different physics differ only in what sets ``v``: a fluid carries the species along
(:class:`festim.AdvectionTerm`), a temperature gradient drives thermodiffusion
(:class:`SoretTerm`), an electric potential drives charged defects
(:class:`ElectromigrationTerm`). They share the assembly in :func:`drift_form`.
"""

import warnings
from abc import ABC, abstractmethod

import ufl
from dolfinx import fem

from festim import k_B
from festim.helpers import Value
from festim.mesh import CoordinateSystem
from festim.species import Species
from festim.subdomain import VolumeSubdomain


class DriftTermBase(ABC):
    """Base class for drift terms.

    Subclasses supply :meth:`drift_velocity`; everything else -- validation, the
    conversion hook and the weak form -- is shared.

    Args:
        subdomain: the volume subdomain where the drift applies
        species: the species the drift acts on

    Attributes:
        subdomain: the volume subdomain where the drift applies
        species: the species the drift acts on
    """

    subdomain: VolumeSubdomain
    species: list[Species]

    def __init__(
        self,
        subdomain: VolumeSubdomain,
        species: Species | list[Species],
    ):
        self.subdomain = subdomain
        self.species = species

    @property
    def subdomain(self):
        return self._subdomain

    @subdomain.setter
    def subdomain(self, value):
        if value is None:
            self._subdomain = value
        elif isinstance(value, VolumeSubdomain):
            self._subdomain = value
        else:
            raise TypeError(
                f"Subdomain must be a festim.Subdomain object, not {type(value)}"
            )

    @property
    def species(self) -> list[Species]:
        return self._species

    @species.setter
    def species(self, value):
        if not isinstance(value, list):
            value = [value]
        # check that all species are of type festim.Species
        for spe in value:
            if not isinstance(spe, Species):
                raise TypeError(
                    f"elements of species must be of type festim.Species not "
                    f"{type(spe)}"
                )
        self._species = value

    def convert_inputs(
        self,
        function_space: fem.FunctionSpace,
        t: fem.Constant | None = None,
        temperature=None,
    ):
        """Convert the user-given coefficients of this term to fenics objects.

        Called by the problem after the temperature is defined, so that a coefficient
        given as a function of ``T`` has one to read.

        Args:
            function_space: the function space of the problem
            t: the time, optional
            temperature: the temperature, optional
        """

    def update_time_dependent_inputs(self, t: float):
        """Re-evaluate the explicitly time-dependent coefficients of this term.

        Called once per time step. The default updates every :class:`Value` returned by
        :meth:`time_dependent_inputs`.

        Args:
            t: the time
        """
        for value in self.time_dependent_inputs():
            if value.explicit_time_dependent:
                value.update(t=t)

    def time_dependent_inputs(self) -> list[Value]:
        """The user-given coefficients of this term that may depend on time."""
        return []

    @abstractmethod
    def drift_velocity(self, D, temperature):
        """The drift velocity of this term, as a ufl expression.

        Args:
            D: the diffusion coefficient of the species in this subdomain
            temperature: the temperature on the mesh the term is assembled on

        Returns:
            the velocity, a vector-valued ufl expression. ``ufl.zero`` when the driving
            gradient vanishes, in which case the term is skipped.
        """

    def inert_reason(self) -> str:
        """Why this term's velocity came out identically zero, for the user.

        Reported when the term is dropped, so the message can name the input at fault
        rather than leaving the user to work out why the physics did nothing.
        """
        return "its drift velocity is identically zero"


class SoretTerm(DriftTermBase):
    r"""Thermodiffusion -- the Soret effect, also called thermophoresis.

    Adds the temperature-gradient term of the flux

    .. math::

        J = -D \nabla c - D \frac{Q^* c}{k_B T^2} \nabla T

    so hydrogen drifts down the temperature gradient for a positive heat of transport.
    The term vanishes when the temperature is uniform.

    Args:
        species: the species the drift acts on
        Q_star: the heat of transport :math:`Q^*` in eV. A float, a callable of ``x``,
            ``t`` and/or ``T``, or a fenics object
        subdomain: the volume subdomain where the drift applies

    Attributes:
        Q_star: the heat of transport, wrapped in a :class:`festim.helpers.Value`

    Examples:
        .. code-block:: python

            import festim as F

            F.SoretTerm(species=mobile_H, Q_star=0.2, subdomain=my_volume)
    """

    def __init__(
        self,
        species: Species | list[Species],
        Q_star,
        subdomain: VolumeSubdomain,
    ):
        super().__init__(subdomain=subdomain, species=species)
        self.Q_star = Value(Q_star)

    def convert_inputs(self, function_space, t=None, temperature=None):
        self.Q_star.convert_input_value(
            function_space=function_space,
            t=t,
            temperature=temperature,
            up_to_ufl_expr=True,
        )

    def time_dependent_inputs(self):
        return [self.Q_star]

    def drift_velocity(self, D, temperature):
        Q_star = self.Q_star.fenics_object
        return -D * Q_star / (k_B * temperature**2) * ufl.grad(temperature)

    def inert_reason(self) -> str:
        return (
            "the temperature is spatially uniform, so grad(T) is zero. Give the "
            "temperature as a function of x, or couple a heat transfer problem with "
            "festim.CoupledTransientHeatTransferHydrogenTransport"
        )


class ElectromigrationTerm(DriftTermBase):
    r"""Electromigration of a charged species in an electric potential.

    The Nernst-Planck drift term of the flux

    .. math::

        J = -D \nabla c - \frac{z D c}{k_B T} \nabla \varphi

    with :math:`\varphi` in volts and :math:`z` the charge number, so that
    :math:`k_B` in eV/K carries the elementary charge. Positive species drift down the
    potential gradient. The term vanishes when the potential is uniform.

    Args:
        species: the species the drift acts on
        charge: the charge number :math:`z` of the species (``+1`` for a proton,
            ``+2`` for an oxygen vacancy, ``-1`` for an electron)
        potential: the electric potential in V. A float, a callable of ``x``, ``t``
            and/or ``T``, or a fenics object
        subdomain: the volume subdomain where the drift applies

    Attributes:
        charge: the charge number of the species
        potential: the potential, wrapped in a :class:`festim.helpers.Value`

    Examples:
        .. code-block:: python

            import festim as F

            F.ElectromigrationTerm(
                species=hydroxyl,
                charge=1,
                potential=lambda x: 0.5 * (1 - x[0] / 5e-4),
                subdomain=membrane,
            )
    """

    def __init__(
        self,
        species: Species | list[Species],
        charge: float,
        potential,
        subdomain: VolumeSubdomain,
    ):
        super().__init__(subdomain=subdomain, species=species)
        self.charge = charge
        self.potential = Value(potential)

    def convert_inputs(self, function_space, t=None, temperature=None):
        self.potential.convert_input_value(
            function_space=function_space,
            t=t,
            temperature=temperature,
            up_to_ufl_expr=True,
        )

    def time_dependent_inputs(self):
        return [self.potential]

    def drift_velocity(self, D, temperature):
        potential = self.potential.fenics_object
        return -self.charge * D / (k_B * temperature) * ufl.grad(potential)

    def inert_reason(self) -> str:
        return (
            "the potential is spatially uniform, so grad(phi) is zero. Give the "
            "potential as a function of x"
        )


def is_zero_velocity(velocity) -> bool:
    """Whether a drift velocity is identically zero.

    ``ufl.grad`` of a spatially constant field is a ``Zero`` at construction, and that
    propagates through the arithmetic of a drift velocity. This is a *structural* zero,
    so it is zero at every time -- a velocity that merely happens to vanish at ``t=0``
    is a ``Function`` and is never reported here.

    Such a term contributes nothing to a form it is added to: UFL sums the integrands
    sharing a measure, so the zero collapses before FFCx sees it. It is used to warn the
    user, and to keep a form that would consist of *nothing but* zero integrands from
    being built at all -- such a form compiles to one with no arguments, which cannot be
    assembled.
    """
    return isinstance(velocity, ufl.constantvalue.Zero)


def warn_if_no_effect(term: DriftTermBase, species: Species, velocity) -> bool:
    """Warn that ``term`` contributes nothing, and say why.

    Writing a drift term is a statement of intent, so a term whose driving gradient
    vanishes is almost always an input that is uniform when the user meant it to vary.
    The term is still assembled -- it costs nothing, and quietly discarding what the
    user asked for is worse than carrying a term that contributes zero -- but it is
    worth saying so.

    Callers that would otherwise build a form out of nothing but this term use the
    return value to skip it; see :func:`is_zero_velocity`.

    Args:
        term: the drift term
        species: the species it would have acted on
        velocity: the velocity it produced

    Returns:
        True if the term has no effect
    """
    if not is_zero_velocity(velocity):
        return False
    subdomain_id = getattr(term.subdomain, "id", term.subdomain)
    warnings.warn(
        f"{type(term).__name__} on volume subdomain {subdomain_id} has no effect on "
        f"species {species.name}: {term.inert_reason()}.",
        stacklevel=2,
    )
    return True


def drift_form(
    concentration,
    test_function,
    velocity,
    dx: ufl.Measure,
    coordinate_system: CoordinateSystem,
    mesh,
):
    r"""The weak form contribution of a drift term, in divergence form.

    .. math::

        -\int c\, \mathbf{v} \cdot \nabla w

    which is :math:`\nabla \cdot (c \mathbf{v})` integrated by parts. Because the
    boundary term it leaves behind is the natural boundary condition, the flux boundary
    conditions of the problem constrain the **total** flux, diffusive and drift alike,
    and a boundary with no condition on it is genuinely no-flux.

    The metric factor matters here. FESTIM multiplies the equation by ``m`` (``r`` in
    cylindrical, ``r^2`` in spherical) and uses ``w / m`` as the test function; those do
    not cancel in this form, unlike in the non-conservative
    :math:`\int (\mathbf{v} \cdot \nabla c)\, w` that FESTIM assembled before.

    Args:
        concentration: the concentration of the species
        test_function: the test function of the species
        velocity: the drift velocity, a vector-valued ufl expression
        dx: the volume measure, already indexed by subdomain id
        coordinate_system: the coordinate system of the mesh
        mesh: the mesh the term is assembled on

    Returns:
        the ufl form to add to the problem's formulation

    Raises:
        NotImplementedError: for an unknown coordinate system
    """
    match coordinate_system:
        case CoordinateSystem.CARTESIAN:
            return -concentration * ufl.dot(velocity, ufl.grad(test_function)) * dx
        case CoordinateSystem.CYLINDRICAL:
            r = ufl.SpatialCoordinate(mesh)[0]
            return (
                -r * concentration * ufl.dot(velocity, ufl.grad(test_function / r)) * dx
            )
        case CoordinateSystem.SPHERICAL:
            r = ufl.SpatialCoordinate(mesh)[0]
            return (
                -(r**2)
                * concentration
                * ufl.dot(velocity, ufl.grad(test_function / r**2))
                * dx
            )
        case _:
            raise NotImplementedError(
                f"Unknown coordinate system {coordinate_system!s}"
            )
