from collections.abc import Callable

from dolfinx import fem

from festim import k_B_SI
from festim.enclosure.gas_species import GasSpecies
from festim.enclosure.openings import OpeningBase
from festim.helpers import Value, as_fenics_constant
from festim.subdomain.surface_subdomain import SurfaceSubdomain


def check_positive(value: float, description: str):
    """Checks that a value of an enclosure parameter is strictly positive.

    Args:
        value: the value to check
        description: how to refer to the parameter in the error message

    Raises:
        ValueError: if the value is not strictly positive
    """
    if value <= 0:
        raise ValueError(f"{description} must be positive, got {value}")


def as_scalar_value(value, description: str) -> Value:
    """Wraps a user input for an enclosure parameter in a
    :py:class:`festim.helpers.Value`.

    An enclosure is a 0D gas volume: its parameters are uniform scalars. They may vary
    in time, but not in space, and they cannot depend on the temperature of the
    transport problem -- the gas has a temperature of its own.

    Args:
        value: the user input: a positive number, a callable of time, or a fenics
            Constant
        description: how to refer to the parameter in error messages

    Returns:
        the wrapped value

    Raises:
        TypeError: if the value is not a number, a callable or a fenics Constant
        ValueError: if the value is not positive, or is a callable of anything but time
    """
    if isinstance(value, fem.Constant):
        pass
    elif isinstance(value, float | int):
        check_positive(value, description)
    elif callable(value):
        code = getattr(value, "__code__", None)
        forbidden = [
            name for name in ("x", "T") if code is not None and name in code.co_varnames
        ]
        if forbidden:
            raise ValueError(
                f"{description} can only be a callable of time, but its arguments "
                f"include {', '.join(forbidden)}. An enclosure is a 0D gas volume: its "
                "parameters are uniform in space, and the temperature of the gas is "
                "independent of the temperature of the transport problem."
            )
    else:
        raise TypeError(
            f"{description} must be a number, a callable of time or a fenics Constant, "
            f"got {type(value)}"
        )
    return Value(value)


class Enclosure:
    """A gas enclosure in contact with the model through one or more surfaces.

    The partial pressure of each gas species in the enclosure is an unknown of the
    problem, solved together with the transport problem. The balance is written on the
    number of particles :math:`N = PV/(k_B T)` rather than on the pressure itself, so
    that an enclosure whose volume or temperature changes over time neither gains nor
    loses particles by doing so. For each species:

    .. math::

        \\frac{d}{dt} \\left( \\frac{P V}{k_B T} \\right) = \\sum_\\Gamma A_\\Gamma
        \\int_\\Gamma \\varphi \\, dS + \\sum_\\text{openings} Q

    where :math:`\\varphi` is the rate of particles entering the gas from the solid,
    :math:`Q` the flow rate through the openings, and :math:`A_\\Gamma` the physical
    area of each contact surface (see ``surfaces``).

    The volume, the temperature and the contact areas can all be given as callables of
    time. Compressing a closed enclosure therefore raises its pressure, and heating it
    at constant volume does too.

    Args:
        volume: the volume of the enclosure (m3). Can be a callable of time.
        species: the gas species in the enclosure
        temperature: the temperature of the gas (K). Can be a callable of time. This is
            independent of the temperature of the transport problem.
        surfaces: the surfaces in contact with the enclosure, as a dict mapping each
            :py:class:`festim.SurfaceSubdomain` to its physical area. The area is what
            turns the flux through the surface into a number of particles per second,
            and the mesh only provides it in 3D:

            - **1D**: a surface is a point and carries no extent, so the area is the
              area of the membrane facing the enclosure, in m2. Required.
            - **2D**: a surface is a line, so the area is the out-of-plane depth of the
              model, in m. Required.
            - **3D**: the mesh already measures the area, so pass 1.0. A plain list of
              surfaces is accepted in 3D and means an area of 1.0 for each.

            Each area can be a callable of time. An enclosure with no contact surfaces
            is allowed (it then only exchanges through its openings).
        openings: the openings of the enclosure (see :py:class:`festim.Pump`,
            :py:class:`festim.Reservoir`, :py:class:`festim.PrescribedFlowRate`,
            :py:class:`festim.EnclosureConnection`)
        gas_constant: the constant relating pressure to particle density
            (:math:`P = n k T`). Defaults to :py:data:`festim.k_B_SI` (J/K), which
            matches FESTIM's convention of concentrations in particles/m3. Pass
            :py:data:`festim.R` (J/mol/K) if working in mol/m3.
        name: a name given to the enclosure

    Attributes:
        volume: the volume of the enclosure (m3), wrapped in a
            :py:class:`festim.helpers.Value`
        species: the gas species in the enclosure
        temperature: the temperature of the gas, wrapped in a
            :py:class:`festim.helpers.Value`
        surfaces: a dict mapping each contact surface to its physical area, each
            wrapped in a :py:class:`festim.helpers.Value`
        openings: the openings of the enclosure
        gas_constant: the constant relating pressure to particle density
        name: a name given to the enclosure

    Examples:

        .. testsetup:: Enclosure

            import festim as F

        .. testcode:: Enclosure

            H2 = F.GasSpecies(name="H2", initial_pressure=1e5)
            my_enclosure = F.Enclosure(
                volume=1e-3,
                species=[H2],
                temperature=500,
                openings=[F.Pump(pumping_speed=1e-4)],
            )

        A vessel being inflated, in contact with a membrane whose exposed area grows:

        .. testcode:: Enclosure

            my_enclosure = F.Enclosure(
                volume=lambda t: 1e-3 * (1 + 0.01 * t),
                species=[F.GasSpecies(name="H2", initial_pressure=1e5)],
                temperature=lambda t: 300 + t,
            )
    """

    def __init__(
        self,
        volume: float | Callable,
        species: list[GasSpecies],
        temperature: float | Callable,
        surfaces: dict[SurfaceSubdomain, float | Callable]
        | list[SurfaceSubdomain]
        | None = None,
        openings: list[OpeningBase] | None = None,
        gas_constant: float = k_B_SI,
        name: str | None = None,
    ):
        self.volume = volume
        self.species = species
        self.temperature = temperature
        self.surfaces = surfaces
        self.openings = openings or []
        self.gas_constant = gas_constant
        self.name = name

        # the volume and temperature of the previous timestep, needed by the particle
        # balance. Created when the input values are converted to fenics objects.
        self._prev_volume = None
        self._prev_temperature = None

        for gas_species in self.species:
            gas_species.enclosure = self

    def __repr__(self) -> str:
        return f"Enclosure({self.name})" if self.name else "Enclosure"

    @property
    def volume(self) -> Value:
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = as_scalar_value(value, "Enclosure volume")

    @property
    def temperature(self) -> Value:
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = as_scalar_value(value, "Enclosure temperature")

    @property
    def surfaces(self) -> dict[SurfaceSubdomain, Value]:
        return self._surfaces

    @surfaces.setter
    def surfaces(self, value):
        if value is None:
            value = {}
        # a plain list means "no areas given"; only valid in 3D, where the mesh
        # measures the area itself. Checked against the mesh in the problem class.
        self.areas_given = isinstance(value, dict)
        if not self.areas_given:
            value = dict.fromkeys(value, 1.0)
        surfaces = {}
        for surface, area in value.items():
            if not isinstance(surface, SurfaceSubdomain):
                raise TypeError(
                    "surfaces must map festim.SurfaceSubdomain objects to their area, "
                    f"got a key of type {type(surface)}"
                )
            surfaces[surface] = as_scalar_value(
                area, f"The area of surface {surface.id}"
            )
        self._surfaces = surfaces

    @property
    def species(self) -> list[GasSpecies]:
        return self._species

    @species.setter
    def species(self, value):
        if not value:
            raise ValueError("Enclosure must have at least one GasSpecies")
        if not isinstance(value, list | tuple):
            raise TypeError("species must be a list of GasSpecies")
        for gas_species in value:
            if not isinstance(gas_species, GasSpecies):
                raise TypeError(
                    f"species must be a list of GasSpecies, got {type(gas_species)}"
                )
        self._species = list(value)

    @property
    def thermal_energy(self):
        """The quantity relating pressure to particle density (:math:`P = n k T`).

        Returns ``gas_constant * T`` as a fenics object once the problem has been
        initialised.
        """
        return self.gas_constant * self.temperature.fenics_object

    @property
    def previous_thermal_energy(self):
        """:py:attr:`thermal_energy` at the previous timestep."""
        return self.gas_constant * self._prev_temperature

    @property
    def previous_volume(self):
        """The volume at the previous timestep, as a fenics Constant."""
        return self._prev_volume

    @property
    def _values(self) -> list[Value]:
        """The Value objects held by the enclosure itself, not counting its openings."""
        return [self.volume, self.temperature, *self.surfaces.values()]

    @property
    def _positive_values(self) -> list[tuple[str, Value]]:
        """The values that have to stay positive, with how to refer to them."""
        return [
            ("Enclosure volume", self.volume),
            ("Enclosure temperature", self.temperature),
            *(
                (f"The area of surface {surface.id}", area)
                for surface, area in self.surfaces.items()
            ),
        ]

    def convert_input_values_to_fenics_objects(self, function_space, t):
        """Converts the user input values of the enclosure and its openings to fenics
        objects.

        Args:
            function_space: a function space on the parent mesh
            t: the time, as a fenics Constant
        """
        for value in self._values:
            value.convert_input_value(function_space=function_space, t=t)
        for opening in self.openings:
            opening.convert_input_values_to_fenics_objects(
                function_space=function_space, t=t
            )
        self.check_positive_values()

        # the balance is written on the number of particles P*V/(k*T), so it needs the
        # volume and the temperature of the previous timestep as well as the current
        # ones
        mesh = function_space.mesh
        self._prev_volume = as_fenics_constant(
            float(self.volume.fenics_object), mesh=mesh
        )
        self._prev_temperature = as_fenics_constant(
            float(self.temperature.fenics_object), mesh=mesh
        )

    def check_positive_values(self):
        """Checks that the values that have to stay positive still are at the current
        time.

        A number given by the user is checked when it is given, but a callable of time
        can only be checked as it is evaluated.

        Raises:
            ValueError: if the volume, the temperature or an area is not positive
        """
        for description, value in self._positive_values:
            if value.explicit_time_dependent:
                check_positive(float(value.fenics_object), description)

    def update_time_dependent_values(self, t: float):
        """Updates the time-dependent values of the enclosure and its openings.

        Args:
            t: the time
        """
        for value in self._values:
            if value.explicit_time_dependent:
                value.update(t=t)
        self.check_positive_values()
        for opening in self.openings:
            opening.update(t=t)

    def update_previous_values(self):
        """Stores the current volume and temperature as those of the previous timestep.

        Called at the end of a timestep, once the pressures have been solved for.
        """
        self._prev_volume.value = float(self.volume.fenics_object)
        self._prev_temperature.value = float(self.temperature.fenics_object)
