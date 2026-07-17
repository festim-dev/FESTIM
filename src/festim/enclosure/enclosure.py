from collections.abc import Callable

from festim import k_B_SI
from festim.enclosure.gas_species import GasSpecies
from festim.enclosure.openings import OpeningBase
from festim.helpers import Value
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class Enclosure:
    """A gas enclosure in contact with the model through one or more surfaces.

    The partial pressure of each gas species in the enclosure is an unknown of the
    problem, solved together with the transport problem. For each species, the pressure
    evolves as:

    .. math::

        \\frac{dP}{dt} = \\frac{k_B T}{V} \\left( \\sum_\\Gamma A_\\Gamma
        \\int_\\Gamma \\varphi \\, dS + \\sum_\\text{openings} Q \\right)

    where :math:`\\varphi` is the rate of particles entering the gas from the solid,
    :math:`Q` the flow rate through the openings, and :math:`A_\\Gamma` the physical
    area of each contact surface (see ``surfaces``).

    Args:
        volume: the volume of the enclosure (m3)
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

            An enclosure with no contact surfaces is allowed (it then only exchanges
            through its openings).
        openings: the openings of the enclosure (see :py:class:`festim.Pump`,
            :py:class:`festim.Reservoir`, :py:class:`festim.PrescribedFlowRate`,
            :py:class:`festim.EnclosureConnection`)
        gas_constant: the constant relating pressure to particle density
            (:math:`P = n k T`). Defaults to :py:data:`festim.k_B_SI` (J/K), which
            matches FESTIM's convention of concentrations in particles/m3. Pass
            :py:data:`festim.R` (J/mol/K) if working in mol/m3.
        name: a name given to the enclosure

    Attributes:
        volume: the volume of the enclosure (m3)
        species: the gas species in the enclosure
        temperature: the temperature of the gas, wrapped in a
            :py:class:`festim.helpers.Value`
        surfaces: a dict mapping each contact surface to its physical area
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
    """

    def __init__(
        self,
        volume: float,
        species: list[GasSpecies],
        temperature: float | Callable,
        surfaces: dict[SurfaceSubdomain, float] | list[SurfaceSubdomain] | None = None,
        openings: list[OpeningBase] | None = None,
        gas_constant: float = k_B_SI,
        name: str | None = None,
    ):
        self.volume = volume
        self.species = species
        self.temperature = Value(temperature)
        self.surfaces = surfaces
        self.openings = openings or []
        self.gas_constant = gas_constant
        self.name = name

        for gas_species in self.species:
            gas_species.enclosure = self

    def __repr__(self) -> str:
        return f"Enclosure({self.name})" if self.name else "Enclosure"

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value):
        if value <= 0:
            raise ValueError(f"Enclosure volume must be positive, got {value}")
        self._volume = value

    @property
    def surfaces(self) -> dict[SurfaceSubdomain, float]:
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
        for surface, area in value.items():
            if not isinstance(surface, SurfaceSubdomain):
                raise TypeError(
                    "surfaces must map festim.SurfaceSubdomain objects to their area, "
                    f"got a key of type {type(surface)}"
                )
            if area <= 0:
                raise ValueError(
                    f"The area of surface {surface.id} must be positive, got {area}"
                )
        self._surfaces = value

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

    def convert_input_values_to_fenics_objects(self, function_space, t):
        """Converts the user input values of the enclosure and its openings to fenics
        objects.

        Args:
            function_space: a function space on the parent mesh
            t: the time, as a fenics Constant
        """
        self.temperature.convert_input_value(function_space=function_space, t=t)
        for opening in self.openings:
            opening.convert_input_values_to_fenics_objects(
                function_space=function_space, t=t
            )

    def update_time_dependent_values(self, t: float):
        """Updates the time-dependent values of the enclosure and its openings.

        Args:
            t: the time
        """
        if self.temperature.explicit_time_dependent:
            self.temperature.update(t=t)
        for opening in self.openings:
            opening.update(t=t)
