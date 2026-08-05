from collections.abc import Callable

from festim.helpers import Value


class OpeningBase:
    """Base class for enclosure openings.

    An opening lets gas in or out of an enclosure, modifying its mass balance.
    Subclasses implement :py:meth:`molar_flow_rate`.

    Args:
        species: the gas species this opening applies to. If None, the opening applies
            to every species in the enclosure.

    Attributes:
        species: the gas species this opening applies to, or None for all of them
    """

    def __init__(self, species=None):
        if species is None or isinstance(species, list | tuple):
            self.species = species
        else:
            self.species = [species]

    def applies_to(self, gas_species) -> bool:
        """Whether this opening acts on a given gas species."""
        if self.species is None:
            return True
        return gas_species in self.species

    @property
    def _values(self) -> list[Value]:
        """The Value objects held by this opening. Used to convert user input to fenics
        objects and to update time-dependent values."""
        return []

    def convert_input_values_to_fenics_objects(self, function_space, t):
        """Converts the user input values to fenics objects.

        Args:
            function_space: a function space on the parent mesh. Only its mesh is used,
                since opening parameters are scalars.
            t: the time, as a fenics Constant
        """
        for value in self._values:
            value.convert_input_value(function_space=function_space, t=t)

    def update(self, t: float):
        for value in self._values:
            if value.explicit_time_dependent:
                value.update(t=t)

    def molar_flow_rate(self, gas_species, enclosure):
        """The flow rate of particles into the enclosure (particles/s).

        A positive value means particles entering the enclosure.

        Args:
            gas_species: the gas species this rate is computed for
            enclosure: the enclosure the opening belongs to

        Returns:
            a ufl expression for the flow rate
        """
        raise NotImplementedError


class Pump(OpeningBase):
    """An opening to vacuum with a given pumping speed.

    The flow rate out of the enclosure is ``S * P / (k * T)``, giving an exponential
    decay of the pressure ``P(t) = P_0 * exp(-S * t / V)`` for a closed enclosure.

    Args:
        pumping_speed: the pumping speed (m3/s). Can be a callable of time.
        species: the gas species this pump applies to. If None, applies to all of them.

    Examples:

        .. testsetup:: Pump

            from festim import Enclosure, GasSpecies, Pump

        .. testcode:: Pump

            H2 = GasSpecies(name="H2", initial_pressure=1e5)
            my_enclosure = Enclosure(
                volume=1e-3,
                species=[H2],
                temperature=500,
                openings=[Pump(pumping_speed=1e-4)],
            )
    """

    def __init__(self, pumping_speed: float | Callable, species=None):
        super().__init__(species=species)
        self.pumping_speed = Value(pumping_speed)

    @property
    def _values(self) -> list[Value]:
        return [self.pumping_speed]

    def molar_flow_rate(self, gas_species, enclosure):
        speed = self.pumping_speed.fenics_object
        return -speed * gas_species.solution / enclosure.thermal_energy


class Reservoir(OpeningBase):
    """An opening to an external reservoir held at a given pressure.

    The flow rate into the enclosure is ``C * (P_ext - P) / (k * T)``, giving
    ``P(t) = P_ext + (P_0 - P_ext) * exp(-C * t / V)`` for an enclosure with no other
    exchange.

    Args:
        conductance: the conductance of the opening (m3/s). Can be a callable of time.
        pressure: the pressure of the reservoir (Pa). Can be a callable of time.
        species: the gas species this opening applies to. If None, applies to all.

    Examples:

        .. testsetup:: Reservoir

            from festim import Enclosure, GasSpecies, Reservoir

        .. testcode:: Reservoir

            H2 = GasSpecies(name="H2", initial_pressure=1e5)
            my_enclosure = Enclosure(
                volume=1e-3,
                species=[H2],
                temperature=500,
                openings=[Reservoir(conductance=1e-4, pressure=1e3)],
            )
    """

    def __init__(
        self, conductance: float | Callable, pressure: float | Callable, species=None
    ):
        super().__init__(species=species)
        self.conductance = Value(conductance)
        self.pressure = Value(pressure)

    @property
    def _values(self) -> list[Value]:
        return [self.conductance, self.pressure]

    def molar_flow_rate(self, gas_species, enclosure):
        conductance = self.conductance.fenics_object
        pressure = self.pressure.fenics_object
        return (
            conductance * (pressure - gas_species.solution) / enclosure.thermal_energy
        )


class PrescribedFlowRate(OpeningBase):
    """An opening with a directly imposed flow rate, independent of the pressure.

    Args:
        flow_rate: the flow rate of particles into the enclosure (particles/s). A
            negative value removes particles. Can be a callable of time.
        species: the gas species this opening applies to. If None, applies to all.

    Examples:

        .. testsetup:: PrescribedFlowRate

            from festim import Enclosure, GasSpecies, PrescribedFlowRate

        .. testcode:: PrescribedFlowRate

            H2 = GasSpecies(name="H2")
            my_enclosure = Enclosure(
                volume=1e-3,
                species=[H2],
                temperature=500,
                openings=[PrescribedFlowRate(flow_rate=1e18, species=H2)],
            )
    """

    def __init__(self, flow_rate: float | Callable, species=None):
        super().__init__(species=species)
        self.flow_rate = Value(flow_rate)

    @property
    def _values(self) -> list[Value]:
        return [self.flow_rate]

    def molar_flow_rate(self, gas_species, enclosure):
        return self.flow_rate.fenics_object


class EnclosureConnection(OpeningBase):
    """An opening connecting two enclosures, coupling their pressures.

    The flow rate into the enclosure holding ``species[0]`` is
    ``C * (P_1 - P_0) / (k * T)``, and the opposite for the other side.

    The connection only needs to be declared in the ``openings`` of one of the two
    enclosures: the mirrored term is added to the partner enclosure automatically when
    the problem is initialised.

    Args:
        conductance: the conductance of the connection (m3/s). Can be a callable of
            time.
        species: a tuple of the two gas species being connected, one in each enclosure

    Examples:

        .. testsetup:: EnclosureConnection

            from festim import Enclosure, EnclosureConnection, GasSpecies

        .. testcode:: EnclosureConnection

            H2_a = GasSpecies(name="H2", initial_pressure=1e5)
            H2_b = GasSpecies(name="H2", initial_pressure=0)
            connection = EnclosureConnection(conductance=1e-4, species=(H2_a, H2_b))
            enclosure_a = Enclosure(
                volume=1e-3, species=[H2_a], temperature=500, openings=[connection]
            )
            enclosure_b = Enclosure(volume=2e-3, species=[H2_b], temperature=500)
    """

    def __init__(self, conductance: float | Callable, species):
        if not isinstance(species, list | tuple) or len(species) != 2:
            raise ValueError(
                "EnclosureConnection needs exactly two gas species, one in each "
                "connected enclosure, eg. species=(H2_a, H2_b)"
            )
        super().__init__(species=list(species))
        self.conductance = Value(conductance)

    @property
    def _values(self) -> list[Value]:
        return [self.conductance]

    def _partner(self, gas_species):
        """The gas species on the other side of the connection."""
        if gas_species is self.species[0]:
            return self.species[1]
        elif gas_species is self.species[1]:
            return self.species[0]
        raise ValueError(f"{gas_species} is not connected by {self}")

    def molar_flow_rate(self, gas_species, enclosure):
        other = self._partner(gas_species)
        conductance = self.conductance.fenics_object
        return (
            conductance
            * (other.solution - gas_species.solution)
            / enclosure.thermal_energy
        )
