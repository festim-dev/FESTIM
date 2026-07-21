from festim.enclosure.gas_species import GasSpecies
from festim.exports.derived_quantity import DerivedQuantity


class GasPressure(DerivedQuantity):
    """Exports the partial pressure of a gas species in an enclosure over time.

    Args:
        field: the gas species to export the pressure of
        filename: name of the file to which the pressure is exported

    Attributes:
        field: the gas species to export the pressure of
        filename: name of the file to which the pressure is exported
        t: list of time values
        data: list of pressure values (Pa)
        value: the pressure at the last computed timestep (Pa)

    Examples:

        .. testsetup:: GasPressure

            import festim as F

        .. testcode:: GasPressure

            H2 = F.GasSpecies(name="H2", initial_pressure=1e5)
            my_enclosure = F.Enclosure(volume=1e-3, species=[H2], temperature=500)
            my_export = F.GasPressure(field=H2, filename="pressure.csv")
    """

    def __init__(self, field: GasSpecies, filename: str | None = None) -> None:
        super().__init__(filename=filename)
        self.field = field
        self.value = None

    @property
    def field(self) -> GasSpecies:
        return self._field

    @field.setter
    def field(self, value):
        if not isinstance(value, GasSpecies):
            raise TypeError(f"field must be a festim.GasSpecies, got {type(value)}")
        self._field = value

    @property
    def title(self) -> str:
        enclosure = self.field.enclosure
        suffix = f" ({enclosure.name})" if enclosure and enclosure.name else ""
        return f"{self.field.name} pressure{suffix} (Pa)"

    def compute(self):
        """Computes the pressure of the gas species and appends it to ``data``."""
        self.value = self.field.value
        self.data.append(self.value)
