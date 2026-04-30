from festim.exports.derived_quantity import DerivedQuantity
from festim.species import Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class VolumeQuantity(DerivedQuantity):
    """Export VolumeQuantity.

    Args:
        field: species for which the volume quantity is computed
        volume: volume subdomain
        filename: name of the file to which the volume quantity is exported

    Attributes:
        field: species for which the volume quantity is computed
        volume: volume subdomain
        filename: name of the file to which the volume quantity is exported
        t: list of time values
        data: list of values of the volume quantity
    """

    field: Species
    volume: VolumeSubdomain
    filename: str | None

    t: list[float]
    data: list[float]

    def __init__(
        self,
        field: Species | str,
        volume: VolumeSubdomain | int,
        filename: str | None = None,
    ) -> None:
        super().__init__(filename=filename)
        self.field = field
        self.volume = volume

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        if not isinstance(value, Species | str):
            raise TypeError("field must be of type F.Species or str")

        self._field = value
