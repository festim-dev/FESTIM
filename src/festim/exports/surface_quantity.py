from abc import abstractmethod

from festim.exports.derived_quantity import DerivedQuantity
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class SurfaceQuantity(DerivedQuantity):
    """Export SurfaceQuantity.

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain
        filename: name of the file to which the surface flux is exported

    Attributes:
        field: species for which the surface flux is computed
        surface: surface subdomain
        filename: name of the file to which the surface flux is exported
        t: list of time values
        data: list of values of the surface quantity
    """

    field: Species
    surface: SurfaceSubdomain
    filename: str | None

    t: list[float]
    data: list[float]

    def __init__(
        self,
        field: Species | str,
        surface: SurfaceSubdomain | int,
        filename: str | None = None,
    ) -> None:
        super().__init__(filename=filename)
        self.field = field
        self.surface = surface

    @property
    @abstractmethod
    def title(self):
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, value):
        if not isinstance(value, int | SurfaceSubdomain) or isinstance(value, bool):
            raise TypeError("surface should be an int or F.SurfaceSubdomain")

        self._surface = value

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        if not isinstance(value, Species | str):
            raise TypeError("field must be of type F.Species or str")

        self._field = value
