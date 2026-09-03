from abc import abstractmethod

from festim.exports.derived_quantity import DerivedQuantity
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain
from festim.subdomain.volume_subdomain import VolumeSubdomain


class SurfaceQuantity(DerivedQuantity):
    """Export SurfaceQuantity.

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain. A codim-1 ``VolumeSubdomain`` (a manifold) may be
            given instead, to compute the quantity on the facets it occupies -- for a
            *bulk* species, since a manifold's own species has no flux across it. A
            codim-2 ``SurfaceSubdomain`` computes it on the boundary of a manifold.
        filename: name of the file to which the surface flux is exported

    Attributes:
        field: species for which the surface flux is computed
        surface: surface subdomain
        filename: name of the file to which the surface flux is exported
        t: list of time values
        data: list of values of the surface quantity
    """

    field: Species
    surface: SurfaceSubdomain | VolumeSubdomain
    filename: str | None

    t: list[float]
    data: list[float]

    def __init__(
        self,
        field: Species | str,
        surface: SurfaceSubdomain | VolumeSubdomain | int,
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
        # a manifold is a codim-1 VolumeSubdomain and is passed directly wherever a
        # surface is expected. Checking whether it really is codim 1 needs the mesh,
        # so it's checked at initialisation: via export_surface_context in
        # HydrogenTransportProblemDiscontinuous, and rejected outright in
        # HydrogenTransportProblem and its children since they don't support codim
        accepted = int | SurfaceSubdomain | VolumeSubdomain
        if not isinstance(value, accepted) or isinstance(value, bool):
            raise TypeError(
                "surface should be an int, F.SurfaceSubdomain or a codim-1 "
                "F.VolumeSubdomain"
            )

        self._surface = value

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        if not isinstance(value, Species | str):
            raise TypeError("field must be of type F.Species or str")

        self._field = value
