import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_quantity import SurfaceQuantity


class AverageSurface(SurfaceQuantity):
    """Computes the average value of a field on a given surface

    Args:
        field (festim.Species): species for which the average surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the average surface is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    @property
    def title(self):
        return f"Average {self.field.name} surface {self.surface.id}"

    def compute(
        self, u: fem.Function | ufl.indexed.Indexed, ds: ufl.Measure, entity_maps=None
    ):
        """
        Computes the average value of the field on the defined surface
        subdomain, and appends it to the data list

        Args:
            u: field for which the average value is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """

        self.value = assemble_scalar(u * ds(self.surface.id)) / assemble_scalar(
            1 * ds(self.surface.id)
        )
        self.data.append(self.value)
