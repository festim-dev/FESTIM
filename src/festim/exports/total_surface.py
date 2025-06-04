import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_quantity import SurfaceQuantity


class TotalSurface(SurfaceQuantity):
    """Computes the total value of a field on a given surface

    Args:
        field (`festim.Species`): species for which the total volume is computed
        surface (`festim.SurfaceSubdomain`): surface subdomain
        filename (str, optional): name of the file to which the total volume is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    @property
    def title(self):
        return f"Total {self.field.name} surface {self.surface.id}"

    def compute(self, ds: ufl.Measure):
        """
        Computes the total value of the field on the defined surface
        subdomain, and appends it to the data list

        Args:
            ds (ufl.Measure): surface measure of the model
        """
        self.value = assemble_scalar(
            fem.form(self.field.solution * ds(self.surface.id))
        )
        self.data.append(self.value)
