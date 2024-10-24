import ufl
from dolfinx import fem

from .surface_quantity import SurfaceQuantity


class SurfaceFlux(SurfaceQuantity):
    """Computes the flux of a field on a given surface

    Args:
        field (festim.Species): species for which the surface flux is computed
        surface (festim.SurfaceSubdomain1D): surface subdomain
        filename (str, optional): name of the file to which the surface flux is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    @property
    def title(self):
        return f"{self.field.name} flux surface {self.surface.id}"

    def compute(self, ds):
        """Computes the value of the flux at the surface

        Args:
            ds (ufl.Measure): surface measure of the model
        """

        # obtain mesh normal from field
        # if case multispecies, solution is an index, use sub_function_space
        if isinstance(self.field.solution, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = self.field.solution.function_space.mesh
        n = ufl.FacetNormal(mesh)

        self.value = fem.assemble_scalar(
            fem.form(
                -self.D
                * ufl.dot(ufl.grad(self.field.solution), n)
                * ds(self.surface.id)
            )
        )
        self.data.append(self.value)
