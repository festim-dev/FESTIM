import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_quantity import SurfaceQuantity


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

    def compute(self, u, ds: ufl.Measure, entity_maps=None):
        """Computes the value of the flux at the surface

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """

        # obtain mesh normal from field
        # if case multispecies, solution is an index, use sub_function_space
        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh
        n = ufl.FacetNormal(mesh)

        self.value = assemble_scalar(
            fem.form(
                -self.D * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        self.data.append(self.value)
