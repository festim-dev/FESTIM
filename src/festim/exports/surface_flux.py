import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_quantity import SurfaceQuantity
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class SurfaceFlux(SurfaceQuantity):
    """Computes the flux of a field on a given surface

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain
        filename: name of the file to which the surface flux is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    field: Species
    surface: SurfaceSubdomain
    filename: str

    title: str
    value: float
    data: list[float]

    @property
    def title(self):
        return f"{self.field.name} flux surface {self.surface.id}"

    def __init__(
        self, field: Species, surface: SurfaceSubdomain, filename: str | None = None
    ) -> None:
        super().__init__(field=field, surface=surface, filename=filename)

    def compute(
        self, u: fem.Function | ufl.indexed.Indexed, ds: ufl.Measure, entity_maps=None
    ):
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
