import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_quantity import SurfaceQuantity
from festim.helpers import restrict
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain
from festim.subdomain.volume_subdomain import VolumeSubdomain


class SurfaceFlux(SurfaceQuantity):
    """Computes the flux of a field on a given surface.

    Only the diffusive flux is computed: an advection term on the subdomain is not
    accounted for, and FESTIM warns when one is present.

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain. See `festim.SurfaceQuantity` for the codimensional
            cases -- a manifold, or the boundary of one.
        filename: name of the file to which the surface flux is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    field: Species
    surface: SurfaceSubdomain | VolumeSubdomain
    filename: str

    title: str
    value: float
    data: list[float]

    @property
    def title(self):
        return f"{self.field.name} flux surface {self.surface.id}"

    def __init__(
        self,
        field: Species,
        surface: SurfaceSubdomain | VolumeSubdomain,
        filename: str | None = None,
    ) -> None:
        super().__init__(field=field, surface=surface, filename=filename)

    def compute(
        self,
        u: fem.Function | ufl.indexed.Indexed,
        ds: ufl.Measure,
        entity_maps=None,
        restriction: str | None = None,
    ):
        """Computes the value of the flux at the surface.

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
            restriction: which side of an interior facet to evaluate the flux on, when
                ``ds`` is an interior facet measure. The whole integrand is restricted,
                so the normal is the one pointing out of that side and the sign
                convention matches an exterior surface.
        """

        # obtain mesh normal from integration domain
        mesh = ds.ufl_domain()
        n = ufl.FacetNormal(mesh)

        integrand = -self.D * ufl.dot(ufl.grad(u), n)

        self.value = assemble_scalar(
            fem.form(
                restrict(integrand, restriction) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        self.data.append(self.value)
