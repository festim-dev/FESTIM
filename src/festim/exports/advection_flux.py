import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_flux import SurfaceFlux
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class Advection_flux(SurfaceFlux):
    """Computes the advective flux of a field on a given surface

    Args:
        field: species for which the advective flux is computed
        surface: surface subdomain
        filename: name of the file to which the advective flux is
            exported
        velocity_field: velocity field for which the advective flux is computed

    Attributes:
        see `festim.SurfaceFlux`
    """

    field: Species
    surface: SurfaceSubdomain
    velocity_field: fem.Function

    def __init__(self, field, surface, filename, velocity_field):
        super().__init__(field=field, surface=surface, filename=filename)

        self.velocity_field = velocity_field

    @property
    def title(self):
        return f"{self.field.name} advective flux surface {self.surface.id}"

    def compute(self, u, ds: ufl.Measure, entity_maps=None):
        """Computes the value of the flux at the surface

        Args:
            ds (ufl.Measure): surface measure of the model
        """

        mesh = ds.ufl_domain
        n = ufl.FacetNormal(mesh)

        surface_flux = assemble_scalar(
            fem.form(
                -self.D * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        advective_flux = assemble_scalar(
            fem.form(
                u * ufl.inner(self.velocity_field, n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )

        self.value = surface_flux + advective_flux
        self.data.append(self.value)
