from dolfinx import fem
import festim as F
import ufl
import csv


class SurfaceFlux(F.SurfaceQuantity):
    """Exports surface flux at a given subdomain

    Args:
        field (festim.Species): species for which the surface flux is computed
        surface (festim.SurfaceSubdomain1D): surface subdomain
        filename (str, optional): name of the file to which the surface flux is exported

    Attributes:
        field (festim.Species): species for which the surface flux is computed
        surface (festim.SurfaceSubdomain1D): surface subdomain
        filename (str): name of the file to which the surface flux is exported
    """

    def __init__(
        self,
        field,
        surface: F.SurfaceSubdomain1D,
        filename: str = None,
    ) -> None:
        super().__init__(field, surface, filename)

    def compute(self, n, ds):
        """Computes the value of the surface flux at the surface

        Args:
            n (ufl.geometry.FacetNormal): normal vector to the surface
            ds (ufl.Measure): surface measure of the model
        """
        self.value = fem.assemble_scalar(
            fem.form(
                -self.D
                * ufl.dot(ufl.grad(self.field.solution), n)
                * ds(self.surface.id)
            )
        )
        self.data.append(self.value)

    def initialise_export(self):
        title = "Flux surface {}: {}".format(self.surface.id, self.field.name)

        if self.filename is not None:
            with open(self.filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["t(s)", f"{title}"])
