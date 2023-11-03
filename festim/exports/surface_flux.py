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
        t (list): list of time values
        data (list): list of surface flux values
        title (str): title of the exported data
    """

    def __init__(
        self,
        field,
        surface: F.SurfaceSubdomain1D,
        filename: str = None,
    ) -> None:
        super().__init__(field, surface, filename)

        self.t = []
        self.data = []

    def compute(self, n, ds):
        self.value = fem.assemble_scalar(
            fem.form(
                -self.D
                * ufl.dot(ufl.grad(self.field.solution), n)
                * ds(self.surface.id)
            )
        )
        self.data.append(self.value)

    def initialise_export(self):
        self.title = "Flux surface {}: {}".format(self.surface.id, self.field.name)

        if self.filename is not None:
            with open(self.filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["t(s)", f"{self.title}"])
