from dolfinx import fem
import festim as F
import ufl
import csv


class SurfaceFlux(F.SurfaceQuantity):
    """Exports surface flux at a given subdomain

    Args:
        field (festim.Species): species for which the surface flux is computed
        surface_subdomain (festim.SurfaceSubdomain1D): surface subdomain
        filename (str, optional): name of the file to which the surface flux is exported

    Attributes:
        field (festim.Species): species for which the surface flux is computed
        surface_subdomain (festim.SurfaceSubdomain1D): surface subdomain
        filename (str): name of the file to which the surface flux is exported
        t (list): list of time values
        data (list): list of surface flux values
        title (str): title of the exported data
        write_to_file (bool): True if the data is exported to a file
    """

    def __init__(
        self,
        field,
        surface_subdomain: int or F.SurfaceSubdomain1D,
        filename: str = None,
    ) -> None:
        super().__init__(field, surface_subdomain, filename)

        self.t = []
        self.data = []

        self.title = "Flux surface {}: {}".format(
            self.surface_subdomain.id, self.field.name
        )

        if self.write_to_file:
            with open(self.filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Time", f"{self.title}"])

    def compute(self, n, ds):
        self.value = fem.assemble_scalar(
            fem.form(
                -self.D
                * ufl.dot(ufl.grad(self.field.solution), n)
                * ds(self.surface_subdomain.id)
            )
        )
        self.data.append(self.value)
