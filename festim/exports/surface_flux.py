from dolfinx import fem
import festim as F
import ufl
import csv


class SurfaceFlux(F.SurfaceQuantity):
    """Exports surface flux at a given subdomain

    Args:

    Attributes:

    Usage:
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

    def compute(self, mesh, ds):
        self.value = fem.assemble_scalar(
            fem.form(
                self.D
                * ufl.dot(ufl.grad(self.field.solution), mesh.n)
                * ds(self.surface_subdomain.id)
            )
        )
        self.data.append(self.value)

    def write(self, t):
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([t, self.value])
