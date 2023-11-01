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
        self.field = field
        self.surface_subdomain = surface_subdomain
        self.filename = filename

        self.D = None

        if self.filename is None:
            self.filename = f"Surface_Flux_subdomain_{surface_subdomain.id}.csv"

        self.title = "Flux surface {}: {}".format(
            self.surface_subdomain.id, self.field.name
        )

        with open(self.filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", f"{self.title}"])

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, str):
            raise TypeError("filename must be of type str")
        if not value.endswith(".csv"):
            raise ValueError("filename must end with .csv")
        self._filename = value

    @property
    def surface_subdomain(self):
        return self._surface_subdomain

    @surface_subdomain.setter
    def surface_subdomain(self, value):
        if not isinstance(value, (int, F.SurfaceSubdomain1D)) or isinstance(
            value, bool
        ):
            raise TypeError("surface should be an int or F.SurfaceSubdomain1D")
        self._surface_subdomain = value

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        # check that field is festim.Species
        if not isinstance(value, (F.Species, str)):
            raise TypeError("field must be of type festim.Species")

        self._field = value

    def compute(self, mesh, ds):
        self.value = fem.assemble_scalar(
            fem.form(
                self.D
                * ufl.dot(ufl.grad(self.field.solution), mesh.n)
                * ds(self.surface_subdomain.id)
            )
        )

    def write(self, t):
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([t, self.value])
