import csv
from dolfinx import fem
import ufl
from .surface_quantity import SurfaceQuantity


class AverageSurfaceTemperature(SurfaceQuantity):
    """Exports the average temperature on a given surface.

    Args:
        surface: the surface subdomain
        filename: name of the file to which the average surface temperature is exported

    Attributes:
        temperature_field: the temperature field
        surface (int or festim.SurfaceSubdomain): the surface subdomain
        filename (str): name of the file to which the surface temperature is exported
        t (list): list of time values
        data (list): list of average temperature values on the surface
    """
    
    surface: int | SurfaceSubdomain
    filename: str | None
    
    temperature_field: fem.Constant | fem.Function


    def __init__(self, surface, filename: str = None) -> None:
        self.surface = surface
        self.filename = filename

        self.temperature_field = None
        self.t = []
        self.data = []
        self._first_time_export = True

    @property
    def title(self):
        return f"Temperature surface {self.surface.id}"

    def compute(self, ds):
        """Computes the average temperature on the surface.

        Args:
            ds (ufl.Measure): surface measure of the model
        """
        temperature_field = self.temperature_field

        surface_integral = fem.assemble_scalar(
            fem.form(temperature_field * ds(self.surface.id))
        )  # integral over surface

        surface_area = fem.assemble_scalar(fem.form(1 * ds(self.surface.id)))

        self.value = surface_integral / surface_area  # avg temp

        self.data.append(self.value)
