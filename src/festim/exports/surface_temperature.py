import csv
from dolfinx import fem
import ufl
import festim as F

class SurfaceTemperature(F.SurfaceQuantity):
    """Exports the average temperature on a given surface.

    Args:
        temperature_field (fem.Constant or fem.Function): the temperature field to be computed
        surface (int or festim.SurfaceSubdomain): the surface subdomain
        filename (str, optional): name of the file to which the average surface temperature is exported

    Attributes:
        temperature_field (fem.Constant or fem.Function): the temperature field
        surface (int or festim.SurfaceSubdomain): the surface subdomain
        filename (str): name of the file to which the surface temperature is exported
        t (list): list of time values
        data (list): list of average temperature values on the surface
    """

    def __init__(self, temperature_field, surface, filename: str = None) -> None:
        self.temperature_field = temperature_field
        self.surface = surface
        self.filename = filename

        self.t = []
        self.data = []
        self._first_time_export = True

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if value is None:
            self._filename = None
        elif not isinstance(value, str):
            raise TypeError("filename must be of type str")
        elif not value.endswith(".csv") and not value.endswith(".txt"):
            raise ValueError("filename must end with .csv or .txt")
        self._filename = value

    @property
    def temperature_field(self):
        return self._temperature_field
    
    @temperature_field.setter
    def temperature_field(self, value):
        # check that temperature field is float, int, fem.Constant, fem.Function, or fem.Expression
        if not isinstance(value, (fem.Constant, fem.Function, fem.Expression, int, float)):
            raise TypeError("field must be of type float, int, fem.Constant, fem.Function, or fem.Expression")
                        
        self._temperature_field = value

    @property
    def title(self):
        return f"Temperature surface {self.surface.id}"

    def compute(self, ds):
        """Computes the average temperature on the surface.

        Args:
            ds (ufl.Measure): surface measure of the model
        """
        temperature_field = self.temperature_field  

        surface_integral = fem.assemble_scalar(fem.form(temperature_field * ds(self.surface.id))) # integral over surface
        
        surface_area = fem.assemble_scalar(fem.form(1 * ds(self.surface.id)))

        self.value = surface_integral / surface_area # avg temp

        self.data.append(self.value)

    def write(self, t):
        """Writes the time and temperature value to the file.

        Args:
            t (float): current time value
        """
        if self.filename is not None:
            if self._first_time_export:
                header = ["t(s)", f"{self.title}"]
                with open(self.filename, mode="w+", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                self._first_time_export = False
            with open(self.filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([t, self.value])