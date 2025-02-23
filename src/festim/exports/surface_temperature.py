import ufl
from dolfinx import fem
from .surface_quantity import SurfaceQuantity

class SurfaceTemperature(SurfaceQuantity):
    """Computes the temperature on a given surface

    Args:
        temperature_field (festim.Temperature): temperature field to be computed
        surface (festim.SurfaceSubdomain1D): surface subdomain
        filename (str, optional): name of the file to which the surface temperature is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    @property
    def title(self):
        return f"Temperature surface {self.surface.id}"

    def compute(self, ds):
        """Computes the value of the temperature at the surface

        Args:
            ds (ufl.Measure): surface measure of the model
        """

        # Obtain the temperature field
        temperature_field = self.field.solution

        # Compute the average temperature on the surface
        self.value = fem.assemble_scalar(
            fem.form(
                temperature_field * ds(self.surface.id)
            )
        )
        self.data.append(self.value)

# Example usage:
# Assuming you have a temperature field and a surface defined
# surface_temp = SurfaceTemperature(temperature_field, surface)
# surface_temp.compute(ds)