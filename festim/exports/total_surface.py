from dolphinx import fem
import festim as F
import csv
import numpy as np


class TotalSurface(F.SurfaceQuantity):
    """
    Args:
        field (str): the field from which the total value is commputed (ex: "solute", "retention", "T"...)
        surface (festim.SurfaceSubdomain1D): the surface id where the total vaue is computed
        filename (str, optional): name of the file to which the total value is computed
    """

    def __init__(self, field, surface, function, filename: str = None) -> None:
        super().__init__(field, surface, filename)
        self.function = function
        # self.title = "Total {} Surface {}".format(self. field, self.surface)

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, funct):
        if not isinstance(funct, (dolfinx.fem.function.Function)):
            raise TypeError("function must be of type dolfinx.fem.function.Function")

        self._function = funct

    def compute(self, ds):
        """Compute the total value of the function on the surface

        Args:
            ds (ufl.Measure): surface measure of the model
        """

        self.value = fem.assemble_scalar(
            fem.form(self.field.solution * ds(self.surface.id))
        )
        self.data.append(self.value)
