from dolfinx import fem
import festim as F
import csv
import numpy as np


class MinimumSurface(F.SurfaceQuantity):
    """
    Args:
        field (str): the field from which the minimum is computed (ex: "solute", "retention", "T"...)
        surface (festim.SurfaceSubdomain1D): the surface id where the minimum is computed
        filename (str, optional) : name of the file to which the minimum of the surface is exported
    """

    def __init__(self, field, surface, function, filename: str = None) -> None:
        super().__init__(field, surface, filename)
        self.function = function
        # self.title = "Minimum {} surface {}".format(self.field, self.surface)

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, funct):
        if not isinstance(funct, (dolfinx.fem.function.Function)):
            raise TypeError("function must be of type dolfinx.fem.function.Function")

        self._function = funct

    def compute(self):
        """Compute the minimum value of function on the surface"""
        return np.min(function.x.array)
