from dolphinx import fem
import festim as F
import csv
import numpy as np


class MaximumSurface(F.SurfaceQuantity):
    """
    Args:
        field (str): the field from which the maximum is computed (ex: "solute", "retention", "T"...)
        surface (festim.SurfaceSubdomain1D): the surface id where the maximum is computed
        filename (str, optional) : name of the file to which the maximum of the surface is exported
    """

    def __init__(self, field, surface, function, filename: str = None) -> None:
        super().__init__(field, surface, filename)
        self.function = function
        # self.title = "Maximum {} surface {}".format(self.field, self.surface)

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, funct):
        if not isinstance(funct, (dolfinx.fem.function.Function)):
            raise TypeError("function must be of type festim.Function")

        self._function = funct

    def compute(self):
        """Compute the maximum value of function on the surface"""
        return np.max(function.x.array)
