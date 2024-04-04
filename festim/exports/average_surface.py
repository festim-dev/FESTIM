from dolfinx import fem
import festim as F
import csv
import numpy as np


class AverageSurface(F.SurfaceQuantity):
    """
    Args:
        field (str): the field from which the average value is computed
        surface (festim.SurfaceSubdomain1D): the surface id where the average value is computed
        filename (str, optional): name of the file to which the total value is computed
    """

    def __init__(self, field, surface, filename: str = None) -> None:
        super().__init__(field, surface, filename)
        self.function = function

    @property
    def function(self):
        return self.function

    @function.setter
    def function(self, funct):
        if not isinstance(funct, (dolfinx.fem.function.Function)):
            raise TypeError("function must be of type dolfinx.fem.function.Function")

        self._function = funct

    def compute(self):
        """Compute the total value of the function on the surface"""
        # placeholder for the coming function
