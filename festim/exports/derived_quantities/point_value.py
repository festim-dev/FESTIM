from festim import DerivedQuantity
from mpi4py import MPI as pyMPI
import numpy as np


class PointValue(DerivedQuantity):
    """
    Computes the value of a field at a given point

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        x (int, float, tuple, list): the point coordinates

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        x (int, float, tuple, list): the point coordinates
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field

    .. note::
        Units are in H/m3 for hydrogen concentration and K for temperature

    """

    def __init__(self, field: str | int, x: int | float | tuple | list) -> None:
        super().__init__(field=field)
        # make sure x is an iterable
        if not hasattr(x, "__iter__"):
            x = [x]
        self.x = x

    @property
    def export_unit(self):
        if self.field == "T":
            return "K"
        else:
            return "H m-3"

    @property
    def title(self):
        quantity_title = f"{self.field} value at {self.x}"
        if self.show_units:
            return quantity_title + f" ({self.export_unit})"
        else:
            return quantity_title

    def compute(self):
        """Evaluates the value of a function at the given point"""

        # Handles the RuntimeError due to mesh partitioning
        try:
            value_local = self.function(self.x)
        except RuntimeError:
            value_local = np.inf * np.ones(self.function.value_shape())

        mesh = self.function.function_space().mesh()
        comm = mesh.mpi_comm()

        value_global = np.zeros_like(value_local)
        comm.Allreduce(value_local, value_global, op=pyMPI.MIN)

        return value_global
