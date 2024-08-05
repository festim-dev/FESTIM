from festim import SurfaceQuantity
import fenics as f
import numpy as np


class MinimumSurface(SurfaceQuantity):
    """
    Computes the minimum value of a field on a given surface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field

    .. note::
        Units are in H/m3 for hydrogen concentration and K for temperature

    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)

    @property
    def title(self):
        quantity_title = f"Minimum {self.field} surface {self.surface}"
        if self.show_units:
            if self.field == "T":
                return quantity_title + " (K)"
            else:
                return quantity_title + " (H m-3)"
        else:
            return quantity_title

    def compute(self, surface_markers):
        """Minimum of f over subdomains facets marked with self.surface"""
        V = self.function.function_space()

        dm = V.dofmap()

        subd_dofs = np.unique(
            np.hstack(
                [
                    dm.cell_dofs(c.index())
                    for c in f.SubsetIterator(surface_markers, self.surface)
                ]
            )
        )

        return np.min(self.function.vector().get_local()[subd_dofs])
