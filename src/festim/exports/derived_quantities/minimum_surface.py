from festim import DerivedQuantity
import fenics as f
import numpy as np


class MinimumSurface(DerivedQuantity):
    """
    Args:
        field (str): the field from which the minimum
            is computed (ex: "solute", "retention", "T"...)
        surface (int): the surface id where the minimum is computed
    """

    def __init__(self, field, surface) -> None:

        super().__init__(field)
        self.surface = surface
        self.title = "Minimum {} surface {}".format(self.field, self.surface)

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
