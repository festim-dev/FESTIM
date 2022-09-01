from festim import DerivedQuantity
import fenics as f
import numpy as np


class MaximumSurface(DerivedQuantity):
    """
    Args:
        field (str): the field from which the maximum
            is computed (ex: "solute", "retention", "T"...)
        surface (int): the surface id where the maximum is computed
    """

    def __init__(self, field, surface) -> None:

        super().__init__(field)
        self.surface = surface
        self.title = "Maximum {} surface {}".format(self.field, self.surface)

    def compute(self, surface_markers):
        """Maximum of f over subdomains facets marked with self.surface"""
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

        return np.max(self.function.vector().get_local()[subd_dofs])
