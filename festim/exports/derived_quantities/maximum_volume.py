from festim import VolumeQuantity
import fenics as f
import numpy as np


class MaximumVolume(VolumeQuantity):
    """
    Computes the maximum value of a field in a given volume

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function for
            the field

    .. note::
        Units are in H/m3 for hydrogen concentration and K for temperature

    """

    def __init__(self, field, volume) -> None:
        super().__init__(field=field, volume=volume)

    @property
    def export_unit(self):
        if self.field == "T":
            return "K"
        else:
            return "H m-3"

    @property
    def title(self):
        quantity_title = f"Maximum {self.field} volume {self.volume}"
        if self.show_units:
            return quantity_title + f" ({self.export_unit})"
        else:
            return quantity_title

    def compute(self, volume_markers):
        """Minimum of f over subdomains cells marked with self.volume"""
        V = self.function.function_space()

        dm = V.dofmap()

        subd_dofs = np.unique(
            np.hstack(
                [
                    dm.cell_dofs(c.index())
                    for c in f.SubsetIterator(volume_markers, self.volume)
                ]
            )
        )

        return np.max(self.function.vector().get_local()[subd_dofs])
