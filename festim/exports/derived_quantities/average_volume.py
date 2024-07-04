from festim import VolumeQuantity
import fenics as f


class AverageVolume(VolumeQuantity):
    """
    Computes the average value of a field in a given volume
    int(f dx) / int (1 * dx)

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field

    .. note::
        Units are in H/m3 for hydrogen concentration and K for temperature
    """

    def __init__(self, field, volume: int) -> None:
        super().__init__(field=field, volume=volume)

    @property
    def title(self):
        quantity_title = f"Average {self.field} volume {self.volume}"
        if self.show_units:
            if self.field == "T":
                return quantity_title + " (K)"
            else:
                return quantity_title + " (H m-3)"
        else:
            return quantity_title

    def compute(self):
        return f.assemble(self.function * self.dx(self.volume)) / f.assemble(
            1 * self.dx(self.volume)
        )
