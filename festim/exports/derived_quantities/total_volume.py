from festim import VolumeQuantity
import fenics as f


class TotalVolume(VolumeQuantity):
    """
    Computes the total value of a field in a given volume
    int(f dx)

    Args:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id

    Attribtutes
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the hydrogen solute field

    Notes:
        units are in H/m2 in 1D, H/m in 2D and H in 3D domains for hydrogen
        concentration and K m in 1D, K m2 in 2D and K m3 in 3D domains for temperature

    """

    def __init__(self, field, volume) -> None:
        super().__init__(field=field, volume=volume)

    @property
    def title(self):
        quantity_title = f"Total {self.field} volume {self.volume}"
        if self.show_units:
            # obtain domain dimension
            dim = self.function.function_space().mesh().topology().dim()
            if self.field == "T":
                if dim == 1:
                    return quantity_title + " (K m)"
                elif dim == 2:
                    return quantity_title + " (K m2)"
                elif dim == 3:
                    return quantity_title + " (K m3)"
            else:
                if dim == 1:
                    return quantity_title + " (H m-2)"
                elif dim == 2:
                    return quantity_title + " (H m-1)"
                elif dim == 3:
                    return quantity_title + " (H)"
        else:
            return quantity_title

    def compute(self):
        return f.assemble(self.function * self.dx(self.volume))
