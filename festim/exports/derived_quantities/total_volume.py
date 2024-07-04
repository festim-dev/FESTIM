from festim import VolumeQuantity
import fenics as f


class TotalVolume(VolumeQuantity):
    """
    Computes the total value of a field in a given volume
    int(f dx)

    Args:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id

    Attributes:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        export_unit (str): the unit of the derived quantity for exporting
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the hydrogen solute field

    .. note::
        units are in H/m2 in 1D, H/m in 2D and H in 3D domains for hydrogen
        concentration and K m in 1D, K m2 in 2D and K m3 in 3D domains for temperature

    """

    def __init__(self, field, volume) -> None:
        super().__init__(field=field, volume=volume)

    @property
    def export_unit(self):
        # obtain domain dimension
        try:
            dim = self.function.function_space().mesh().topology().dim()
        except AttributeError:
            dim = self.dx._domain._topological_dimension
            # TODO we could simply do that all the time
        # return unit depending on field and dimension of domain
        if self.field == "T":
            return f"K m{dim}".replace("1", "")
        else:
            return f"H m{dim-3}".replace(" m0", "")

    @property
    def title(self):
        quantity_title = f"Total {self.field} volume {self.volume}"
        if self.show_units:
            return quantity_title + f" ({self.export_unit})"
        else:
            return quantity_title

    def compute(self):
        return f.assemble(self.function * self.dx(self.volume))
