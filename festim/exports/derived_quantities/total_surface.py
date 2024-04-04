from festim import SurfaceQuantity
import fenics as f


class TotalSurface(SurfaceQuantity):
    """
    Computes the total value of a field on a given surface
    int(f ds)

    Args:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id

    Attribtutes
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        export_unit (str): the unit of the derived quantity for exporting
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the hydrogen solute field

    Notes:
        units are in H/m2 in 1D, H/m in 2D and H in 3D domains for hydrogen
        concentration and K in 1D, K m in 2D and K m2 in 3D domains for temperature

    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)

    @property
    def export_unit(self):
        # obtain domain dimension
        dim = self.function.function_space().mesh().topology().dim()
        if self.field == "T":
            if dim == 1:
                return "K"
            if dim == 2:
                return "K m"
            if dim == 3:
                return "K m2"
        else:
            if dim == 1:
                return "H m-2"
            if dim == 2:
                return "H m-1"
            if dim == 3:
                return "H"

    @property
    def title(self):
        quantity_title = f"Total {self.field} surface {self.surface}"
        if self.show_units:
            return quantity_title + f" ({self.export_unit})"
        else:
            return quantity_title

    def compute(self):
        return f.assemble(self.function * self.ds(self.surface))
