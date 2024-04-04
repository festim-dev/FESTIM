from festim import SurfaceQuantity
import fenics as f


class TotalSurface(SurfaceQuantity):
    """
    Computes the total value of a field on a given surface

    Args:
        field (str, int): the field
        surface (int): the surface id

    Attribtutes
        field (str, int): the field
        surface (int): the surface id
        show_units (bool): show the units in the title in the derived quantities
            file
        title (str): the title of the derived quantity
        function (dolfin.function.function.Function): the solution function of
            the hydrogen solute field

    Notes:
        units are in H/m2 in 1D, H/m in 2D and H in 3D domains for hydrogen
        concentration and K in 1D, K m in 2D and K m2 in 3D domains for temperature

    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)

    @property
    def title(self):
        quantity_title = f"Total {self.field} surface {self.surface}"
        if self.show_units:
            # obtain domain dimension
            dim = self.function.function_space().mesh().topology().dim()
            if self.field == "T":
                if dim == 1:
                    return quantity_title + " (K)"
                elif dim == 2:
                    return quantity_title + " (K m)"
                elif dim == 3:
                    return quantity_title + " (K m2)"
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
        return f.assemble(self.function * self.ds(self.surface))
