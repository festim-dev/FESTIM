from festim import SurfaceFlux


class HydrogenFlux(SurfaceFlux):
    """
    Computes the surface flux of hydrogen at a given surface

    Args:
        surface (int): the surface id

    Attribtutes
        field (str, int): the hydrogen solute field
        surface (int): the surface id
        show_units (bool): show the units in the title in the derived quantities
            file
        title (str): the title of the derived quantity
        function (dolfin.function.function.Function): the solution function of
            the hydrogen solute field

    Notes:
        units are in H/m2/s in 1D, H/m/s in 2D and H/s in 3D domains

    """

    def __init__(self, surface) -> None:
        super().__init__(field="solute", surface=surface)

    @property
    def title(self):
        quantity_title = f"Flux surface {self.surface}: {self.field}"
        if self.show_units:
            # obtain domain dimension
            dim = self.function.function_space().mesh().topology().dim()
            if dim == 1:
                return quantity_title + " (H m-2 s-1)"
            if dim == 2:
                return quantity_title + " (H m-1 s-1)"
            if dim == 3:
                return quantity_title + " (H s-1)"
        else:
            return quantity_title
