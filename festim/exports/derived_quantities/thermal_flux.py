from festim import SurfaceFlux


class ThermalFlux(SurfaceFlux):
    """
    Computes the surface flux of heat at a given surface

    Args:
        surface (int): the surface id

    Attribtutes
        surface (int): the surface id
        field (str): the temperature field
        show_units (bool): show the units in the title in the derived quantities
            file
        title (str): the title of the derived quantity
        function (dolfin.function.function.Function): the solution function of
            the temperature field

    Notes:
        units are in W/m2 in 1D, W/m in 2D and W in 3D domains

    """

    def __init__(self, surface) -> None:
        super().__init__(field="T", surface=surface)

    @property
    def title(self):
        quantity_title = f"Flux surface {self.surface}: {self.field}"
        if self.show_units:
            # obtain domain dimension
            dim = self.function.function_space().mesh().topology().dim()
            if dim == 1:
                return quantity_title + " (W m-2)"
            if dim == 2:
                return quantity_title + " (W m-1)"
            if dim == 3:
                return quantity_title + " (W)"
        else:
            return quantity_title
