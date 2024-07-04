from festim import SurfaceFlux


class ThermalFlux(SurfaceFlux):
    """
    Computes the surface flux of heat at a given surface

    Args:
        surface (int): the surface id

    Attributes:
        surface (int): the surface id
        field (str): the temperature field
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the temperature field

    .. note::
        units are in W/m2 in 1D, W/m in 2D and W in 3D domains

    """

    def __init__(self, surface) -> None:
        super().__init__(field="T", surface=surface)
