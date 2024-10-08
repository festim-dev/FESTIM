from festim import Export


class DerivedQuantity(Export):
    """
    Parent class of all derived quantities

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (fenics.Function): the solution function of
            the field
        dx (fenics.Measure): the measure of the volume
        ds (fenics.Measure): the measure of the surface
        n (fenics.Function): the normal vector
        D (fenics.Function): the diffusion coefficient
        S (fenics.Function): the source term
        thermal_cond (fenics.Function): the thermal conductivity
        Q (fenics.Function): the heat source term
        data (list): the data of the derived quantity
        t (list): the time values of the data
        allowed_meshes (list): the allowed meshes for the derived quantity
    """

    def __init__(self, field) -> None:
        super().__init__(field=field)
        self.dx = None
        self.ds = None
        self.n = None
        self.D = None
        self.S = None
        self.thermal_cond = None
        self.Q = None
        self.T = None
        self.data = []
        self.t = []
        self.show_units = True

    @property
    def allowed_meshes(self):
        # by default, all meshes are allowed
        # override this method if that's not the case
        return ["cartesian", "cylindrical", "spherical"]


class VolumeQuantity(DerivedQuantity):
    """DerivedQuantity relative to a volume

    Args:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id

    """

    def __init__(self, field: str or int, volume: int) -> None:
        super().__init__(field)
        self.volume = volume

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("volume should be an int")

        self._volume = value


class SurfaceQuantity(DerivedQuantity):
    """DerivedQuantity relative to a surface

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id

    """

    def __init__(self, field: str or int, surface: int) -> None:

        super().__init__(field)
        self.surface = surface

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, value):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("surface should be an int")
        self._surface = value
