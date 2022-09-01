from festim import Export


class DerivedQuantity(Export):
    """
    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
    """

    def __init__(self, field) -> None:

        super().__init__(field=field)
        self.dx = None
        self.ds = None
        self.n = None
        self.D = None
        self.S = None
        self.thermal_cond = None
        self.H = None
        self.data = []
        self.t = []


class VolumeQuantity(DerivedQuantity):
    def __init__(self, field: str or int, volume: int) -> None:
        """DerivedQuantity relative to a volume

        Args:
            field (str, int): the field ("solute", 0, 1, "T", "retention")
            volume (int): the volume id
        """
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
    def __init__(self, field: str or int, surface: int) -> None:
        """DerivedQuantity relative to a surface

        Args:
            field (str, int):  the field ("solute", 0, 1, "T", "retention")
            surface (int): the surface id
        """
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
