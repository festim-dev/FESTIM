import festim


class SurfaceConcentrations(list):
    """
    A list of festim.SurfaceConcentration objects
    """

    def __init__(self, *args):
        # checks that input is list
        if len(args) == 0:
            super().__init__()
        else:
            if not isinstance(*args, list):
                raise TypeError("festim.SurfaceConcentrations must be a list")
            super().__init__(self._validate_surf_conc(item) for item in args[0])

        self.F = None
        self.sub_expressions = []

    def __setitem__(self, index, item):
        super().__setitem__(index, self._validate_surf_conc(item))

    def insert(self, index, item):
        super().insert(index, self._validate_surf_conc(item))

    def append(self, item):
        super().append(self._validate_surf_conc(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(self._validate_surf_conc(item) for item in other)

    def _validate_surf_conc(self, value):
        if isinstance(value, festim.SurfaceConcentration):
            return value
        raise TypeError(
            "festim.SurfaceConcentrations must be a list of festim.SurfaceConcentration"
        )

    def create_forms(self, mobile, T, ds, dt):
        self.F = 0
        for surf_conc in self:
            surf_conc.create_form(mobile, T, ds, dt)
            self.F += surf_conc.F
            self.sub_expressions += surf_conc.sub_expressions
