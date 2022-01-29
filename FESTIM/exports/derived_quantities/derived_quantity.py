from FESTIM import Export


class DerivedQuantity(Export):
    def __init__(self, field) -> None:
        super().__init__(field=field)
        self.dx = None
        self.ds = None
        self.n = None
        self.D = None
        self.S = None
        self.thermal_cond = None
        self.H = None
