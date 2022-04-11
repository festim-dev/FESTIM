from FESTIM import DerivedQuantity
import fenics as f


class TotalSurface(DerivedQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field)
        self.surface = surface
        self.title = "Total {} surface {}".format(self.field, self.surface)

    def compute(self):
        return f.assemble(self.function * self.ds(self.surface))
