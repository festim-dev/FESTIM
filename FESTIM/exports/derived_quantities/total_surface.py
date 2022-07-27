from festim import SurfaceQuantity
import fenics as f


class TotalSurface(SurfaceQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field, surface=surface)
        self.title = "Total {} surface {}".format(self.field, self.surface)

    def compute(self):
        return f.assemble(self.function * self.ds(self.surface))
