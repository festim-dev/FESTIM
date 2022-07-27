from festim import SurfaceQuantity
import fenics as f


class AverageSurface(SurfaceQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)
        self.title = "Average {} surface {}".format(self.field, self.surface)

    def compute(self):
        return f.assemble(self.function * self.ds(self.surface)) / f.assemble(
            1 * self.ds(self.surface)
        )
