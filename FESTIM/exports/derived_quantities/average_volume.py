from FESTIM import DerivedQuantity
import fenics as f


class AverageVolume(DerivedQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field)
        self.volume = volume
        self.title = "Average {} volume {}".format(self.field, self.volume)

    def compute(self):
        return f.assemble(self.function * self.dx(self.volume)
                          ) / f.assemble(1 * self.dx(self.volume))
