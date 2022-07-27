from festim import VolumeQuantity
import fenics as f


class TotalVolume(VolumeQuantity):
    def __init__(self, field, volume) -> None:
        super().__init__(field, volume=volume)
        self.title = "Total {} volume {}".format(self.field, self.volume)

    def compute(self):
        return f.assemble(self.function * self.dx(self.volume))
