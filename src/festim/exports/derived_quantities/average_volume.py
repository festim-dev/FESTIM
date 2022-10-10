from festim import VolumeQuantity
import fenics as f


class AverageVolume(VolumeQuantity):
    def __init__(self, field, volume: int) -> None:
        super().__init__(field, volume)
        self.title = "Average {} volume {}".format(self.field, self.volume)

    def compute(self):
        return f.assemble(self.function * self.dx(self.volume)) / f.assemble(
            1 * self.dx(self.volume)
        )
