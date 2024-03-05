import festim as F
from dolfinx import fem


class TotalVolume(F.VolumeQuantity):
    def __init__(
        self,
        species: F.Species,
        volume: F.VolumeSubdomain,
        filename: str = None,
    ):
        self.volume = volume
        self.species = species
        self.filename = filename

    def compute(self, dx):
        return fem.assemble_scalar(
            fem.form(self.species.concentration * dx(self.volume))
        )
