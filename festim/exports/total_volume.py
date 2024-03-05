import festim as F
from dolfinx import fem


class TotalVolume(F.VolumeQuantity):
    def __init__(
        self,
        field: F.Species,
        volume: F.VolumeSubdomain,
        filename: str = None,
    ) -> None:
        super().__init__(field, volume, filename)

    def compute(self, dx):
        return fem.assemble_scalar(fem.form(self.field.solution * dx(self.volume.id)))
