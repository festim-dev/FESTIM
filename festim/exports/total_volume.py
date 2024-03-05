import festim as F
from dolfinx import fem
import ufl


class TotalVolume(F.VolumeQuantity):
    """Export TotalVolume

    Args:
        field (`festim.Species`): species for which the total volume is computed
        volume (`festim.VolumeSubdomain`): volume subdomain
        filename (str, optional): name of the file to which the total volume is exported

    Attributes:
        see `festim.VolumeQuantity`
    """

    def __init__(
        self,
        field: F.Species,
        volume: F.VolumeSubdomain,
        filename: str = None,
    ) -> None:
        super().__init__(field, volume, filename)

    def compute(self, dx: ufl.Measure):
        """
        Computes the value of the total volume of the field in the volume subdomain
        and appends it to the data list

        Args:
            dx (ufl.Measure): volume measure of the model
        """
        self.value = fem.assemble_scalar(
            fem.form(self.field.solution * dx(self.volume.id))
        )
        self.data.append(self.value)
