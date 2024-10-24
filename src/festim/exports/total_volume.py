import ufl
from dolfinx import fem

from .volume_quantity import VolumeQuantity


class TotalVolume(VolumeQuantity):
    """Computes the total value of a field in a given volume

    Args:
        field (festim.Species): species for which the total volume is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the total volume is exported

    Attributes:
        see `festim.VolumeQuantity`
    """

    @property
    def title(self):
        return f"Total {self.field.name} volume {self.volume.id}"

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
