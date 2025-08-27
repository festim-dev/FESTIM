from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.volume_quantity import VolumeQuantity


class AverageVolume(VolumeQuantity):
    """Computes the average value of a field in a given volume

    Args:
        field (festim.Species): species for which the average volume is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the average volume is exported

    Attributes:
        see `festim.VolumeQuantity`
    """

    @property
    def title(self):
        return f"Average {self.field.name} volume {self.volume.id}"

    def compute(self, u, dx, entity_maps=None):
        """
        Computes the average value of solution function within the defined volume
        subdomain, and appends it to the data list
        """
        self.value = assemble_scalar(
            fem.form(u * dx(self.volume.id), entity_maps=entity_maps)
        ) / assemble_scalar(fem.form(1 * dx(self.volume.id)))
        self.data.append(self.value)
