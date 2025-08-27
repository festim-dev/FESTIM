import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.volume_quantity import VolumeQuantity


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

    def compute(self, u, dx: ufl.Measure, entity_maps=None):
        """
        Computes the value of the total volume of the field in the volume subdomain
        and appends it to the data list

        Args:
            u: field for which the total volume is computed
            dx: volume measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """
        self.value = assemble_scalar(
            fem.form(u * dx(self.volume.id), entity_maps=entity_maps)
        )
        self.data.append(self.value)
