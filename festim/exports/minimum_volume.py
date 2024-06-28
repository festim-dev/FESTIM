import festim as F
import numpy as np


class MinimumVolume(F.VolumeQuantity):
    """Computes the minmum value of a field in a given volume

    Args:
        field (festim.Species): species for which the minmum volume is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the minmum volume is exported

    Attributes:
        see `festim.VolumeQuantity`
    """

    @property
    def title(self):
        return f"Minimum {self.field.name} volume {self.volume.id}"

    def compute(self):
        """
        Computes the minimum value of solution function within the defined volume
        subdomain, and appends it to the data list
        """
        self.value = np.min(self.field.solution.x.array[self.volume.entities])
        self.data.append(self.value)
