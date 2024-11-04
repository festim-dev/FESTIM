import numpy as np

from festim.exports import VolumeQuantity


class MinimumVolume(VolumeQuantity):
    """Computes the minimum value of a field in a given volume

    Args:
        field (festim.Species): species for which the minimum volume is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the minimum volume is exported

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
        solution = self.field.solution
        indices = self.volume.locate_subdomain_entities(solution.function_space.mesh)
        # FIXME: np.min/np.max is not parallel safe (unique value per process)
        # Needs to use a reduction operation (MPI.comm.allreduce(..., op=...))
        self.value = np.min(self.field.solution.x.array[indices])
        self.data.append(self.value)
