from mpi4py import MPI

from festim.exports import VolumeQuantity


class MaximumVolume(VolumeQuantity):
    """Computes the maximum value of a field in a given volume

    Args:
        field (festim.Species): species for which the maximum volume is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the maximum volume is exported

    Attributes:
        see `festim.VolumeQuantity`

    """

    @property
    def title(self):
        return f"Maximum {self.field.name} volume {self.volume.id}"

    def compute(self):
        """
        Computes the maximum value of solution function within the defined volume
        subdomain, and appends it to the data list
        """
        solution = self.field.solution
        indices = self.volume.locate_subdomain_entities(solution.function_space.mesh)
        max_value = max(self.field.solution.x.array[indices])
        MPI.COMM_WORLD.barrier()
        self.value = solution.function_space.mesh.comm.allreduce(max_value, op=MPI.MAX)
        self.data.append(self.value)
