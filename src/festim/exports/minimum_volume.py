from mpi4py import MPI
import numpy as np
import dolfinx

from festim.exports.volume_quantity import VolumeQuantity


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
        solution = self.field.post_processing_solution
        entities = self.volume_meshtags.find(self.volume.id)

        if isinstance(solution, dolfinx.fem.Function):
            V = solution.function_space
        else:
            V = self.field.sub_function_space
        mesh = V.mesh
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(
            V=V, entity_dim=mesh.topology.dim, entities=entities
        )

        self.value = mesh.comm.allreduce(np.min(solution.x.array[dofs]), op=MPI.MIN)
        self.data.append(self.value)
