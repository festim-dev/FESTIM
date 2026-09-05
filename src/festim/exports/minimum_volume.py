from mpi4py import MPI

import dolfinx
import numpy as np

from festim.exports.volume_quantity import VolumeQuantity


class MinimumVolume(VolumeQuantity):
    """Computes the minimum value of a field in a given volume.

    Args:
        field (festim.Species): species for which the minimum volume is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the minimum volume
        is exported

    Attributes:
        see `festim.VolumeQuantity`
        volume_meshtags: the cell meshtags of the mesh the field is defined on. Is
            ``None`` when the field is defined on a submesh that already coincides
            with the volume subdomain (``festim.HydrogenTransportProblemDiscontinuous``)
    """

    volume_meshtags: dolfinx.mesh.MeshTags | None = None

    @property
    def title(self):
        return f"Minimum {self.field.name} volume {self.volume.id}"

    @property
    def is_submesh(self) -> bool:
        """Whether the field's solution lives on a submesh of ``volume`` itself, as in
        ``festim.HydrogenTransportProblemDiscontinuous``. See issue #1191."""
        return self.volume in self.field.subdomain_to_post_processing_solution

    @property
    def solution(self):
        if self.is_submesh:
            return self.field.subdomain_to_post_processing_solution[self.volume]
        return self.field.post_processing_solution

    def compute(self):
        """
        Computes the minimum value of solution function within the defined volume
        subdomain, and appends it to the data list.
        """
        solution = self.solution
        V = (
            solution.function_space
            if isinstance(solution, dolfinx.fem.Function)
            else self.field.sub_function_space
        )
        mesh = V.mesh

        if self.is_submesh:
            # the mesh of the field is the volume subdomain itself (submesh case)
            values = solution.x.array
        else:
            assert self.volume_meshtags is not None, (
                "volume meshtags must be set before computing the min volume value"
            )
            entities = self.volume_meshtags.find(self.volume.id)
            mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
            dofs = dolfinx.fem.locate_dofs_topological(
                V=V, entity_dim=mesh.topology.dim, entities=entities
            )
            values = solution.x.array[dofs]

        # a process may hold no dof of the volume at all, np.min would then raise
        local_min = np.min(values) if values.size > 0 else np.inf

        self.value = mesh.comm.allreduce(local_min, op=MPI.MIN)
        self.data.append(self.value)
