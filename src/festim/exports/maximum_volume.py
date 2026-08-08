from mpi4py import MPI

import dolfinx
import numpy as np

from festim.exports.volume_quantity import VolumeQuantity


class MaximumVolume(VolumeQuantity):
    """Computes the maximum value of a field in a given volume.

    Args:
        field (festim.Species): species for which the maximum volume is computed
        volume (festim.VolumeSubdomain): volume subdomain
        filename (str, optional): name of the file to which the maximum volume
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
        return f"Maximum {self.field.name} volume {self.volume.id}"

    def compute(
        self,
        u: dolfinx.fem.Function | None = None,
        volume_meshtags: dolfinx.mesh.MeshTags | None = None,
    ):
        """Computes the maximum value of solution function within the defined volume
        subdomain, and appends it to the data list.

        Args:
            u: the field the maximum is computed from. Defaults to
                ``self.field.post_processing_solution``
            volume_meshtags: the cell meshtags used to locate the cells of the volume
                subdomain. Defaults to ``self.volume_meshtags``. If neither is
                available, the maximum is computed over the whole mesh ``u`` is
                defined on. This is the expected behaviour for the discontinuous
                problem, where each volume subdomain owns its own submesh.
        """
        solution = self.field.post_processing_solution if u is None else u
        meshtags = self.volume_meshtags if volume_meshtags is None else volume_meshtags

        if isinstance(solution, dolfinx.fem.Function):
            V = solution.function_space
        else:
            V = self.field.sub_function_space
        mesh = V.mesh

        if meshtags is None:
            # the mesh of the field is the volume subdomain itself (submesh case)
            values = solution.x.array
        else:
            entities = meshtags.find(self.volume.id)
            mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
            dofs = dolfinx.fem.locate_dofs_topological(
                V=V, entity_dim=mesh.topology.dim, entities=entities
            )
            values = solution.x.array[dofs]

        # a process may hold no dof of the volume at all, np.max would then raise
        local_max = np.max(values) if values.size > 0 else -np.inf

        self.value = mesh.comm.allreduce(local_max, op=MPI.MAX)
        self.data.append(self.value)
