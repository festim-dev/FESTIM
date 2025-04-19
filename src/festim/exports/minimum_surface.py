from mpi4py import MPI
import dolfinx
import numpy as np

from festim.exports.surface_quantity import SurfaceQuantity


class MinimumSurface(SurfaceQuantity):
    """Computes the minimum value of a field on a given surface

    Args:
        field (festim.Species): species for which the minimum surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the minimum surface is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    @property
    def title(self):
        return f"Minimum {self.field.name} surface {self.surface.id}"

    def compute(self):
        """
        Computes the minimum value of the field on the defined surface
        subdomain, and appends it to the data list
        """
        solution = self.field.solution

        entities = self.facet_meshtags.find(self.surface.id)
        V = solution.function_space
        mesh = V.mesh
        # mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim - 1)
        dofs = dolfinx.fem.locate_dofs_topological(
            V=V, entity_dim=mesh.topology.dim - 1, entities=entities
        )

        self.value = solution.function_space.mesh.comm.allreduce(
            np.min(solution.x.array[dofs]), op=MPI.MIN
        )
        self.data.append(self.value)
