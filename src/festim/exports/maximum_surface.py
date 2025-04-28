from mpi4py import MPI
import dolfinx
import numpy as np

from festim.exports.surface_quantity import SurfaceQuantity


class MaximumSurface(SurfaceQuantity):
    """Computes the maximum value of a field on a given surface

    Args:
        field (festim.Species): species for which the maximum surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the maximum surface is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    @property
    def title(self):
        return f"Maximum {self.field.name} surface {self.surface.id}"

    def compute(self):
        """
        Computes the maximum value of the field on the defined surface
        subdomain, and appends it to the data list
        """
        solution = self.field.post_processing_solution
        entities = self.facet_meshtags.find(self.surface.id)
        if isinstance(solution, dolfinx.fem.Function):
            V = solution.function_space
        else:
            V = self.field.sub_function_space
        mesh = V.mesh

        dofs = dolfinx.fem.locate_dofs_topological(
            V=V, entity_dim=mesh.topology.dim - 1, entities=entities
        )

        self.value = mesh.comm.allreduce(np.max(solution.x.array[dofs]), op=MPI.MAX)
        self.data.append(self.value)
