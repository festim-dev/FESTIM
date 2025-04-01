from mpi4py import MPI

import numpy as np

import festim.exports.surface_quantity as sq


class MinimumSurface(sq.SurfaceQuantity):
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
        indices = self.surface.locate_boundary_facet_indices(
            solution.function_space.mesh
        )

        self.value = solution.function_space.mesh.comm.allreduce(
            np.min(self.field.solution.x.array[indices]), op=MPI.MIN
        )
        self.data.append(self.value)
