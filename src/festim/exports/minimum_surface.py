from mpi4py import MPI

import dolfinx
import numpy as np

from festim.exports.surface_quantity import SurfaceQuantity
from festim.subdomain.volume_subdomain import VolumeSubdomain


class MinimumSurface(SurfaceQuantity):
    """Computes the minimum value of a field on a given surface.

    Args:
        field (festim.Species): species for which the minimum surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the minimum surface
        is exported

    Attributes:
        see `festim.SurfaceQuantity`
        facet_meshtags: the facet meshtags of the parent mesh
        volume: the volume subdomain the surface bounds. Set by the problem;
            ``None`` outside `festim.HydrogenTransportProblemDiscontinuous`
    """

    facet_meshtags: dolfinx.mesh.MeshTags | None = None
    volume: VolumeSubdomain | None = None

    @property
    def title(self):
        return f"Minimum {self.field.name} surface {self.surface.id}"

    @property
    def is_submesh(self) -> bool:
        """Whether the field's solution lives on the submesh of ``volume``, as in
        ``festim.HydrogenTransportProblemDiscontinuous``. See issue #1191."""
        return (
            self.volume is not None
            and self.volume in self.field.subdomain_to_post_processing_solution
        )

    @property
    def solution(self):
        if self.is_submesh:
            return self.field.subdomain_to_post_processing_solution[self.volume]
        return self.field.post_processing_solution

    @property
    def meshtags(self):
        """Facet meshtags of whichever mesh ``solution`` lives on."""
        return self.volume.ft if self.is_submesh else self.facet_meshtags

    def compute(self):
        """Computes the minimum value of the field on the defined surface subdomain, and
        appends it to the data list.
        """
        assert self.meshtags is not None, (
            "facet meshtags must be set before computing the min surface value"
        )
        solution = self.solution
        V = (
            solution.function_space
            if isinstance(solution, dolfinx.fem.Function)
            else self.field.sub_function_space
        )
        mesh = V.mesh
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(
            V=V, entity_dim=fdim, entities=self.meshtags.find(self.surface.id)
        )
        values = solution.x.array[dofs]

        local_min = np.min(values) if values.size > 0 else np.inf
        self.value = mesh.comm.allreduce(local_min, op=MPI.MIN)
        self.data.append(self.value)
