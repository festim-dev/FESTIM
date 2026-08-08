from mpi4py import MPI

import dolfinx
import numpy as np

from festim.exports.surface_quantity import SurfaceQuantity


class MinimumSurface(SurfaceQuantity):
    """Computes the minimum value of a field on a given surface.

    Args:
        field (festim.Species): species for which the minimum surface is computed
        surface (festim.SurfaceSubdomain): surface subdomain
        filename (str, optional): name of the file to which the minimum surface
        is exported

    Attributes:
        see `festim.SurfaceQuantity`
        facet_meshtags: the facet meshtags of the mesh the field is defined on
    """

    facet_meshtags: dolfinx.mesh.MeshTags | None = None

    @property
    def title(self):
        return f"Minimum {self.field.name} surface {self.surface.id}"

    def compute(
        self,
        u: dolfinx.fem.Function | None = None,
        facet_meshtags: dolfinx.mesh.MeshTags | None = None,
    ):
        """Computes the minimum value of the field on the defined surface subdomain, and
        appends it to the data list.

        Args:
            u: the field the minimum is computed from. Defaults to
                ``self.field.post_processing_solution``
            facet_meshtags: the facet meshtags used to locate the facets of the
                surface subdomain. Defaults to ``self.facet_meshtags``. For the
                discontinuous problem these are the facet meshtags of the submesh
                the field lives on (``VolumeSubdomain.ft``)

        Raises:
            ValueError: if no facet meshtags are available
        """
        solution = self.field.post_processing_solution if u is None else u
        meshtags = self.facet_meshtags if facet_meshtags is None else facet_meshtags

        if meshtags is None:
            raise ValueError(
                f"facet meshtags are required to compute {self.__class__.__name__}"
            )

        if isinstance(solution, dolfinx.fem.Function):
            V = solution.function_space
        else:
            V = self.field.sub_function_space
        mesh = V.mesh
        fdim = mesh.topology.dim - 1

        entities = meshtags.find(self.surface.id)
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(
            V=V, entity_dim=fdim, entities=entities
        )
        values = solution.x.array[dofs]

        # a process may hold no dof of the surface at all, np.min would then raise
        local_min = np.min(values) if values.size > 0 else np.inf

        self.value = mesh.comm.allreduce(local_min, op=MPI.MIN)
        self.data.append(self.value)
