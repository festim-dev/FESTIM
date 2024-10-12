import ufl
import dolfinx
import numpy as np
from dolfinx.mesh import meshtags


class Mesh:
    """
    Mesh class

    Args:
        mesh (dolfinx.mesh.Mesh, optional): the mesh. Defaults to None.

    Attributes:
        mesh (dolfinx.mesh.Mesh): the mesh
        vdim (int): the dimension of the mesh cells
        fdim (int): the dimension of the mesh facets
        n (ufl.FacetNormal): the normal vector to the facets
    """

    def __init__(self, mesh=None):
        self.mesh = mesh

        if self.mesh is not None:
            # create cell to facet connectivity
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim, self.mesh.topology.dim - 1
            )

            # create facet to cell connectivity
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if isinstance(value, dolfinx.mesh.Mesh):
            self._mesh = value
        else:
            raise TypeError(f"Mesh must be of type dolfinx.mesh.Mesh")

    @property
    def vdim(self):
        return self.mesh.topology.dim

    @property
    def fdim(self):
        return self.mesh.topology.dim - 1

    @property
    def n(self):
        return ufl.FacetNormal(self.mesh)

    def define_meshtags(self, surface_subdomains, volume_subdomains):
        """Defines the facet and volume meshtags of the mesh

        Args:
            surface_subdomains (list of festim.SufaceSubdomains): the surface subdomains of the model
            volume_subdomains (list of festim.VolumeSubdomains): the volume subdomains of the model

        Returns:
            dolfinx.mesh.MeshTags: the facet meshtags
            dolfinx.mesh.MeshTags: the volume meshtags
        """
        # find all cells in domain and mark them as 0
        num_cells = self.mesh.topology.index_map(self.vdim).size_local
        mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
        tags_volumes = np.full(num_cells, 0, dtype=np.int32)

        # find all facets in domain and mark them as 0
        num_facets = self.mesh.topology.index_map(self.fdim).size_local
        mesh_facet_indices = np.arange(num_facets, dtype=np.int32)
        tags_facets = np.full(num_facets, 0, dtype=np.int32)

        for surf in surface_subdomains:
            # find all facets in subdomain and mark them as surf.id
            entities = surf.locate_boundary_facet_indices(self.mesh)
            tags_facets[entities] = surf.id

        for vol in volume_subdomains:
            # find all cells in subdomain and mark them as vol.id
            entities = vol.locate_subdomain_entities(self.mesh, self.vdim)
            tags_volumes[entities] = vol.id

        # define mesh tags
        facet_meshtags = meshtags(self.mesh, self.fdim, mesh_facet_indices, tags_facets)
        volume_meshtags = meshtags(
            self.mesh, self.vdim, mesh_cell_indices, tags_volumes
        )

        return facet_meshtags, volume_meshtags
