import dolfinx.mesh
from mpi4py import MPI
import ufl
import numpy as np
import festim as F
from dolfinx.mesh import meshtags


class Mesh1D(F.Mesh):
    """
    1D Mesh

    Args:
        vertices (list): the mesh x-coordinates (m)

    Attributes:
        vertices (list): the mesh x-coordinates (m)
    """

    def __init__(self, vertices, **kwargs) -> None:
        self.vertices = vertices

        mesh = self.generate_mesh()
        super().__init__(mesh=mesh, **kwargs)

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = sorted(np.unique(value))

    def generate_mesh(self):
        """Generates a 1D mesh"""
        gdim, shape, degree = 1, "interval", 1
        cell = ufl.Cell(shape, geometric_dimension=gdim)
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

        mesh_points = np.reshape(self.vertices, (len(self.vertices), 1))
        indexes = np.arange(self.vertices.shape[0])
        cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)

        return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)

    def check_borders(self, volume_subdomains):
        """Checks that the borders of the subdomain are within the domain

        Args:
            volume_subdomains (list of festim.VolumeSubdomain1D): the volume subdomains

        Raises:
            Value error: if borders outside the domain
        """
        # check that subdomains are connected
        all_borders = [border for vol in volume_subdomains for border in vol.borders]
        sorted_borders = np.sort(all_borders)
        for start, end in zip(sorted_borders[1:-2:2], sorted_borders[2:-1:2]):
            if start != end:
                raise ValueError("Subdomain borders don't match to each other")

        # check volume subdomain is defined
        # TODO this possible by default
        if len(all_borders) == 0:
            raise ValueError("No volume subdomains defined")

        # check that subdomains are within the domain
        if (
            sorted_borders[0] != self.vertices[0]
            or sorted_borders[-1] != self.vertices[-1]
        ):
            raise ValueError("borders dont match domain borders")

    def define_meshtags(self, surface_subdomains, volume_subdomains):
        """Defines the facet and volume meshtags of the mesh

        Args:
            surface_subdomains (list of festim.SufaceSubdomains): the surface subdomains of the model
            volume_subdomains (list of festim.VolumeSubdomains): the volume subdomains of the model

        Returns:
            dolfinx.mesh.MeshTags: the facet meshtags
            dolfinx.mesh.MeshTags: the volume meshtags
        """
        facet_indices, tags_facets = [], []

        # find all cells in domain and mark them as 0
        num_cells = self.mesh.topology.index_map(self.vdim).size_local
        mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
        tags_volumes = np.full(num_cells, 0, dtype=np.int32)

        for surf in surface_subdomains:
            facet_index = surf.locate_boundary_facet_indices(self.mesh, self.fdim)
            facet_indices.append(facet_index)
            tags_facets.append(surf.id)

        for vol in volume_subdomains:
            # find all cells in subdomain and mark them as sub_dom.id
            entities = vol.locate_subdomain_entities(self.mesh, self.vdim)
            tags_volumes[entities] = vol.id

        # check if all borders are defined
        self.check_borders(volume_subdomains)

        # dofs and tags need to be in np.in32 format for meshtags
        facet_indices = np.array(facet_indices, dtype=np.int32)
        tags_facets = np.array(tags_facets, dtype=np.int32)

        # define mesh tags
        facet_meshtags = meshtags(self.mesh, self.fdim, facet_indices, tags_facets)
        volume_meshtags = meshtags(
            self.mesh, self.vdim, mesh_cell_indices, tags_volumes
        )

        return facet_meshtags, volume_meshtags
