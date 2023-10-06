from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
import numpy as np
from festim import Mesh


class Mesh1D(Mesh):
    """
    1D Mesh

    Attributes:
        vertices (list): the mesh x-coordinates (m)
    """

    def __init__(self, vertices, **kwargs) -> None:
        """Inits Mesh1D

        Args:
            vertices (list): the mesh x-coordinates (m)
        """

        self.vertices = vertices

        mesh = self.generate_mesh()
        super().__init__(mesh=mesh, **kwargs)

    def generate_mesh(self):
        """Generates a 1D mesh"""
        gdim, shape, degree = 1, "interval", 1
        cell = ufl.Cell(shape, geometric_dimension=gdim)
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

        mesh_points = np.reshape(self.vertices, (len(self.vertices), 1))
        indexes = np.arange(self.vertices.shape[0])
        cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)

        return mesh.create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)

    def create_meshtags(self, function_space):
        """Creates the meshtags for a given function space
        Args:
            function_space (dolfinx.fem.function.FunctionSpace): the function
                space of the model

        Returns:
            dolfinx.mesh.MeshTagsMetaClass: the tags containing the facet
                and volume tags
        """

        dofs_L = fem.locate_dofs_geometrical(
            function_space, lambda x: np.isclose(x[0], min(self.vertices))
        )
        dofs_R = fem.locate_dofs_geometrical(
            function_space, lambda x: np.isclose(x[0], max(self.vertices))
        )

        dofs_facets = np.array([dofs_L[0], dofs_R[0]], dtype=np.int32)
        tags_facets = np.array([1, 2], dtype=np.int32)

        facet_dimension = self.mesh.topology.dim - 1
        mesh_tags_facets = mesh.meshtags(
            self.mesh, facet_dimension, dofs_facets, tags_facets
        )

        # TODO implement this
        mesh_tags_volumes = None

        return mesh_tags_facets, mesh_tags_volumes
