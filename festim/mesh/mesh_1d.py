from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
import numpy as np
from festim import Mesh


class Mesh1D(Mesh):
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

    def generate_mesh(self):
        """Generates a 1D mesh"""
        gdim, shape, degree = 1, "interval", 1
        cell = ufl.Cell(shape, geometric_dimension=gdim)
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

        mesh_points = np.reshape(self.vertices, (len(self.vertices), 1))
        indexes = np.arange(self.vertices.shape[0])
        cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)

        return mesh.create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)
