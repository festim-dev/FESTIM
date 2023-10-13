import dolfinx.mesh
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

        return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, mesh_points, domain)

    def check_borders(self, volume_subdomains):
        """Checks that the borders of the subdomain are within the domain

        Args:
            mesh (festim.Mesh): the mesh of the model

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
