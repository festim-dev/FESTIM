from mpi4py import MPI

import basix.ufl
import dolfinx.mesh
import numpy as np
import ufl

from festim.mesh.mesh import Mesh


class Mesh1D(Mesh):
    """
    1D Mesh

    Args:
        vertices (list or np.ndarray): the mesh x-coordinates (m)

    Attributes:
        vertices (np.ndarray): the mesh x-coordinates (m)
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
        self._vertices = np.sort(np.unique(value)).astype(np.float64)

    def generate_mesh(self):
        """Generates a 1D mesh"""

        if MPI.COMM_WORLD.rank == 0:
            mesh_points = np.reshape(self.vertices, (len(self.vertices), 1))
            indexes = np.arange(self.vertices.shape[0])
            cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)

        else:
            mesh_points = np.empty((0, 1), dtype=np.float64)
            cells = np.empty((0, 2), dtype=np.int64)

        degree = 1
        domain = ufl.Mesh(
            basix.ufl.element(basix.ElementFamily.P, "interval", degree, shape=(1,))
        )
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

    def define_meshtags(self, surface_subdomains, volume_subdomains, interfaces=None):
        # check if all borders are defined
        self.check_borders(volume_subdomains)
        return super().define_meshtags(
            surface_subdomains, volume_subdomains, interfaces
        )
