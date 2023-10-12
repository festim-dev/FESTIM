from dolfinx.mesh import locate_entities
import numpy as np


class VolumeSubdomain1D:
    """
    Volume subdomain class for 1D cases

    Args:
        id (int): the id of the volume subdomain
        borders (list of float): the borders of the volume subdomain
        material (festim.Material): the material of the volume subdomain

    Attributes:
        id (int): the id of the volume subdomain
        borders (list of float): the borders of the volume subdomain
        material (festim.Material): the material of the volume subdomain

    Usage:
        >>> vol_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 1],
        ...     material=F.Material(...))
    """

    def __init__(self, id, borders, material) -> None:
        self.borders = borders
        self.material = material
        self.id = id

    def locate_subdomain_entities(self, mesh, vdim):
        """Locates all cells in subdomain borders within domain

        Args:
            mesh (dolfinx.cpp.mesh.Mesh): the mesh of the model
            vdim (int): the dimension of the volumes of the mesh,
                for 1D this is always 1

        Returns:
            entities (np.array): the entities of the subdomain
        """
        self.check_borders_within_domain(mesh)

        entities = locate_entities(
            mesh.mesh,
            vdim,
            lambda x: np.logical_and(x[0] >= self.borders[0], x[0] <= self.borders[1]),
        )
        return entities

    def check_borders_within_domain(self, mesh):
        """Checks that the borders of the subdomain are within the domain

        Returns:
            bool: True if borders are within domain, False otherwise
        """
        if self.borders[0] < mesh.vertices[0] or self.borders[1] > mesh.vertices[-1]:
            raise ValueError("borders of subdomain are outside of domain")
