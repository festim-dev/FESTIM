import numpy as np
from dolfinx.mesh import locate_entities

from festim.subdomain import VolumeSubdomain


class VolumeSubdomain1D(VolumeSubdomain):
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
        super().__init__(id, material)
        self.borders = borders

    def locate_subdomain_entities(self, mesh, vdim):
        """Locates all cells in subdomain borders within domain

        Args:
            mesh (dolfinx.mesh.Mesh): the mesh of the model
            vdim (int): the dimension of the volumes of the mesh,
                for 1D this is always 1

        Returns:
            entities (np.array): the entities of the subdomain
        """
        self.entities = locate_entities(
            mesh,
            vdim,
            lambda x: np.logical_and(x[0] >= self.borders[0], x[0] <= self.borders[1]),
        )
        return self.entities
