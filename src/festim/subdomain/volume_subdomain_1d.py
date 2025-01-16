import numpy as np
import numpy.typing as npt
from dolfinx.mesh import Mesh, locate_entities

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

    def locate_subdomain_entities(self, mesh: Mesh) -> npt.NDArray[np.int32]:
        """Locates all cells in subdomain borders within domain

        Args:
            mesh (dolfinx.mesh.Mesh): the mesh of the model

        Returns:
            entities (np.array): the entities of the subdomain
        """
        entities = locate_entities(
            mesh,
            mesh.topology.dim,
            lambda x: np.logical_and(x[0] >= self.borders[0], x[0] <= self.borders[1]),
        )
        return entities