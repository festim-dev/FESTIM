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
            mesh (dolfinx.mesh.Mesh): the mesh of the model
            vdim (int): the dimension of the volumes of the mesh,
                for 1D this is always 1

        Returns:
            entities (np.array): the entities of the subdomain
        """
        entities = locate_entities(
            mesh,
            vdim,
            lambda x: np.logical_and(x[0] >= self.borders[0], x[0] <= self.borders[1]),
        )
        return entities


def find_volume_from_id(id: int, volumes: list):
    """Returns the correct volume subdomain object from a list of volume ids
    based on an int

    Args:
        id (int): the id of the volume subdomain
        volumes (list): the list of volumes

    Returns:
        volume (festim.VolumeSubdomain1D): the volume subdomain object with the correct id

    Raises:
        ValueError: if the volume name is not found in the list of volumes

    """
    for vol in volumes:
        if vol.id == id:
            return vol
    raise ValueError(f"id {id} not found in list of volumes")
