from dolfinx.mesh import locate_entities

import numpy as np


class VolumeSubdomain:
    """
    Volume subdomain class

    Args:
        id (int): the id of the volume subdomain
    """

    def __init__(self, id, material):
        self.id = id
        self.material = material

    def locate_subdomain_entities(self, mesh, vdim):
        """Locates all cells in subdomain borders within domain

        Args:
            mesh (dolfinx.mesh.Mesh): the mesh of the model
            vdim (int): the dimension of the volumes of the mesh,
                for 1D this is always 1

        Returns:
            entities (np.array): the entities of the subdomain
        """
        # By default, all entities are included
        # return array like x full of True
        entities = locate_entities(mesh, vdim, lambda x: np.full(x.shape[1], True))
        return entities


def find_volume_from_id(id: int, volumes: list):
    """Returns the correct volume subdomain object from a list of volume ids
    based on an int

    Args:
        id (int): the id of the volume subdomain
        volumes (list): the list of volumes

    Returns:
        festim.VolumeSubdomain: the volume subdomain object with the correct id

    Raises:
        ValueError: if the volume name is not found in the list of volumes

    """
    for vol in volumes:
        if vol.id == id:
            return vol
    raise ValueError(f"id {id} not found in list of volumes")
