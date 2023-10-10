from dolfinx.mesh import locate_entities
import numpy as np


class VolumeSubdomain1D:
    """
    Volume subdomain class for 1D cases
    """

    def __init__(self, id=None, borders=None, material=None) -> None:
        """Inits Mesh
        Args:
            id (int): the id of the volume subdomain
            borders (list of float): the borders of the volume subdomain
            material (festim.Material): the material of the volume subdomain
        """
        self.borders = borders
        material = material
        self.id = id

    def locate_subdomain_entities(self, mesh, vdim):
        """Locates all cells in subdomain within domain"""
        entities = locate_entities(
            mesh,
            vdim,
            lambda x: np.logical_and(x[0] >= self.borders[0], x[0] <= self.borders[1]),
        )
        return entities
