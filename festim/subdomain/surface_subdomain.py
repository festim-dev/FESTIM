from dolfinx import fem
import numpy as np


class SurfaceSubdomain1D:
    """
    Surface subdomain class for 1D cases

    Args:
        id (int): the id of the surface subdomain
        x (float): the x coordinate of the surface subdomain

    Attributes:
        id (int): the id of the surface subdomain
        x (float): the x coordinate of the surface subdomain

    """

    def __init__(self, id=None, x=None) -> None:
        self.id = id
        self.x = x
        self.dofs = None

    def locate_dof(self, function_space):
        """Locates the dof of the surface subdomain within the function space"""
        dofs = fem.locate_dofs_geometrical(
            function_space, lambda x: np.isclose(x[0], self.x)
        )
        return dofs[0]
