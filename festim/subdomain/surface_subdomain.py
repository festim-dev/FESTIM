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

    Usage:
        >>> surf_subdomain = F.SurfaceSubdomain1D(id=1, x=1)
    """

    def __init__(self, id, x) -> None:
        self.id = id
        self.x = x

    def locate_dof(self, function_space):
        """Locates the dof of the surface subdomain within the function space

        Args:
            function_space (dolfinx.fem.FunctionSpace): the function space of
                the model

        Returns:
            dof (np.array): the first value in the list of dofs of the surface
                subdomain
        """
        dofs = fem.locate_dofs_geometrical(
            function_space, lambda x: np.isclose(x[0], self.x)
        )
        return dofs[0]
