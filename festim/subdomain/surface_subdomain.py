from dolfinx import fem
import dolfinx.mesh
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

    def locate_dof(self, mesh, fdim):
        """Locates the dof of the surface subdomain within the function space

        Args:
            mesh (dolfinx.mesh.Mesh): the mesh of the simulation
            fdim (int): the dimension of the model facets

        Returns:
            dof (np.array): the first value in the list of dofs of the surface
                subdomain
        """
        dofs = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[0], self.x)
        )
        return dofs[0]
