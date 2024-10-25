import dolfinx.mesh
import numpy as np

import festim as F


class SurfaceSubdomain1D(F.SurfaceSubdomain):
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
        super().__init__(id)
        self.x = x

    def locate_boundary_facet_indices(self, mesh):
        """Locates the dof of the surface subdomain within the function space
        and return the index of the dof

        Args:
            mesh (dolfinx.mesh.Mesh): the mesh of the simulation

        Returns:
            index (np.array): the first value in the list of surface facet
                indices of the subdomain
        """
        assert mesh.geometry.dim == 1, "This method is only for 1D meshes"
        fdim = 0
        self.indices = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[0], self.x)
        )
        return self.indices
