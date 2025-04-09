import dolfinx.mesh
import numpy as np

from festim.subdomain.surface_subdomain import SurfaceSubdomain


class SurfaceSubdomain1D(SurfaceSubdomain):
    """
    Surface subdomain class for 1D cases

    Args:
        id (int): the id of the surface subdomain
        x (float): the x coordinate of the surface subdomain

    Attributes:
        id (int): the id of the surface subdomain
        x (float): the x coordinate of the surface subdomain

    Examples:

        .. testsetup:: SurfaceSubdomain1D

            from festim import SurfaceSubdomain1D

        .. testcode:: SurfaceSubdomain1D

            SurfaceSubdomain1D(id=1, x=1)
    """

    # FIXME: Rename this to _id and use getter/setter
    id: int
    x: float

    def __init__(self, id: int, x: float) -> None:
        super().__init__(id)
        self.x = x

    def locate_boundary_facet_indices(self, mesh: dolfinx.mesh.Mesh):
        """Locates the dof of the surface subdomain within the function space
        and return the index of the dof

        Args:
            mesh (dolfinx.mesh.Mesh): the mesh of the simulation

        Returns:
            index (np.array): the first value in the list of surface facet
                indices of the subdomain
        """
        assert mesh.geometry.dim == 1, "This method is only for 1D meshes"
        indices = dolfinx.mesh.locate_entities_boundary(
            mesh, 0, lambda x: np.isclose(x[0], self.x)
        )
        return indices
