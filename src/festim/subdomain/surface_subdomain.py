import dolfinx.mesh
import numpy as np
import numpy.typing as npt


class SurfaceSubdomain:
    """
    Surface subdomain class

    Args:
        id (int): the id of the surface subdomain
    """

    id: int

    def __init__(self, id):
        self.id = id

    def locate_boundary_facet_indices(
        self, mesh: dolfinx.mesh.Mesh
    ) -> npt.NDArray[np.int32]:
        """Locates the dof of the surface subdomain within the function space
        and return the index of the dof

        Args:
            mesh (dolfinx.mesh.Mesh): the mesh of the simulation

        Returns:
            index (np.array): the first value in the list of surface facet
                indices of the subdomain
        """
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, fdim + 1)
        return dolfinx.mesh.exterior_facet_indices(mesh.topology)


def find_surface_from_id(id: int, surfaces: list):
    """Returns the correct surface subdomain object from a list of surface ids
    based on an int

    Args:
        id (int): the id of the surface subdomain
        surfaces (list of F.SurfaceSubdomain): the list of surfaces

    Returns:
        festim.SurfaceSubdomain: the surface subdomain object with the correct id

    Raises:
        ValueError: if the surface name is not found in the list of surfaces

    """
    for surf in surfaces:
        if surf.id == id:
            return surf
    raise ValueError(f"id {id} not found in list of surfaces")
