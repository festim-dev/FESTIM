import dolfinx
import numpy as np
from typing import Callable


class SurfaceSubdomain:
    """
    Surface subdomain class

    Args:
        id: the id of the surface subdomain
        locator: a callable function that locates the boundary facets of the subdomain

    Examples:

        .. testsetup:: SurfaceSubdomain

            from festim import SurfaceSubdomain

        .. testcode:: SurfaceSubdomain

            SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[0], 1.0))
            SurfaceSubdomain(id=1, locator=lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
            SurfaceSubdomain(id=1, locator=lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)))
            SurfaceSubdomain(id=1, locator=lambda x: np.logical_and(np.isclose(x[0], 0.0), x[1] <= 0.5))
    """

    id: int
    locator: Callable

    def __init__(self, id: int, locator: Callable = None):
        self.id = id
        self.locator = locator

    def locate_boundary_facet_indices(self, mesh: dolfinx.mesh.Mesh) -> np.ndarray:
        """Locate boundary facets of the subdomain in the mesh.

        Args:
            mesh: a dolfinx mesh object

        Raises:
            ValueError: if no locator function is provided

        Returns:
            the list of entities (facets) that belong to the subdomain
        """
        if self.locator is None:
            raise ValueError(
                "No locator function provided for locating boundary facets."
            )

        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, self.locator
        )


class SurfaceSubdomain1D(SurfaceSubdomain):
    """
    Surface subdomain class for 1D cases

    Args:
        id: the id of the surface subdomain
        x: the x coordinate of the surface subdomain

    Attributes:
        id: the id of the surface subdomain
        x: the x coordinate of the surface subdomain

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
        super().__init__(id, locator=lambda x_: np.isclose(x_[0], x))
        self.x = x


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
