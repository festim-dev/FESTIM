from collections.abc import Callable

import dolfinx
import numpy as np


class SurfaceSubdomain:
    """Surface subdomain class.

    A surface subdomain is a portion of the boundary of a volume subdomain, where
    boundary conditions and surface exports live.

    Args:
        id: the id of the surface subdomain
        locator: a callable function that locates the boundary facets of the subdomain
        dim: the topological dimension of the surface. Defaults to ``None``, meaning the
            facet dimension of the mesh -- the boundary of an ordinary (codim-0) volume
            subdomain. Set it to ``mesh_dim - 2`` to bound a *manifold* volume subdomain
            (``VolumeSubdomain(dim=mesh_dim - 1)``): the endpoints of a line in a 2D
            mesh, the rim of a surface in a 3D mesh. Such a surface carries no meshtag
            -- its entities are located directly on the manifold's submesh, and which
            manifold that is follows from the species of the boundary condition using
            it.

    Examples:

        .. testsetup:: SurfaceSubdomain

            from festim import SurfaceSubdomain

        .. testcode:: SurfaceSubdomain

            SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[0], 1.0))
            SurfaceSubdomain(id=1, locator=lambda x:
            np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
            SurfaceSubdomain(id=1, locator=lambda x:
            np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)))
            SurfaceSubdomain(id=1, locator=lambda x:
            np.logical_and(np.isclose(x[0], 0.0), x[1] <= 0.5))
    """

    id: int
    locator: Callable

    def __init__(
        self, id: int, locator: Callable | None = None, dim: int | None = None
    ):
        self.id = id
        self.locator = locator
        self.dim = dim

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        if value is not None and not isinstance(value, int | np.integer):
            raise TypeError(f"dim must be an integer or None, not {type(value)}")
        if value is not None and value < 0:
            raise ValueError(f"dim must be positive, got {value}")
        self._dim = None if value is None else int(value)

    def codim(self, mesh_dim: int) -> int:
        """The codimension of the surface in a mesh of dimension ``mesh_dim``.

        Args:
            mesh_dim: the topological dimension of the parent mesh

        Returns:
            1 for the boundary of an ordinary volume subdomain, 2 for the boundary of a
            manifold volume subdomain

        Raises:
            ValueError: if the resulting codimension is not 1 or 2
        """
        codim = 1 if self.dim is None else mesh_dim - self.dim
        if codim not in (1, 2):
            raise ValueError(
                f"surface subdomain {self.id} has dim={self.dim} in a mesh of "
                f"dimension {mesh_dim}, ie. codimension {codim}. Only 1 and 2 are "
                "supported."
            )
        return codim

    def locate_boundary_facet_indices(self, mesh: dolfinx.mesh.Mesh) -> np.ndarray:
        """Locate the boundary entities of the subdomain in ``mesh``.

        ``mesh`` is the parent mesh for an ordinary surface, and the submesh of the
        manifold it bounds for a codim-2 one -- in both cases the entities searched are
        the facets of the mesh they are located in.

        Args:
            mesh: a dolfinx mesh object

        Raises:
            ValueError: if no locator function is provided

        Returns:
            the list of entities that belong to the subdomain
        """
        if self.locator is None:
            raise ValueError(
                "No locator function provided for locating boundary facets."
            )

        return dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, self.locator
        )


class SurfaceSubdomain1D(SurfaceSubdomain):
    """Surface subdomain class for 1D cases.

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
    """Returns the correct surface subdomain object from a list of surface ids based on
    an int.

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
