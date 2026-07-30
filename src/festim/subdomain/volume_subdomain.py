from collections.abc import Callable

import dolfinx
import numpy as np
from dolfinx import fem
from dolfinx.mesh import EntityMap, Mesh
from numpy import typing as npt

try:
    from dolfinx.mesh import transfer_meshtags_to_submesh
except ImportError:
    from scifem.mesh import transfer_meshtags_to_submesh

from festim.material import Material
from festim.subdomain.surface_subdomain import SurfaceSubdomain

entity_map_type = EntityMap


class VolumeSubdomain:
    """Volume subdomain class.

    A subdomain normally has the same topological dimension as the parent mesh
    (codimension 0). Passing ``dim = mesh.topology.dim - 1`` instead declares a
    *codimension 1* subdomain: a manifold embedded in the parent mesh, on which
    transport is solved on a facet submesh and coupled to the surrounding bulk.

    Args:
        id: the id of the volume subdomain (> 0)
        material: the material assigned to the subdomain
        locator: a callable that locates the entities of the subdomain
        name: an optional name for the subdomain
        dim: the topological dimension of the subdomain's entities. ``None``
            (the default) means the same dimension as the parent mesh, i.e.
            codimension 0. Keyword-only. Only codimensions 0 and 1 are
            supported; see :meth:`codim`.

    Attributes:
        id: the id of the volume subdomain (> 0)
        dim: the topological dimension of the subdomain's entities, or ``None``
            for codimension 0
        submesh: the submesh of the volume subdomain
        cell_map: the cell map of the volume subdomain
        parent_mesh: the parent mesh of the volume subdomain
        v_map: the vertex map of the volume subdomain
        n_map: the normal map of the volume subdomain
        ft: the facet meshtags of the volume subdomain. ``None`` for
            codimension 1 subdomains, see :meth:`transfer_meshtag`
        u: the solution function of the subdomain
        u_n: the previous solution function of the subdomain
        material: the material assigned to the subdomain
        sub_T: the sub temperature field in the subdomain
    """

    id: int
    submesh: dolfinx.mesh.Mesh
    cell_map: "entity_map_type"
    parent_mesh: dolfinx.mesh.Mesh
    v_map: "entity_map_type"
    n_map: np.ndarray
    ft: dolfinx.mesh.MeshTags | None
    u: dolfinx.fem.Function
    u_n: dolfinx.fem.Function
    material: Material
    sub_T: fem.Function | float

    def __init__(
        self,
        id,
        material,
        locator: Callable | None = None,
        name: str | None = None,
        *,
        dim: int | None = None,
    ):
        assert id != 0, "Volume subdomain id cannot be 0"
        self.id = id
        self.material = material
        self.locator = locator
        self.name = name
        self.dim = dim

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = None
        elif isinstance(value, str):
            self._name = value
        else:
            raise TypeError("Name must be a string")

    @property
    def dim(self) -> int | None:
        """Topological dimension of the subdomain's entities.

        ``None`` is same as the parent mesh. Use :meth:`entity_dim` to get
        a concrete integer once a mesh is available.
        """
        return self._dim

    @dim.setter
    def dim(self, value):
        if value is None:
            self._dim = None
            return
        # bool is a subclass of int; np.int32/int64 are not
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise TypeError(
                f"Subdomain dimension must be an integer or None, got {type(value)}"
            )
        if value < 0:
            raise ValueError(f"Subdomain {self.id} has dim={value}, must be >= 0")
        self._dim = int(value)

    def codim(self, mesh_dim: int) -> int:
        """Codimension of the subdomain with respect to the parent mesh.

        Args:
            mesh_dim: the topological dimension of the parent mesh

        Returns:
            0 if the subdomain fills cells of the parent mesh, 1 if it lives on
            facets of the parent mesh

        Raises:
            ValueError: if the resulting codimension is not 0 or 1
        """
        if self._dim is None:
            return 0

        codim = mesh_dim - self._dim
        if codim not in (0, 1):
            raise ValueError(
                f"Subdomain {self.id} has dim={self._dim} in a mesh of "
                f"topological dimension {mesh_dim}, giving codimension {codim}. "
                "Only codimensions 0 (cells) and 1 (facets) are supported."
            )
        return codim

    def entity_dim(self, mesh_dim: int) -> int:
        """Resolved topological dimension of the subdomain's entities.

        Unlike :attr:`dim` this is never ``None``, and it validates the
        codimension on the way through.

        Args:
            mesh_dim: the topological dimension of the parent mesh
        """
        return mesh_dim - self.codim(mesh_dim)

    def create_subdomain(self, mesh: dolfinx.mesh.Mesh, marker: dolfinx.mesh.MeshTags):
        """
        Creates the following attributes: ``.parent_mesh``, ``.submesh``, ``.cell_map``,
        and ``.v_map``.

        Only used in ``festim.HydrogenTransportProblemDiscontinuous``

        Args:
            mesh: the parent mesh
            marker: the parent markers. Cell meshtags for a codimension 0
                subdomain, facet meshtags for a codimension 1 subdomain.
        """
        dim = self.entity_dim(mesh.topology.dim)
        assert marker.dim == dim, (
            f"Subdomain {self.id} has dim={dim} but was given markers of "
            f"dimension {marker.dim}. Codimension 1 subdomains must be created "
            "from the facet meshtags, not the volume meshtags."
        )
        self.parent_mesh = mesh
        entities = marker.find(self.id)
        self.submesh, self.cell_map, self.v_map, self.n_map = (
            dolfinx.mesh.create_submesh(mesh, dim, entities)
        )

    def transfer_meshtag(self, mesh: dolfinx.mesh.Mesh, tag: dolfinx.mesh.MeshTags):
        """Transfers parent facet meshtags onto the submesh and stores them in ``.ft``.

        For a codimension 1 subdomain the submesh's *cells* are parent facets, so
        parent facet tags are of the wrong dimension to transfer. ``.ft`` is set
        to ``None`` instead: it is only consumed by ``create_dirichletbc_form``,
        which is unreachable without a strong BC on the manifold.

        Args:
            mesh: the parent mesh
            tag: the parent facet meshtags
        """
        assert self.submesh is not None, "Need to call create_subdomain first"

        if self.codim(mesh.topology.dim) == 1:
            self.ft = None
            return

        sub_tag = transfer_meshtags_to_submesh(
            tag, self.submesh, self.v_map, self.cell_map
        )
        if isinstance(sub_tag, dolfinx.mesh.MeshTags):
            self.ft = sub_tag
        else:
            self.ft, _ = sub_tag

    def locate_subdomain_entities(self, mesh: Mesh) -> npt.NDArray[np.int32]:
        """Locates all entities of the subdomain within the parent mesh.

        For a codimension 0 subdomain these are cells; for a codimension 1
        subdomain they are facets.

        Args:
            mesh: the mesh of the model

        Returns:
            entities: the entities of the subdomain
        """
        if self.locator is None:
            raise ValueError("No locator function provided for locating cells.")


class VolumeSubdomain1D(VolumeSubdomain):
    """Volume subdomain class for 1D cases.

    Args:
        id (int): the id of the volume subdomain
        borders (list of float): the borders of the volume subdomain
        material (festim.Material): the material of the volume subdomain

    Attributes:
        id (int): the id of the volume subdomain
        borders (list of float): the borders of the volume subdomain
        material (festim.Material): the material of the volume subdomain

    Examples:

        .. testsetup:: VolumeSubdomain1D

            from festim import VolumeSubdomain1D, Material
            my_mat = Material(D_0=1, E_D=1, name="test_mat")

        .. testcode:: VolumeSubdomain1D

            VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    """

    def __init__(self, id, borders, material) -> None:
        super().__init__(
            id,
            material,
            locator=lambda x: np.logical_and(x[0] >= borders[0], x[0] <= borders[1]),
        )
        self.borders = borders


def find_volume_from_id(id: int, volumes: list):
    """Returns the correct volume subdomain object from a list of volume ids based on an
    int.

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


# NOTE this still needs to be updated for this case
def map_surface_to_volume_subdomains(
    ft: dolfinx.mesh.MeshTags,
    ct: dolfinx.mesh.MeshTags,
    facet_to_cell: dolfinx.cpp.graph.AdjacencyList_int32,
    volume_subdomains: list[VolumeSubdomain],
    surface_subdomains: list[SurfaceSubdomain],
    comm=None,
) -> dict[SurfaceSubdomain, VolumeSubdomain]:
    """Maps surface subdomains to volume subdomains based on the facet and cell meshtags
    and the facet to cell connectivity.


    Raises:
        AssertionError: if a surface subdomain is connected to multiple volume
            subdomains

    Args:
        ft: the facet meshtags of the parent mesh
        ct: the cell meshtags of the parent mesh
        facet_to_cell: the facet to cell connectivity of the parent mesh
        volume_subdomains: the list of volume subdomains
        surface_subdomains: the list of surface subdomains
        comm: MPI communicator (required for parallel runs)

    Returns:
        dict[SurfaceSubdomain, VolumeSubdomain]: a dictionary mapping surface subdomains
            to volume subdomains
    """

    # get connected cells for tagged facets
    start_indices = facet_to_cell.offsets[ft.indices]
    end_indices = facet_to_cell.offsets[ft.indices + 1]
    num_connections = end_indices - start_indices

    # A facet is connected to at most 2 cells (boundary = 1, interior = 2)
    cell_ids_0 = facet_to_cell.array[start_indices]
    has_second_cell = num_connections == 2
    cell_ids_1 = facet_to_cell.array[start_indices[has_second_cell] + 1]

    connected_cells = np.concatenate([cell_ids_0, cell_ids_1])
    connected_facet_tags = np.concatenate([ft.values, ft.values[has_second_cell]])

    # map connected cells to their cell tags
    sort_idx = np.argsort(ct.indices)
    sorted_ct_indices = ct.indices[sort_idx]
    sorted_ct_values = ct.values[sort_idx]

    idx = np.searchsorted(sorted_ct_indices, connected_cells)
    # mask out-of-bounds
    valid = idx < len(sorted_ct_indices)
    # of those in bounds, check if they actually match
    valid[valid] = sorted_ct_indices[idx[valid]] == connected_cells[valid]

    valid_cell_tags = sorted_ct_values[idx[valid]]
    valid_facet_tags = connected_facet_tags[valid]

    unique_pairs = np.unique(np.vstack((valid_facet_tags, valid_cell_tags)).T, axis=0)
    if comm is not None and comm.size > 1:
        all_pairs = comm.allgather(unique_pairs)
        non_empty = [p for p in all_pairs if len(p) > 0]
        if non_empty:
            unique_pairs = np.unique(np.vstack(non_empty), axis=0)

    surface_tag_to_subdomain = {s.id: s for s in surface_subdomains}
    volume_tag_to_subdomain = {v.id: v for v in volume_subdomains}

    surface_to_subdomain = {}

    for s_tag, v_tag in unique_pairs:
        dolfinx.log.log(
            dolfinx.log.LogLevel.INFO,
            f"Facet tag {s_tag} is connected to cell tag {v_tag}",
        )
        s_subdomain = surface_tag_to_subdomain.get(s_tag)
        v_subdomain = volume_tag_to_subdomain.get(v_tag)

        if s_subdomain and v_subdomain:
            if s_subdomain in surface_to_subdomain:
                assert surface_to_subdomain[s_subdomain] == v_subdomain, (
                    f"Surface subdomain {s_subdomain.id} is connected "
                    f"to multiple volume subdomains: "
                    f"{surface_to_subdomain[s_subdomain].id} and {v_subdomain.id}"
                )
            else:
                surface_to_subdomain[s_subdomain] = v_subdomain
    return surface_to_subdomain
