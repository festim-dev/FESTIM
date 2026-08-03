from collections.abc import Callable

import dolfinx
import numpy as np
from dolfinx import fem
from dolfinx.mesh import EntityMap, Mesh, locate_entities
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

    Args:
        id: the id of the volume subdomain (> 0)
        submesh: the submesh of the volume subdomain
        cell_map: the cell map of the volume subdomain
        parent_mesh: the parent mesh of the volume subdomain
        v_map: the vertex map of the volume subdomain
        n_map: the normal map of the volume subdomain
        ft: the facet meshtags of the volume subdomain
        u: the solution function of the subdomain
        u_n: the previous solution function of the subdomain
        material: the material assigned to the subdomain
        sub_T: the sub temperature field in the subdomain
        sub_t: for a manifold (codim-1) subdomain, the current time as a constant living
            on its submesh. ``None`` for a codim-0 subdomain, which uses the parent-mesh
            constant
        sub_dt: for a manifold (codim-1) subdomain, the timestep as a constant living on
            its submesh. ``None`` for a codim-0 subdomain
        dim: the topological dimension of the subdomain. Defaults to ``None``, meaning
            the dimension of the mesh. Set it to ``mesh_dim - 1`` to solve a transport
            equation on a manifold embedded in the mesh (a line in a 2D mesh, a surface
            in a 3D mesh). Such a subdomain is tagged in the *facet* meshtags, and can
            be used wherever a surface is expected (eg. ``ParticleFluxBC``).
    """

    id: int
    submesh: dolfinx.mesh.Mesh
    cell_map: "entity_map_type"
    parent_mesh: dolfinx.mesh.Mesh
    v_map: "entity_map_type"
    n_map: np.ndarray
    ft: dolfinx.mesh.MeshTags
    u: dolfinx.fem.Function
    u_n: dolfinx.fem.Function
    material: Material
    sub_T: fem.Function | float
    sub_t: fem.Constant | None
    sub_dt: fem.Constant | None

    def __init__(
        self,
        id,
        material,
        locator: Callable | None = None,
        name: str | None = None,
        dim=None,
    ):
        assert id != 0, "Volume subdomain id cannot be 0"
        self.id = id
        self.sub_t = None
        self.sub_dt = None
        self.material = material
        self.locator = locator
        self.name = name
        self.dim = dim

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        if value is not None and not isinstance(value, int | np.integer):
            raise TypeError(f"dim must be an integer or None, not {type(value)}")
        if value is not None and value < 1:
            raise ValueError(f"dim must be strictly positive, got {value}")
        self._dim = None if value is None else int(value)

    def codim(self, mesh_dim: int) -> int:
        """The codimension of the subdomain in a mesh of dimension ``mesh_dim``.

        Args:
            mesh_dim: the topological dimension of the parent mesh

        Returns:
            0 for a regular volume subdomain, 1 for a manifold subdomain

        Raises:
            ValueError: if the resulting codimension is not 0 or 1
        """
        codim = 0 if self.dim is None else mesh_dim - self.dim
        if codim not in (0, 1):
            raise ValueError(
                f"volume subdomain {self.id} has dim={self.dim} in a mesh of dimension "
                f"{mesh_dim}, ie. codimension {codim}. Only 0 and 1 are supported."
            )
        return codim

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

    def create_subdomain(self, mesh: dolfinx.mesh.Mesh, marker: dolfinx.mesh.MeshTags):
        """
        Creates the following attributes: ``.parent_mesh``, ``.submesh``, ``.cell_map``,
        and ``.v_map``.

        Only used in ``festim.HydrogenTransportProblemDiscontinuous``

        Args:
            mesh (dolfinx.mesh.Mesh): the parent mesh
            marker (dolfinx.mesh.MeshTags): the markers the subdomain is tagged in: the
                volume markers for a codim-0 subdomain, the facet markers for a codim-1
                one
        """
        assert marker.dim == (mesh.topology.dim if self.dim is None else self.dim)
        self.parent_mesh = (
            mesh  # NOTE: it doesn't seem like we use this attribute anywhere
        )
        entities = marker.find(self.id)
        self.submesh, self.cell_map, self.v_map, self.n_map = (
            dolfinx.mesh.create_submesh(mesh, marker.dim, entities)
        )

    def transfer_meshtag(self, mesh: dolfinx.mesh.Mesh, tag: dolfinx.mesh.MeshTags):
        # Transfer meshtags to submesh
        assert self.submesh is not None, "Need to call create_subdomain first"
        if self.codim(mesh.topology.dim) == 1:
            # the parent facet tags are the *cell* tags of a codim-1 submesh, so there
            # is nothing meaningful to transfer. ``ft`` is only read to apply strong
            # Dirichlet BCs, which are not supported on a manifold subdomain yet.
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
        """Locates all entities of the subdomain within the domain.

        These are the cells of the mesh for a regular volume subdomain, and the facets
        for a codim-1 (manifold) one.

        Args:
            mesh: the mesh of the model

        Returns:
            entities: the entities of the subdomain
        """
        if self.locator is None:
            raise ValueError("No locator function provided for locating cells.")

        dim = mesh.topology.dim if self.dim is None else self.dim
        entities = locate_entities(mesh, dim, self.locator)
        return entities


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


def _facet_cell_tag_pairs(
    ft: dolfinx.mesh.MeshTags,
    ct: dolfinx.mesh.MeshTags,
    facet_to_cell: dolfinx.cpp.graph.AdjacencyList_int32,
    comm=None,
):
    """The unique ``(facet tag, cell tag)`` pairs present on the mesh, gathered across
    all ranks."""
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
    return unique_pairs


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

    unique_pairs = _facet_cell_tag_pairs(ft, ct, facet_to_cell, comm)

    surface_tag_to_subdomain = {s.id: s for s in surface_subdomains}
    volume_tag_to_subdomain = {v.id: v for v in volume_subdomains}

    adjacency = map_facet_tags_to_volume_subdomains(
        unique_pairs, surface_tag_to_subdomain, volume_tag_to_subdomain
    )

    surface_to_subdomain = {}
    for s_subdomain, volumes in adjacency.items():
        assert len(volumes) == 1, (
            f"Surface subdomain {s_subdomain.id} is connected "
            f"to multiple volume subdomains: "
            f"{' and '.join(str(v.id) for v in volumes)}"
        )
        surface_to_subdomain[s_subdomain] = volumes[0]
    return surface_to_subdomain


def map_facet_tags_to_volume_subdomains(
    unique_pairs, surface_tag_to_subdomain: dict, volume_tag_to_subdomain: dict
) -> dict:
    """Group the ``(facet tag, cell tag)`` pairs into a facet-subdomain to
    volume-subdomains mapping.

    Unlike :func:`map_surface_to_volume_subdomains` this keeps *every* adjacent volume,
    which is what a manifold subdomain sitting between two volumes needs. The volumes of
    each entry are sorted by id so that the ordering does not depend on how the mesh
    happens to be partitioned.

    Args:
        unique_pairs: the ``(facet tag, cell tag)`` pairs found on the mesh
        surface_tag_to_subdomain: facet tag -> subdomain to report on
        volume_tag_to_subdomain: cell tag -> volume subdomain

    Returns:
        a dictionary mapping each facet subdomain to the list of volume subdomains it
        is adjacent to
    """
    adjacency = {}
    for s_tag, v_tag in unique_pairs:
        dolfinx.log.log(
            dolfinx.log.LogLevel.INFO,
            f"Facet tag {s_tag} is connected to cell tag {v_tag}",
        )
        s_subdomain = surface_tag_to_subdomain.get(s_tag)
        v_subdomain = volume_tag_to_subdomain.get(v_tag)

        if s_subdomain is not None and v_subdomain is not None:
            adjacency.setdefault(s_subdomain, [])
            if v_subdomain not in adjacency[s_subdomain]:
                adjacency[s_subdomain].append(v_subdomain)

    for volumes in adjacency.values():
        volumes.sort(key=lambda v: v.id)
    return adjacency


def map_manifold_to_volume_subdomains(
    ft: dolfinx.mesh.MeshTags,
    ct: dolfinx.mesh.MeshTags,
    facet_to_cell: dolfinx.cpp.graph.AdjacencyList_int32,
    volume_subdomains: list[VolumeSubdomain],
    manifold_subdomains: list[VolumeSubdomain],
    comm=None,
) -> dict[VolumeSubdomain, list[VolumeSubdomain]]:
    """Maps each codim-1 (manifold) volume subdomain to the volume subdomains it is
    adjacent to: one for a manifold on the boundary of the domain, two for one sitting
    on an interior interface.

    Args:
        ft: the facet meshtags of the parent mesh
        ct: the cell meshtags of the parent mesh
        facet_to_cell: the facet to cell connectivity of the parent mesh
        volume_subdomains: the list of volume subdomains
        manifold_subdomains: the codim-1 volume subdomains
        comm: MPI communicator (required for parallel runs)

    Returns:
        a dictionary mapping each manifold subdomain to its adjacent volume subdomains,
        sorted by id

    Raises:
        ValueError: if a manifold is adjacent to no volume, or to more than two
    """
    unique_pairs = _facet_cell_tag_pairs(ft, ct, facet_to_cell, comm)
    bulk = [v for v in volume_subdomains if v not in manifold_subdomains]
    adjacency = map_facet_tags_to_volume_subdomains(
        unique_pairs, {m.id: m for m in manifold_subdomains}, {v.id: v for v in bulk}
    )

    for manifold in manifold_subdomains:
        volumes = adjacency.get(manifold, [])
        if not 1 <= len(volumes) <= 2:
            raise ValueError(
                f"codim-1 volume subdomain {manifold.id} is adjacent to "
                f"{len(volumes)} volume subdomains; expected 1 (on the boundary of the "
                "domain) or 2 (on an interface)"
            )
    return adjacency
