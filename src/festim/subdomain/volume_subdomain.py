import types
from collections.abc import Callable

import dolfinx
import numpy as np
from dolfinx import fem
from dolfinx.mesh import Mesh, locate_entities
from numpy import typing as npt
from scifem.mesh import transfer_meshtags_to_submesh

from festim.material import Material
from festim.subdomain.surface_subdomain import SurfaceSubdomain

# Define the appropriate method based on the version
try:
    from dolfinx.mesh import EntityMap

    entity_map_type = EntityMap
except ImportError:
    entity_map_type = npt.NDArray[np.int32]


class VolumeSubdomain:
    """Volume subdomain class.

    Args:
        id: the id of the volume subdomain (> 0)
        submesh: the submesh of the volume subdomain
        cell_map: the cell map of the volume subdomain
        parent_mesh: the parent mesh of the volume subdomain
        parent_to_submesh: the parent to submesh map of the volume subdomain
        v_map: the vertex map of the volume subdomain
        n_map: the normal map of the volume subdomain
        facet_to_parent: the facet to parent map of the volume subdomain
        ft: the facet meshtags of the volume subdomain
        padded: whether the subdomain is padded (for 0.9 compatibility)
        u: the solution function of the subdomain
        u_n: the previous solution function of the subdomain
        material: the material assigned to the subdomain
        sub_T: the sub temperature field in the subdomain
    """

    id: int
    submesh: dolfinx.mesh.Mesh
    cell_map: "entity_map_type"
    parent_mesh: dolfinx.mesh.Mesh
    parent_to_submesh: "entity_map_type"
    v_map: "entity_map_type"
    n_map: np.ndarray
    facet_to_parent: np.ndarray
    ft: dolfinx.mesh.MeshTags
    padded: bool  # NOTE: Once 0.9 support is dropped, this can be removed
    u: dolfinx.fem.Function
    u_n: dolfinx.fem.Function
    material: Material
    sub_T: fem.Function | float

    def __init__(
        self, id, material, locator: Callable | None = None, name: str | None = None
    ):
        assert id != 0, "Volume subdomain id cannot be 0"
        self.id = id
        self.material = material
        self.locator = locator
        self.name = name

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
        ``.v_map``, ``padded``, and the entity map ``parent_to_submesh``.

        Only used in ``festim.HydrogenTransportProblemDiscontinuous``

        Args:
            mesh (dolfinx.mesh.Mesh): the parent mesh
            marker (dolfinx.mesh.MeshTags): the parent volume markers
        """
        assert marker.dim == mesh.topology.dim
        self.parent_mesh = (
            mesh  # NOTE: it doesn't seem like we use this attribute anywhere
        )
        entities = marker.find(self.id)
        self.submesh, self.cell_map, self.v_map, self.n_map = (
            dolfinx.mesh.create_submesh(mesh, marker.dim, entities)
        )
        if isinstance(entity_map_type, types.GenericAlias):
            num_cells_local = (
                mesh.topology.index_map(marker.dim).size_local
                + mesh.topology.index_map(marker.dim).num_ghosts
            )
            self.parent_to_submesh = np.full(num_cells_local, -1, dtype=np.int32)
            self.parent_to_submesh[self.cell_map] = np.arange(
                len(self.cell_map), dtype=np.int32
            )

            self.padded = False

    def transfer_meshtag(self, mesh: dolfinx.mesh.Mesh, tag: dolfinx.mesh.MeshTags):
        # Transfer meshtags to submesh
        assert self.submesh is not None, "Need to call create_subdomain first"
        self.ft, self.facet_to_parent = transfer_meshtags_to_submesh(
            tag, self.submesh, self.v_map, self.cell_map
        )

    def locate_subdomain_entities(self, mesh: Mesh) -> npt.NDArray[np.int32]:
        """Locates all cells in subdomain borders within domain.

        Args:
            mesh: the mesh of the model

        Returns:
            entities: the entities of the subdomain
        """
        if self.locator is None:
            raise ValueError("No locator function provided for locating cells.")

        entities = locate_entities(mesh, mesh.topology.dim, self.locator)
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
