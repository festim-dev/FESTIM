import dolfinx
from dolfinx.mesh import Mesh, locate_entities
import numpy as np
from numpy import typing as npt
from typing import Callable

from festim.helpers_discontinuity import transfer_meshtags_to_submesh
from festim.material import Material


class VolumeSubdomain:
    """
    Volume subdomain class

    Args:
        id (int): the id of the volume subdomain
        material (festim.Material): the material assigned to the subdomain
    """

    id: int
    submesh: dolfinx.mesh.Mesh
    submesh_to_mesh: np.ndarray
    parent_mesh: dolfinx.mesh.Mesh
    parent_to_submesh: np.ndarray
    v_map: np.ndarray
    facet_to_parent: np.ndarray
    ft: dolfinx.mesh.MeshTags
    padded: bool
    u: dolfinx.fem.Function
    u_n: dolfinx.fem.Function
    material: Material

    def __init__(self, id, material, locator: Callable | None = None):
        self.id = id
        self.material = material
        self.locator = locator

    def create_subdomain(self, mesh: dolfinx.mesh.Mesh, marker: dolfinx.mesh.MeshTags):
        """
        Creates the following attributes: ``.parent_mesh``, ``.submesh``, ``.submesh_to_mesh``,
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
        self.submesh, self.submesh_to_mesh, self.v_map = dolfinx.mesh.create_submesh(
            mesh, marker.dim, marker.find(self.id)
        )[0:3]
        num_cells_local = (
            mesh.topology.index_map(marker.dim).size_local
            + mesh.topology.index_map(marker.dim).num_ghosts
        )
        self.parent_to_submesh = np.full(num_cells_local, -1, dtype=np.int32)
        self.parent_to_submesh[self.submesh_to_mesh] = np.arange(
            len(self.submesh_to_mesh), dtype=np.int32
        )
        self.padded = False

    def transfer_meshtag(self, mesh: dolfinx.mesh.Mesh, tag: dolfinx.mesh.MeshTags):
        # Transfer meshtags to submesh
        assert self.submesh is not None, "Need to call create_subdomain first"
        self.ft, self.facet_to_parent = transfer_meshtags_to_submesh(
            mesh, tag, self.submesh, self.v_map, self.submesh_to_mesh
        )

    def locate_subdomain_entities(self, mesh: Mesh) -> npt.NDArray[np.int32]:
        """Locates all cells in subdomain borders within domain

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
    """
    Volume subdomain class for 1D cases

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
    """Returns the correct volume subdomain object from a list of volume ids
    based on an int

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
