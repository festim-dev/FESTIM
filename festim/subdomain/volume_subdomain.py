import dolfinx
import numpy as np
from dolfinx.mesh import locate_entities

from festim.helpers_discontinuity import transfer_meshtags_to_submesh


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

    def __init__(self, id, material):
        self.id = id
        self.material = material

    def locate_subdomain_entities(self, mesh, vdim):
        """Locates all cells in subdomain borders within domain

        Args:
            mesh (dolfinx.mesh.Mesh): the mesh of the model
            vdim (int): the dimension of the volumes of the mesh,
                for 1D this is always 1

        Returns:
            entities (np.array): the entities of the subdomain
        """
        # By default, all entities are included
        # return array like x full of True
        entities = locate_entities(mesh, vdim, lambda x: np.full(x.shape[1], True))
        return entities

    def create_subdomain(self, mesh: dolfinx.mesh.Mesh, marker: dolfinx.mesh.MeshTags):
        """
        Creates the following attributes: ``.parent_mesh``, ``.submesh``, ``.submesh_to_mesh``,
        ``.v_map``, ``padded``, and the entity map ``parent_to_submesh``.

        Only used in ``festim.HTransportProblemDiscontinuous``

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
