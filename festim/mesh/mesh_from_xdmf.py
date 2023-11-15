from dolfinx.io import XDMFFile
from mpi4py import MPI
import festim as F


class MeshFromXDMF(F.Mesh):
    """
    Mesh read from the XDMF files

    Args:
        volume_file (str): path to the volume file
        facet_file (str): path to the facet file
        mesh_name (str, optional): name of the mesh in the XDMF file. Defaults to "Grid".
        meshtags_name (str, optional): name of the meshtags in the XDMF file. Defaults to "Grid".

    Attributes:
        volume_file (str): path to the volume file
        facet_file (str): path to the facet file
        mesh_name (str, optional): name of the mesh in the XDMF file. Defaults to "Grid".
        meshtags_name (str, optional): name of the meshtags in the XDMF file. Defaults to "Grid".
        mesh (fenics.mesh.Mesh): the fenics mesh
    """

    def __init__(
        self, volume_file, facet_file, mesh_name="Grid", meshtags_name="Grid"
    ) -> None:
        self.volume_file = volume_file
        self.facet_file = facet_file
        self.mesh_name = mesh_name
        self.meshtags_name = meshtags_name

        volumes_file = XDMFFile(MPI.COMM_WORLD, self.volume_file, "r")
        mesh = volumes_file.read_mesh(name=f"{self.mesh_name}")

        super().__init__(mesh=mesh)

    def define_surface_markers(self):
        """Creates the facet meshtags

        Returns:
            dolfinx.MeshTags: the facet meshtags
        """
        facets_file = XDMFFile(MPI.COMM_WORLD, self.facet_file, "r")
        facet_meshtags = facets_file.read_meshtags(
            self.mesh, name=f"{self.meshtags_name}"
        )

        return facet_meshtags

    def define_volume_markers(self):
        """Creates the volume meshtags

        Returns:
            dolfinx.MeshTags: the volume meshtags
        """
        volume_file = XDMFFile(MPI.COMM_WORLD, self.volume_file, "r")

        volume_meshtags = volume_file.read_meshtags(
            self.mesh, name=f"{self.meshtags_name}"
        )

        return volume_meshtags
