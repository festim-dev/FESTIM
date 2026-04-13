from mpi4py import MPI

from dolfinx.io import XDMFFile

from festim.mesh.mesh import Mesh


class MeshFromXDMF(Mesh):
    """
    Mesh read from the XDMF files

    Args:
        volume_file (str): path to the volume file
        facet_file (str): path to the facet file
        mesh_name (str, optional): name of the mesh in the volume XDMF file.
            Defaults to "Grid".
        surface_meshtags_name (str, optional): name of the surface meshtags in the
            facet XDMF file. Defaults to "Grid".
        volume_meshtags_name (str, optional): name of the volume meshtags in the
            volume XDMF file. Defaults to "Grid".

    Attributes:
        volume_file (str): path to the volume file
        facet_file (str): path to the facet file
        mesh_name (str): name of the mesh in the volume XDMF file.
        surface_meshtags_name (str): name of the surface meshtags in the facet XDMF file.
        volume_meshtags_name (str): name of the volume meshtags in the volume XDMF file
        mesh (fenics.mesh.Mesh): the fenics mesh
    """

    def __init__(
        self,
        volume_file,
        facet_file,
        mesh_name="Grid",
        surface_meshtags_name="Grid",
        volume_meshtags_name="Grid",
    ) -> None:
        self.volume_file = volume_file
        self.facet_file = facet_file
        self.mesh_name = mesh_name
        self.surface_meshtags_name = surface_meshtags_name
        self.volume_meshtags_name = volume_meshtags_name

        volumes_file = XDMFFile(MPI.COMM_WORLD, self.volume_file, "r")
        mesh = volumes_file.read_mesh(name=f"{self.mesh_name}")

        super().__init__(mesh=mesh)

    def define_surface_meshtags(self):
        """Creates the facet meshtags

        Returns:
            dolfinx.MeshTags: the facet meshtags
        """
        facets_file = XDMFFile(MPI.COMM_WORLD, self.facet_file, "r")
        facet_meshtags = facets_file.read_meshtags(
            self.mesh, name=f"{self.surface_meshtags_name}"
        )

        return facet_meshtags

    def define_volume_meshtags(self):
        """Creates the volume meshtags

        Returns:
            dolfinx.MeshTags: the volume meshtags
        """
        volume_file = XDMFFile(MPI.COMM_WORLD, self.volume_file, "r")

        volume_meshtags = volume_file.read_meshtags(
            self.mesh, name=f"{self.volume_meshtags_name}"
        )

        return volume_meshtags
