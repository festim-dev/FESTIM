import fenics as f
from festim import Mesh


class MeshFromXDMF(Mesh):
    """
    Mesh read from XDMF files

    Args:
        volume_file (str): path to the volume file
        boundary_file (str): path the boundary file

    Attributes:
        volume_file (str): name of the volume file
        boundary_file (str): name of the boundary file
        mesh (fenics.Mesh): the mesh
    """

    def __init__(self, volume_file, boundary_file, **kwargs) -> None:
        super().__init__(**kwargs)

        self.volume_file = volume_file
        self.boundary_file = boundary_file

        self.mesh = f.Mesh()
        f.XDMFFile(self.volume_file).read(self.mesh)

        self.define_markers()

    def define_markers(self):
        """Reads volume and surface entities from XDMF files"""
        mesh = self.mesh

        # Read tags for volume elements
        volume_markers = f.MeshFunction("size_t", mesh, mesh.topology().dim())
        f.XDMFFile(self.volume_file).read(volume_markers)

        # Read tags for surface elements
        # (can also be used for applying DirichletBC)
        surface_markers = f.MeshValueCollection(
            "size_t", mesh, mesh.topology().dim() - 1
        )
        f.XDMFFile(self.boundary_file).read(surface_markers, "f")
        surface_markers = f.MeshFunction("size_t", mesh, surface_markers)

        print("Succesfully load mesh with " + str(len(volume_markers)) + " cells")
        self.volume_markers = volume_markers
        self.surface_markers = surface_markers
