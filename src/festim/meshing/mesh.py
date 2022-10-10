import fenics as f
import festim


class Mesh:
    """
    Mesh class

    Args:
        mesh (fenics.Mesh, optional): the mesh. Defaults to None.
        volume_markers (fenics.MeshFunction, optional): markers of the mesh cells. Defaults to None.
        surface_markers (fenics.MeshFunction, optional): markers of the mesh facets. Defaults to None.
        type (str, optional): "cartesian", "cylindrical" or "spherical". Defaults to "cartesian".

    Attributes:
        mesh (fenics.Mesh): the mesh
        volume_markers (fenics.MeshFunction): markers of the mesh cells
        surface_markers (fenics.MeshFunction): markers of the mesh facets
        dx (fenics.Measure):
        ds (fenics.Measure):
    """

    def __init__(
        self, mesh=None, volume_markers=None, surface_markers=None, type="cartesian"
    ) -> None:

        self.mesh = mesh
        self.volume_markers = volume_markers
        self.surface_markers = surface_markers
        self.type = type

        self.dx = None
        self.ds = None

    def define_measures(self):
        """Creates the fenics.Measure objects for self.dx and self.ds"""

        self.ds = f.Measure("ds", domain=self.mesh, subdomain_data=self.surface_markers)
        self.dx = f.Measure("dx", domain=self.mesh, subdomain_data=self.volume_markers)
