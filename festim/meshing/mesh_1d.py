import fenics as f
from festim import Mesh


class Mesh1D(Mesh):
    """
    1D Mesh

    Attributes:
        size (float): the size of the 1D mesh
        start (float): the starting point of the 1D mesh

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.size = None
        self.start = 0

    def define_markers(self, materials):
        """Iterates through the mesh and mark them
        based on their position in the domain

        Arguments:
            materials {festim.Materials} -- contains the materials
        """
        self.volume_markers = self.define_volume_markers(materials)

        self.surface_markers = self.define_surface_markers()

    def define_surface_markers(self):
        """Creates the surface markers

        Returns:
            fenics.MeshFunction: the meshfunction containing the surface
                markers
        """
        surface_markers = f.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1, 0
        )
        surface_markers.set_all(0)
        i = 0
        for facet in f.facets(self.mesh):
            i += 1
            x0 = facet.midpoint()
            surface_markers[facet] = 0
            if f.near(x0.x(), self.start):
                surface_markers[facet] = 1
            if f.near(x0.x(), self.size):
                surface_markers[facet] = 2
        return surface_markers

    def define_volume_markers(self, materials):
        """Creates the volume markers

        Args:
            materials (festim.Materials): the materials

        Returns:
            fenics.MeshFunction: the meshfunction containing the volume
                markers
        """
        volume_markers = f.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim(), 0
        )
        # iterate through the cells of the mesh and mark them
        for cell in f.cells(self.mesh):
            x = cell.midpoint().x()
            subdomain_id = materials.find_subdomain_from_x_coordinate(x)
            volume_markers[cell] = subdomain_id

        return volume_markers

    def define_measures(self, materials):
        """Creates the fenics.Measure objects for self.dx and self.ds"""
        if materials.materials[0].borders is not None:
            materials.check_borders(self.size)
        self.define_markers(materials)
        super().define_measures()
