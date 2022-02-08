import fenics as f
from FESTIM import Mesh


class Mesh1D(Mesh):
    """
    1D Mesh

    Attributes:
        size (float): the size of the 1D mesh

    """
    def __init__(self) -> None:
        super().__init__()
        self.size = None

    def define_markers(self, materials):
        """Iterates through the mesh and mark them
        based on their position in the domain

        Arguments:
            materials {FESTIM.Materials} -- contains the materials
        """
        mesh = self.mesh
        size = self.size
        volume_markers = f.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
        for cell in f.cells(mesh):
            for material in materials.materials:
                if len(materials.materials) == 1:
                    volume_markers[cell] = material.id
                else:
                    if cell.midpoint().x() >= material.borders[0] \
                    and cell.midpoint().x() <= material.borders[1]:
                        volume_markers[cell] = material.id
        surface_markers = f.MeshFunction(
            "size_t", mesh, mesh.topology().dim()-1, 0)
        surface_markers.set_all(0)
        i = 0
        for facet in f.facets(mesh):
            i += 1
            x0 = facet.midpoint()
            surface_markers[facet] = 0
            if f.near(x0.x(), 0):
                surface_markers[facet] = 1
            if f.near(x0.x(), size):
                surface_markers[facet] = 2
        self.volume_markers = volume_markers
        self.surface_markers = surface_markers

    def define_measures(self, materials):
        """Creates the fenics.Measure objects for self.dx and self.ds
        """
        if len(materials.materials) > 1:
            materials.check_borders(self.size)
        self.define_markers(materials)
        super().define_measures()
