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
        self.volume_markers = self.define_volume_markers(materials)

        self.surface_markers = self.define_surface_markers()

    def define_surface_markers(self):
        surface_markers = f.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim()-1, 0)
        surface_markers.set_all(0)
        i = 0
        for facet in f.facets(self.mesh):
            i += 1
            x0 = facet.midpoint()
            surface_markers[facet] = 0
            if f.near(x0.x(), 0):
                surface_markers[facet] = 1
            if f.near(x0.x(), self.size):
                surface_markers[facet] = 2
        return surface_markers

    def define_volume_markers(self, materials):
        volume_markers = f.MeshFunction("size_t", self.mesh, self.mesh.topology().dim(), 0)
        # iterate through the cells of the mesh
        for cell in f.cells(self.mesh):
            for material in materials.materials:
                if material.borders is None:
                    volume_markers[cell] = material.id
                else:
                    if isinstance(material.borders[0], list) and len(material.borders) > 1:
                        list_of_borders = material.borders
                    else:
                        list_of_borders = [material.borders]
                    if isinstance(material.id, list):
                        subdomains = material.id
                    else:
                        subdomains = [material.id for _ in range(len(list_of_borders))]

                    for borders, subdomain in zip(list_of_borders, subdomains):
                        if borders[0] <= cell.midpoint().x() <= borders[1]:
                            volume_markers[cell] = subdomain
        return volume_markers

    def define_measures(self, materials):
        """Creates the fenics.Measure objects for self.dx and self.ds
        """
        if materials.materials[0].borders is not None:
            materials.check_borders(self.size)
        self.define_markers(materials)
        super().define_measures()
