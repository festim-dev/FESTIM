import fenics as f
import numpy as np


class Mesh:
    def __init__(self, mesh=None, volume_markers=None, surface_markers=None) -> None:
        self.mesh = mesh
        self.volume_markers = volume_markers
        self.surface_markers = surface_markers


class Mesh1D(Mesh):
    def __init__(self) -> None:
        super().__init__()
        self.size = None

    def define_markers(self, materials):
        """Iterates through the mesh and mark them
        based on their position in the domain

        Arguments:
            materials {FESTIM.Materials} -- contains the materials
            size {float} -- size of the domain

        Returns:
            fenics.MeshFunction() -- that contains the subdomains
            (0 if no domain was found)
            fenics.MeshFunction() -- that contains the surfaces
            (0 if no domain was found)
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


class MeshFromVertices(Mesh1D):
    def __init__(self, vertices) -> None:
        super().__init__()
        self.vertices = vertices
        self.size = max(vertices)
        self.generate_mesh_from_vertices()

    def generate_mesh_from_vertices(self):
        '''Generates a 1D mesh from a list of vertices

        Arguments:
        - vertices: list, list of vertices

        Returns:
        - mesh: fenics.Mesh()
        '''
        vertices = sorted(np.unique(self.vertices))
        nb_points = len(vertices)
        nb_cells = nb_points - 1
        editor = f.MeshEditor()
        mesh = f.Mesh()
        editor.open(mesh, "interval", 1, 1)  # top. and geom. dimension are both 1
        editor.init_vertices(nb_points)  # number of vertices
        editor.init_cells(nb_cells)     # number of cells
        for i in range(0, nb_points):
            editor.add_vertex(i, np.array([vertices[i]]))
        for j in range(0, nb_cells):
            editor.add_cell(j, np.array([j, j+1]))
        editor.close()
        self.mesh = mesh


class MeshFromRefinements(Mesh1D):
    def __init__(self, initial_number_of_cells, size, refinements=[]) -> None:
        super().__init__()
        self.initial_unmber_of_cells = initial_number_of_cells
        self.size = size
        self.refinements = refinements
        self.mesh_and_refine()

    def mesh_and_refine(self):
        """Mesh and refine iteratively until meeting the refinement
        conditions.

        Arguments:
            mesh_parameters {dict} -- contains initial number of cells, size,
        and refinements (number of cells and position)

        Returns:
            fenics.Mesh() -- the mesh
        """

        print('Meshing ...')
        initial_number_of_cells = self.initial_unmber_of_cells
        size = self.size
        mesh = f.IntervalMesh(initial_number_of_cells, 0, size)
        for refinement in self.refinements:
            nb_cells_ref = refinement["cells"]
            refinement_point = refinement["x"]
            print("Mesh size before local refinement is " +
                  str(len(mesh.cells())))
            coarse_mesh = True
            while len(mesh.cells()) < \
                    initial_number_of_cells + nb_cells_ref:
                cell_markers = f.MeshFunction(
                    "bool", mesh, mesh.topology().dim())
                cell_markers.set_all(False)
                for cell in f.cells(mesh):
                    if cell.midpoint().x() < refinement_point:
                        cell_markers[cell] = True
                        coarse_mesh = False
                mesh = f.refine(mesh, cell_markers)
                if coarse_mesh:
                    msg = "Infinite loop: Initial number " + \
                        "of cells might be too small"
                    raise ValueError(msg)
            print("Mesh size after local refinement is " +
                  str(len(mesh.cells())))
            initial_number_of_cells = len(mesh.cells())
        self.mesh = mesh


class MeshFromXDMF(Mesh):
    def __init__(self, volume_file, boundary_file) -> None:
        super().__init__()

        self.volume_file = volume_file
        self.boundary_file = boundary_file

        self.mesh = f.Mesh()
        f.XDMFFile(self.volume_file).read(self.mesh)

        self.define_markers()

    def define_markers(self):
        """Reads volume and surface entities from XDMF files

        Raises:
            ValueError: if the reading of attribute f in volumetric file fails
            ValueError: if the reading of attribute f in boundary file fails

        """
        mesh = self.mesh

        # Read tags for volume elements
        volume_markers = f.MeshFunction("size_t", mesh, mesh.topology().dim())
        f.XDMFFile(self.volume_file).read(volume_markers)

        # Read tags for surface elements
        # (can also be used for applying DirichletBC)
        surface_markers = \
            f.MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
        f.XDMFFile(self.boundary_file).read(surface_markers, "f")
        surface_markers = f.MeshFunction("size_t", mesh, surface_markers)

        print("Succesfully load mesh with " + str(len(volume_markers)) + ' cells')
        self.volume_markers = volume_markers
        self.surface_markers = surface_markers
