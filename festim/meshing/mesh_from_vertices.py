import fenics as f
import numpy as np
from festim import Mesh1D


class MeshFromVertices(Mesh1D):
    """
    Description of MeshFromVertices

    Args:
        vertices (list): the mesh vertices

    Attributes:
        vertices (list): the mesh vertices
        size (type): the size of the 1D mesh
    """

    def __init__(self, vertices, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vertices = vertices
        self.size = max(vertices)
        self.start = min(vertices)
        self.generate_mesh_from_vertices()

    def generate_mesh_from_vertices(self):
        """Generates a 1D mesh"""
        vertices = sorted(np.unique(self.vertices))
        nb_points = len(vertices)
        nb_cells = nb_points - 1
        editor = f.MeshEditor()
        mesh = f.Mesh()
        editor.open(mesh, "interval", 1, 1)  # top. and geom. dimension are both 1
        editor.init_vertices(nb_points)  # number of vertices
        editor.init_cells(nb_cells)  # number of cells
        for i in range(0, nb_points):
            editor.add_vertex(i, np.array([vertices[i]]))
        for j in range(0, nb_cells):
            editor.add_cell(j, np.array([j, j + 1]))
        editor.close()
        self.mesh = mesh
