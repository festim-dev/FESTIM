import fenics as f
from festim import Mesh1D


class MeshFromRefinements(Mesh1D):
    """
    1D mesh with iterative refinements (on the left hand side of the domain)

    Args:
        initial_number_of_cells (float): initial number of cells before
        refinement
        size (float): total size of the 1D mesh
        refinements (list, optional): list of dicts
            {"x": ..., "cells": ...}. For each refinement, the mesh will
            have at least ["cells"] in [0, "x"]. Defaults to [].
        start (float, optional): the starting point of the mesh. Defaults to
            0.

    Attributes:
        initial_number_of_cells (int): initial number of cells before
            refinement
        size (float): total size of the 1D mesh
        refinements (list): list of refinements
    """

    def __init__(
        self, initial_number_of_cells, size, refinements=[], start=0.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.initial_number_of_cells = initial_number_of_cells
        self.size = size
        self.start = start
        self.refinements = refinements
        self.mesh_and_refine()

    def mesh_and_refine(self):
        """Mesh and refine iteratively until meeting the refinement
        conditions.
        """

        print("Meshing ...")
        initial_number_of_cells = self.initial_number_of_cells
        size = self.size
        mesh = f.IntervalMesh(initial_number_of_cells, self.start, size)
        for refinement in self.refinements:
            nb_cells_ref = refinement["cells"]
            refinement_point = refinement["x"]
            print("Mesh size before local refinement is " + str(len(mesh.cells())))
            coarse_mesh = True
            while len(mesh.cells()) < initial_number_of_cells + nb_cells_ref:
                cell_markers = f.MeshFunction("bool", mesh, mesh.topology().dim())
                cell_markers.set_all(False)
                for cell in f.cells(mesh):
                    if cell.midpoint().x() < refinement_point:
                        cell_markers[cell] = True
                        coarse_mesh = False
                mesh = f.refine(mesh, cell_markers)
                if coarse_mesh:
                    msg = (
                        "Infinite loop: Initial number " + "of cells might be too small"
                    )
                    raise ValueError(msg)
            print("Mesh size after local refinement is " + str(len(mesh.cells())))
            initial_number_of_cells = len(mesh.cells())
        self.mesh = mesh
