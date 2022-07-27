# Unit tests meshing
from festim import Materials, Material
from festim import Mesh, Mesh1D, MeshFromRefinements, MeshFromVertices, MeshFromXDMF
import fenics
import pytest
from pathlib import Path
import numpy as np


def test_mesh_and_refine_meets_refinement_conditions():
    """
    Test that function mesh_and_refine() gives the right
    refinement conditions
    """

    def create_subdomains(x1, x2):
        class domain(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= x1 and x[0] <= x2

        domain = domain()
        return domain

    refinements = [
        [{"cells": 2, "x": 0.5}, {"cells": 3, "x": 0.25}],
        [{"cells": 3, "x": 0.5}, {"cells": 11, "x": 0.25}],
    ]
    for refinement in refinements:
        my_mesh = MeshFromRefinements(
            initial_number_of_cells=2, size=1, refinements=refinement
        )
        mesh = my_mesh.mesh

        mf1 = fenics.MeshFunction("size_t", mesh, 1)
        mf2 = fenics.MeshFunction("size_t", mesh, 1)
        subdomain1 = create_subdomains(0, refinement[1]["x"])
        subdomain1.mark(mf1, 1)
        subdomain2 = create_subdomains(0, refinement[0]["x"])
        subdomain2.mark(mf2, 2)
        nb_cell_1 = 0
        nb_cell_2 = 0
        for cell in fenics.cells(mesh):
            cell_no = cell.index()
            if mf1.array()[cell_no] == 1:
                nb_cell_1 += 1
            if mf2.array()[cell_no] == 2:
                nb_cell_2 += 1
        assert nb_cell_1 >= refinement[0]["cells"]
        assert nb_cell_2 >= refinement[1]["cells"]


class TestDefineMarkers:
    my_mesh = Mesh1D()
    my_mesh.mesh = fenics.UnitIntervalMesh(19)
    my_mesh.size = 1

    def test_2_materials_2_subdomains(self):
        """
        Test that subdomains are assigned properly
        """
        materials = [
            Material(id=1, D_0=None, E_D=None, borders=[0, 0.5]),
            Material(id=2, D_0=None, E_D=None, borders=[0.5, 1]),
        ]
        my_mats = Materials(materials)

        self.my_mesh.define_markers(my_mats)
        for cell in fenics.cells(self.my_mesh.mesh):
            if cell.midpoint().x() <= 0.5:
                assert self.my_mesh.volume_markers[cell] == 1
            else:
                assert self.my_mesh.volume_markers[cell] == 2

    def test_1_material_2_subdomains(self):
        my_mats = Materials([Material([1, 2], 1, 0, borders=[[0, 0.5], [0.5, 1]])])

        self.my_mesh.define_markers(my_mats)
        for cell in fenics.cells(self.my_mesh.mesh):
            if cell.midpoint().x() <= 0.5:
                assert self.my_mesh.volume_markers[cell] == 1
            else:
                assert self.my_mesh.volume_markers[cell] == 2

    def test_2_materials_3_subdomains(self):
        my_mats = Materials(
            [
                Material([1, 2], 1, 0, borders=[[0, 0.25], [0.25, 0.5]]),
                Material(3, 1, 0, borders=[0.5, 1]),
            ]
        )

        self.my_mesh.define_markers(my_mats)
        for cell in fenics.cells(self.my_mesh.mesh):
            if 0 <= cell.midpoint().x() <= 0.25:
                assert self.my_mesh.volume_markers[cell] == 1
            elif 0.25 <= cell.midpoint().x() <= 0.5:
                assert self.my_mesh.volume_markers[cell] == 2
            else:
                assert self.my_mesh.volume_markers[cell] == 3

    def test_1_material_1_id_2_borders(self):
        my_mats = Materials([Material(1, 1, 0, borders=[[0, 0.5], [0.7, 1]])])

        self.my_mesh.define_markers(my_mats)

        for cell in fenics.cells(self.my_mesh.mesh):
            if 0 < cell.midpoint().x() < 0.5:
                assert self.my_mesh.volume_markers[cell] == 1
            elif 0.7 < cell.midpoint().x() < 1:
                assert self.my_mesh.volume_markers[cell] == 1


class TestMeshVerticesStartNonZero:
    """Tests to check that MeshFromVertices that don't start
    at zero are correctly tagged
    """

    my_mesh = MeshFromVertices([1, 2, 3])
    materials = [
        Material(id=1, D_0=None, E_D=None, borders=[1, 2]),
        Material(id=2, D_0=None, E_D=None, borders=[2, 3]),
    ]
    my_mats = Materials(materials)

    my_mesh.define_markers(my_mats)

    def test_volume_markers(self):
        for cell in fenics.cells(self.my_mesh.mesh):
            if 1 < cell.midpoint().x() < 2:
                assert self.my_mesh.volume_markers[cell] == 1
            elif 2 < cell.midpoint().x() < 3:
                assert self.my_mesh.volume_markers[cell] == 2

    def test_surface_markers(self):
        for facet in fenics.facets(self.my_mesh.mesh):
            x0 = facet.midpoint()
            if fenics.near(x0.x(), 1):
                assert self.my_mesh.surface_markers[facet] == 1
            if fenics.near(x0.x(), 3):
                assert self.my_mesh.surface_markers[facet] == 2


class TestDefineMarkersStartNonZero:
    """Tests to check that MeshFromRefinements that don't start at zero are correctly tagged"""

    my_mesh = MeshFromRefinements(initial_number_of_cells=10, size=2, start=1)
    materials = [
        Material(id=1, D_0=None, E_D=None, borders=[1, 2]),
        Material(id=2, D_0=None, E_D=None, borders=[2, 3]),
    ]
    my_mats = Materials(materials)

    my_mesh.define_markers(my_mats)

    def test_volume_markers(self):
        for cell in fenics.cells(self.my_mesh.mesh):
            if 1 < cell.midpoint().x() < 2:
                assert self.my_mesh.volume_markers[cell] == 1
            elif 2 < cell.midpoint().x() < 3:
                assert self.my_mesh.volume_markers[cell] == 2

    def test_surface_markers(self):
        for facet in fenics.facets(self.my_mesh.mesh):
            x0 = facet.midpoint()
            if fenics.near(x0.x(), 1):
                assert self.my_mesh.surface_markers[facet] == 1
            if fenics.near(x0.x(), 3):
                assert self.my_mesh.surface_markers[facet] == 2


def test_create_mesh_xdmf(tmpdir):

    # write xdmf file
    mesh = fenics.UnitSquareMesh(10, 10)
    file1 = tmpdir.join("mesh.xdmf")
    f = fenics.XDMFFile(str(Path(file1)))
    f.write(mesh)

    # write files
    mf_cells = fenics.MeshFunction("size_t", mesh, mesh.topology().dim())
    mf_facets = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    file1 = tmpdir.join("cell_file.xdmf")
    file2 = tmpdir.join("facet_file.xdmf")
    fenics.XDMFFile(str(Path(file1))).write(mf_cells)
    fenics.XDMFFile(str(Path(file2))).write(mf_facets)

    # read mesh
    my_mesh = MeshFromXDMF(volume_file=str(Path(file1)), boundary_file=str(Path(file1)))

    # check that vertices are the same
    vertices_mesh = []
    for f in fenics.facets(mesh):
        for v in fenics.vertices(f):
            vertices_mesh.append([v.point().x(), v.point().y()])

    vertices_mesh2 = []
    for f in fenics.facets(my_mesh.mesh):
        for v in fenics.vertices(f):
            vertices_mesh2.append([v.point().x(), v.point().y()])
    for i in range(0, len(vertices_mesh)):
        assert vertices_mesh[i] == vertices_mesh2[i]


def test_subdomains_from_xdmf(tmpdir):

    # create mesh functions
    mesh = fenics.UnitCubeMesh(6, 6, 6)
    mf_cells = fenics.MeshFunction("size_t", mesh, mesh.topology().dim())
    mf_facets = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

    # assign values for cells and facets
    for i, c in enumerate(fenics.cells(mesh)):
        mf_cells[c] = i
    for i, f in enumerate(fenics.facets(mesh)):
        mf_facets[c] = i

    # write files
    file1 = tmpdir.join("cell_file.xdmf")
    file2 = tmpdir.join("facet_file.xdmf")
    fenics.XDMFFile(str(Path(file1))).write(mf_cells)
    fenics.XDMFFile(str(Path(file2))).write(mf_facets)

    # read files
    my_mesh = MeshFromXDMF(volume_file=str(Path(file1)), boundary_file=str(Path(file2)))
    mf_cells_2, mf_facets_2 = my_mesh.volume_markers, my_mesh.surface_markers

    # check
    for cell in fenics.cells(mesh):
        assert mf_cells[cell] == mf_cells_2[cell]
        assert mf_facets[cell] == mf_facets_2[cell]


def test_create_mesh_inbuilt():
    """
    Test when mesh is given by the user
    """
    # create mesh
    mesh = fenics.UnitSquareMesh(10, 10)

    my_mesh = Mesh(mesh=mesh)
    mesh2 = my_mesh.mesh
    # check that vertices are the same
    vertices_mesh = []
    for f in fenics.facets(mesh):
        for v in fenics.vertices(f):
            vertices_mesh.append([v.point().x(), v.point().y()])

    vertices_mesh2 = []
    for f in fenics.facets(mesh2):
        for v in fenics.vertices(f):
            vertices_mesh2.append([v.point().x(), v.point().y()])
    for i in range(0, len(vertices_mesh)):
        assert vertices_mesh[i] == vertices_mesh2[i]


def test_subdomains_inbuilt():
    """
    Test when meshfunctions are given by
    the user
    """
    # create
    mesh = fenics.UnitCubeMesh(6, 6, 6)
    mf_cells = fenics.MeshFunction("size_t", mesh, mesh.topology().dim())
    mf_facets = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    mesh_parameters = {
        "mesh": mesh,
        "volume_markers": mf_cells,
        "surface_markers": mf_facets,
    }
    # read

    my_mesh = Mesh(**mesh_parameters)

    mf_cells_2, mf_facets_2 = my_mesh.volume_markers, my_mesh.surface_markers
    # check
    for cell in fenics.cells(my_mesh.mesh):
        assert mf_cells[cell] == mf_cells_2[cell]
        assert mf_facets[cell] == mf_facets_2[cell]


def test_generate_mesh_from_vertices():
    """
    Test the function generate_mesh_from_vertices
    """
    points = [0, 1, 2, 3, 5]
    my_mesh = MeshFromVertices(points)
    mesh = my_mesh.mesh
    assert mesh.num_vertices() == len(points)
    assert mesh.num_edges() == len(points) - 1
    assert mesh.num_cells() == len(points) - 1


def test_mesh_refinement_course_mesh():
    """
    Test for MeshFromRefinements, when mesh too coarse for refinement
        - return error
    """
    mesh_parameters = {
        "initial_number_of_cells": 2,
        "size": 10,
        "refinements": [
            {"cells": 3, "x": 0.00001},
        ],
    }

    with pytest.raises(
        ValueError,
        match="Infinite loop: Initial number " + "of cells might be too small",
    ):
        MeshFromRefinements(**mesh_parameters)
