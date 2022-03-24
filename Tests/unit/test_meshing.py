# Unit tests meshing
from FESTIM import Materials, Material
from FESTIM import Mesh, Mesh1D, MeshFromRefinements, MeshFromVertices, \
    MeshFromXDMF
import fenics
import pytest
from pathlib import Path


def test_mesh_and_refine_meets_refinement_conditions():
    '''
    Test that function mesh_and_refine() gives the right
    refinement conditions
    '''
    def create_subdomains(x1, x2):
        class domain(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= x1 and x[0] <= x2
        domain = domain()
        return domain

    refinements = [
            [{"cells": 2, "x": 0.5}, {"cells": 3, "x": 0.25}],
            [{"cells": 3, "x": 0.5}, {"cells": 11, "x": 0.25}]
            ]
    for refinement in refinements:
        my_mesh = MeshFromRefinements(initial_number_of_cells=2, size=1, refinements=refinement)
        mesh = my_mesh.mesh

        mf1 = fenics.MeshFunction('size_t', mesh, 1)
        mf2 = fenics.MeshFunction('size_t', mesh, 1)
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


def test_subdomains_1D():
    '''
    Test that subdomains are assigned properly
    '''
    mesh = fenics.UnitIntervalMesh(20)

    materials = [
        Material(id=1, D_0=None, E_D=None, borders=[0, 0.5]),
        Material(id=2, D_0=None, E_D=None, borders=[0.5, 1]),
        ]
    my_mats = Materials(materials)
    my_mesh = Mesh1D()
    my_mesh.mesh = mesh
    my_mesh.size = 1
    my_mesh.define_markers(my_mats)
    for cell in fenics.cells(mesh):
        if cell.midpoint().x() < 0.5:
            assert my_mesh.volume_markers[cell] == 1
        else:
            assert my_mesh.volume_markers[cell] == 2


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
            vertices_mesh.append(
                [v.point().x(), v.point().y()])

    vertices_mesh2 = []
    for f in fenics.facets(my_mesh.mesh):
        for v in fenics.vertices(f):
            vertices_mesh2.append(
                [v.point().x(), v.point().y()])
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
    '''
    Test when mesh is given by the user
    '''
    # create mesh
    mesh = fenics.UnitSquareMesh(10, 10)

    my_mesh = Mesh(mesh=mesh)
    mesh2 = my_mesh.mesh
    # check that vertices are the same
    vertices_mesh = []
    for f in fenics.facets(mesh):
        for v in fenics.vertices(f):
            vertices_mesh.append(
                [v.point().x(), v.point().y()])

    vertices_mesh2 = []
    for f in fenics.facets(mesh2):
        for v in fenics.vertices(f):
            vertices_mesh2.append(
                [v.point().x(), v.point().y()])
    for i in range(0, len(vertices_mesh)):
        assert vertices_mesh[i] == vertices_mesh2[i]


def test_subdomains_inbuilt():
    '''
    Test when meshfunctions are given by
    the user
    '''
    # create
    mesh = fenics.UnitCubeMesh(6, 6, 6)
    mf_cells = fenics.MeshFunction("size_t", mesh, mesh.topology().dim())
    mf_facets = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    mesh_parameters = {
            "mesh": mesh,
            "volume_markers": mf_cells,
            "surface_markers": mf_facets
    }
    # read

    my_mesh = Mesh(**mesh_parameters)

    mf_cells_2, mf_facets_2 = \
        my_mesh.volume_markers, my_mesh.surface_markers
    # check
    for cell in fenics.cells(my_mesh.mesh):
        assert mf_cells[cell] == mf_cells_2[cell]
        assert mf_facets[cell] == mf_facets_2[cell]


def test_generate_mesh_from_vertices():
    '''
    Test the function generate_mesh_from_vertices
    '''
    points = [0, 1, 2, 3, 5]
    my_mesh = MeshFromVertices(points)
    mesh = my_mesh.mesh
    assert mesh.num_vertices() == len(points)
    assert mesh.num_edges() == len(points) - 1
    assert mesh.num_cells() == len(points) - 1


def test_mesh_refinement_course_mesh():
    '''
    Test for MeshFromRefinements, when mesh too coarse for refinement
        - return error
    '''
    mesh_parameters = {
        "initial_number_of_cells": 2,
        "size": 10,
        "refinements": [
            {
                "cells": 3,
                "x": 0.00001
            },
        ],
    }

    with pytest.raises(ValueError,
                       match="Infinite loop: Initial number " +
                             "of cells might be too small"):
        MeshFromRefinements(**mesh_parameters)
