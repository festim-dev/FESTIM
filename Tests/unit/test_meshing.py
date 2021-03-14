# Unit tests meshing
from FESTIM.meshing import mesh_and_refine, subdomains_1D,\
    read_subdomains_from_xdmf, check_borders,\
    generate_mesh_from_vertices
from FESTIM import Simulation
import fenics
import pytest
import sympy as sp
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

    def mesh_parameters(ini, size, refs, pos):
        param = {
            "initial_number_of_cells": ini,
            "size": size,
            "refinements":  []
        }
        for i in range(len(refs)):
            param["refinements"].append({"cells": refs[i], "x": pos[i]})
        return param
    refinements = [[[2, 3], [0.5, 0.25]], [[3, 11], [0.5, 0.25]]]
    for i in range(len(refinements)):
        param = mesh_parameters(2, 1, refinements[i][0], refinements[i][1])
        mesh = mesh_and_refine(param)

        mf1 = fenics.MeshFunction('size_t', mesh, 1)
        mf2 = fenics.MeshFunction('size_t', mesh, 1)
        subdomain1 = create_subdomains(0, refinements[i][1][1])
        subdomain1.mark(mf1, 1)
        subdomain2 = create_subdomains(0, refinements[i][1][0])
        subdomain2.mark(mf2, 2)
        nb_cell_1 = 0
        nb_cell_2 = 0
        for cell in fenics.cells(mesh):
            cell_no = cell.index()
            if mf1.array()[cell_no] == 1:
                nb_cell_1 += 1
            if mf2.array()[cell_no] == 2:
                nb_cell_2 += 1
        assert nb_cell_1 >= refinements[i][0][1]
        assert nb_cell_2 >= refinements[i][0][0]


def test_subdomains_1D():
    '''
    Test that subdomains are assigned properly
    '''
    mesh = fenics.UnitIntervalMesh(20)

    materials = [
        {
            "borders": [0, 0.5],
            "id": 1,
            },
        {
            "borders": [0.5, 1],
            "id": 2,
            }
            ]
    volume_markers, surface_markers = subdomains_1D(mesh, materials, 1)
    for cell in fenics.cells(mesh):
        if cell.midpoint().x() < 0.5:
            assert volume_markers[cell] == 1
        else:
            assert volume_markers[cell] == 2


def test_check_borders():
    materials = [
        {
            "borders": [0.5, 0.7],
            "id": 1,
            },
        {
            "borders": [0, 0.5],
            "id": 2,
            }
            ]
    size = 0.7
    assert check_borders(size, materials) is True

    with pytest.raises(ValueError, match=r'zero'):
        size = 0.7
        materials = [
            {
                "borders": [0.5, 0.7],
                "id": 1,
                },
            {
                "borders": [0.2, 0.5],
                "id": 2,
                }
                ]
        check_borders(size, materials)

    with pytest.raises(ValueError, match=r'each other'):
        materials = [
            {
                "borders": [0.5, 1],
                "id": 1,
                },
            {
                "borders": [0, 0.6],
                "id": 2,
                },
            {
                "borders": [0.6, 1],
                "id": 3,
                }
                ]
        size = 1
        check_borders(size, materials)

    with pytest.raises(ValueError, match=r'size'):
        materials = [
            {
                "borders": [0, 1],
                "id": 1,
                }
                ]
        size = 3
        check_borders(size, materials)


def test_create_mesh_xdmf(tmpdir):

    # write xdmf file
    mesh = fenics.UnitSquareMesh(10, 10)
    file1 = tmpdir.join("mesh.xdmf")
    f = fenics.XDMFFile(str(Path(file1)))
    f.write(mesh)

    # read mesh
    mesh_parameters = {
        "mesh_file": str(Path(file1)),
        }
    my_model = Simulation({"mesh_parameters": mesh_parameters})
    my_model.define_mesh()
    mesh2 = my_model.mesh

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
    mf_cells_2, mf_facets_2 = read_subdomains_from_xdmf(
        mesh, str(Path(file1)), str(Path(file2)))

    # check
    for cell in fenics.cells(mesh):
        assert mf_cells[cell] == mf_cells_2[cell]
        assert mf_facets[cell] == mf_facets_2[cell]


def test_subdomains_from_xdmf_with_non_default_attribute_name(tmpdir):

    # create mesh functions
    mesh = fenics.UnitCubeMesh(6, 6, 6)
    mf_cells = fenics.MeshFunction("size_t", mesh, mesh.topology().dim())
    mf_facets = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    mf_cells.rename("a", "a")
    mf_facets.rename("a", "a")

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
    mf_cells_2, mf_facets_2 = read_subdomains_from_xdmf(
        mesh, str(Path(file1)), str(Path(file2)))

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

    # read mesh
    mesh_parameters = {
        "mesh": mesh
        }
    my_model = Simulation({"mesh_parameters": mesh_parameters})
    my_model.define_mesh()
    mesh2 = my_model.mesh

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
            "meshfunction_cells": mf_cells,
            "meshfunction_facets": mf_facets
    }
    # read

    my_model = Simulation({"mesh_parameters": mesh_parameters})
    my_model.mesh = mesh
    my_model.define_markers()

    mf_cells_2, mf_facets_2 = \
        my_model.volume_markers, my_model.surface_markers
    # check
    for cell in fenics.cells(mesh):
        assert mf_cells[cell] == mf_cells_2[cell]
        assert mf_facets[cell] == mf_facets_2[cell]


def test_generate_mesh_from_vertices():
    '''
    Test the function generate_mesh_from_vertices
    '''
    points = [0, 1, 2, 3, 5]
    mesh = generate_mesh_from_vertices(points)
    assert mesh.num_vertices() == len(points)
    assert mesh.num_edges() == len(points) - 1
    assert mesh.num_cells() == len(points) - 1


def test_create_mesh_vertices():
    '''
    Test the function create_mesh with vertices key
    '''
    points = [0, 1, 2, 5, 12, 24]
    mesh_parameters = {
        "vertices": points
    }
    my_model = Simulation({"mesh_parameters": mesh_parameters})
    my_model.define_mesh()
    mesh = my_model.mesh
    assert mesh.num_vertices() == len(points)
    assert mesh.num_edges() == len(points) - 1
    assert mesh.num_cells() == len(points) - 1
    for cell in fenics.cells(mesh):
        for v in fenics.vertices(cell):
            assert v.point().x() in points


def test_integration_mesh_from_vertices_subdomains():
    '''
    Integration test for meshing and subdomain 1D
    when parsing a list of vertices
    Checks that the cells are marked correctly
    '''
    points = [0, 1, 2, 5, 12, 24]
    mesh_parameters = {
        "vertices": points
    }
    materials = [
        {
            "borders": [0, 2],
            "id": 1
        },
        {
            "borders": [2, 24],
            "id": 2
        }
    ]
    parameters = {
        "mesh_parameters": mesh_parameters,
        "materials": materials
    }
    my_model = Simulation(parameters)
    my_model.define_mesh()
    mesh = my_model.mesh
    my_model.define_markers()
    vm, sm = my_model.volume_markers, my_model.surface_markers

    # Testing
    for cell in fenics.cells(mesh):
        if cell.midpoint().x() < 2:
            assert vm[cell] == 1
        elif cell.midpoint().x() > 2:
            assert vm[cell] == 2
    for facet in fenics.facets(mesh):
        if facet.midpoint().x() == 0:
            assert sm[facet] == 1
        if facet.midpoint().x() == max(points):
            assert sm[facet] == 2
