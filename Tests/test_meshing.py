# Unit tests meshing
from FESTIM import meshing
import fenics
import pytest
import sympy as sp


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
        mesh = meshing.mesh_and_refine(param)

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
    volume_markers, surface_markers = meshing.subdomains_1D(mesh, materials, 1)
    for cell in fenics.cells(mesh):
        if cell.midpoint().x() < 0.5:
            assert volume_markers[cell] == 1
        else:
            assert volume_markers[cell] == 2


def test_fail_subdomains_1D_difference_size_borders():
    '''
    Test that an error is raised if the borders don't match
    the size of the mesh
    '''
    mesh = fenics.UnitIntervalMesh(20)

    materials = [
        {
            "borders": [0, 0.5],
            "id": 1,
            },
        {
            "borders": [0.5, 0.7],
            "id": 2,
            }
            ]
    with pytest.raises(ValueError, match=r'match'):
        meshing.subdomains_1D(mesh, materials, 1)
