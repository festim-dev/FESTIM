import FESTIM
from FESTIM.initialising import initialise_solutions
import pytest
import fenics
import os
from pathlib import Path


def test_initialisation_from_xdmf(tmpdir):
    mesh = fenics.UnitSquareMesh(5, 5)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(0).collapse())
    fenics.assign(u.sub(0), ini_u)
    ini_u = fenics.Expression("1", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(1).collapse())
    fenics.assign(u.sub(1), ini_u)

    d = tmpdir.mkdir("Initial solutions")
    file1 = d.join("u_1out.xdmf")
    file2 = d.join("u_2out.xdmf")
    print(Path(file1))
    with fenics.XDMFFile(str(Path(file1))) as file:
        file.write_checkpoint(u.sub(0), "1", 2, fenics.XDMFFile.Encoding.HDF5,
                              append=False)
    with fenics.XDMFFile(str(Path(file2))) as file:
        file.write_checkpoint(u.sub(1), "2", 2, fenics.XDMFFile.Encoding.HDF5,
                              append=False)
        file.write_checkpoint(u.sub(1), "2", 4, fenics.XDMFFile.Encoding.HDF5,
                              append=True)

    parameters = {
        "initial_conditions": [
            {
                "value": str(Path(file1)),
                "component": 0,
                "label": "1",
                "time_step": 0
            },
            {
                "value": str(Path(file2)),
                "component": 1,
                "label": "2",
                "time_step": 1
            },
        ],
    }
    w = initialise_solutions(parameters, V)
    assert fenics.errornorm(u, w) == 0


def test_fail_initialisation_from_xdmf():
    '''
    Test that the function fails initialise_solutions if
    there's a missing key
    '''
    mesh = fenics.UnitSquareMesh(5, 5)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)

    parameters = {
        "initial_conditions": [
            {
                "value": "Initial solutions/u_1out.xdmf",
                "component": 0,
                "label": "1",
            },
            {
                "value": "Initial solutions/u_2out.xdmf",
                "component": 1,
                "label": "2",
                "time_step": 1
            },
        ],
    }
    with pytest.raises(KeyError, match=r'time_step'):
        initialise_solutions(parameters, V)

    parameters = {
        "initial_conditions": [
            {
                "value": "Initial solutions/u_1out.xdmf",
                "component": 0,
                "time_step": 1
            },
            {
                "value": "Initial solutions/u_2out.xdmf",
                "component": 1,
                "time_step": 1
            },
        ],
    }
    with pytest.raises(KeyError, match=r'label'):
        initialise_solutions(parameters, V)


def test_initialisation_with_expression():
    '''
    Test that initialise_solutions interpolates correctly
    from an expression
    '''
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1+x[0] + x[1]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(0).collapse())
    fenics.assign(u.sub(0), ini_u)
    ini_u = fenics.Expression("1+x[0]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(1).collapse())
    fenics.assign(u.sub(1), ini_u)

    parameters = {
        "initial_conditions": [
            {
                "value": 1+FESTIM.x + FESTIM.y,
                "component": 0,
            },
            {
                "value": 1+FESTIM.x,
                "component": 1,
            },
        ],
    }
    w = initialise_solutions(parameters, V)
    assert fenics.errornorm(u, w) == 0


def test_initialisation_with_expression_chemical_pot():
    '''
    Test that initialise_solutions interpolates correctly
    from an expression with conservation of chemical potential
    '''
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1+x[0] + x[1]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(0).collapse())
    fenics.assign(u.sub(0), ini_u)
    ini_u = fenics.Expression("1+x[0]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(1).collapse())
    fenics.assign(u.sub(1), ini_u)

    parameters = {
        "initial_conditions": [
            {
                "value": 1+FESTIM.x + FESTIM.y,
                "component": 0,
            },
            {
                "value": 1+FESTIM.x,
                "component": 1,
            },
        ],
    }
    w = initialise_solutions(parameters, V)
    assert fenics.errornorm(u, w) == 0


def test_initialisation_default():
    '''
    Test that initialise_solutions interpolates correctly
    if nothing is given (default is 0)
    '''
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    u = fenics.Function(V)
    w = fenics.Function(V)

    w = initialise_solutions({"initial_conditions": []}, V)
    assert fenics.errornorm(u, w) == 0


def test_initialisation_solute_only():
    '''
    Test that initialise_solutions interpolates correctly
    if solution has only 1 component (ie solute)
    '''
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1 + x[0] + x[1]", degree=1)
    u = fenics.interpolate(ini_u, V)
    parameters = {
        "initial_conditions": [
            {
                "value": 1+FESTIM.x + FESTIM.y,
                "component": 0
            },
        ],
    }
    w = initialise_solutions(parameters, V)
    assert fenics.errornorm(u, w) == 0


def test_initialisation_no_component():
    '''
    Test that initialise_solutions set component at 0
    by default
    '''
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 3)
    u = fenics.Function(V)
    w = fenics.Function(V)
    ini_u = fenics.Expression("1 + x[0] + x[1]", degree=1)
    ini_u = fenics.interpolate(ini_u, V.sub(0).collapse())
    fenics.assign(u.sub(0), ini_u)

    parameters = {
        "initial_conditions": [
            {
                "value": 1+FESTIM.x + FESTIM.y,
            },
        ],
    }
    w = initialise_solutions(parameters, V)
    assert fenics.errornorm(u, w) == 0


def test_initialisation_duplicates():
    '''
    Test that initialise_solutions set component at 0
    by default
    '''
    mesh = fenics.UnitSquareMesh(8, 8)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 3)

    parameters = {
        "initial_conditions": [
            {
                "value": 1+FESTIM.x + FESTIM.y,
                "component": 0
            },
            {
                "value": 1+FESTIM.x + FESTIM.y,
                "component": 0
            },
        ],
    }
    with pytest.raises(ValueError, match=r'Duplicate'):
        initialise_solutions(parameters, V)
