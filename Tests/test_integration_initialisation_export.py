import FESTIM
from FESTIM import export
from FESTIM.initialising import initialise_solutions
import fenics
import pytest
import sympy as sp
from pathlib import Path


def test_export_and_initialise_xdmf(tmpdir):
    '''
    Test if an exported file can be read as initial condition
    '''
    # Write
    mesh = fenics.UnitSquareMesh(3, 3)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    ini_u = fenics.Expression("0", degree=1)
    u = fenics.interpolate(ini_u, V)

    exports = {
        "xdmf": {
            "functions": ['solute'],
            "labels":  ['1'],
            "folder": "Solution"
        }
        }

    d = tmpdir.mkdir("Initial solutions")
    file1 = d.join("u_1out.xdmf")
    files = [fenics.XDMFFile(str(Path(file1)))]
    export.export_xdmf(
        [u],
        exports, files, 20, append=False)

    #  Read
    parameters = {
        "initial_conditions": [
            {
                "value": str(Path(file1)),
                "component": 0,
                "label": "1",
                "time_step": 0
            },
        ],
    }
    assert initialise_solutions(parameters, V)


def test_initialise_and_export_xdmf(tmpdir):
    '''
    Test if an initialised solution can be exported
    '''
    mesh = fenics.UnitSquareMesh(5, 5)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.Function(V)
    ini_u = fenics.Expression("1", degree=1)
    u = fenics.interpolate(ini_u, V)

    d = tmpdir.mkdir("Initial solutions")
    file1 = d.join("u_1in.xdmf")
    with fenics.XDMFFile(str(Path(file1))) as file:
        file.write_checkpoint(u, "1", 2, fenics.XDMFFile.Encoding.HDF5,
                              append=False)
    # Read
    parameters = {
        "initial_conditions": [
            {
                "value": str(Path(file1)),
                "component": 0,
                "label": "1",
                "time_step": 0
            },
        ],
    }
    v = initialise_solutions(parameters, V)

    # Write
    exports = {
        "xdmf": {
            "functions": ['solute'],
            "labels":  ['1'],
            "folder": "Solution"
        }
        }

    d2 = tmpdir.mkdir("Output")
    file2 = d.join("u_1out.xdmf")
    files = [fenics.XDMFFile(str(Path(file2)))]
    assert export.export_xdmf(
        [v],
        exports, files, 20, append=False) is None
