import FESTIM
import fenics
import pytest
import sympy as sp
from pathlib import Path


def test_define_xdmf_files():
    folder = "Solution"
    expected = [fenics.XDMFFile(folder + "/" + "a.xdmf"),
                fenics.XDMFFile(folder + "/" + "b.xdmf")]
    exports = {
        "xdmf": {
            "functions": ['solute', '1'],
            "labels":  ['a', 'b'],
            "folder": folder
        }
        }
    assert len(expected) == len(FESTIM.export.define_xdmf_files(exports))

    # Test an int type for folder
    with pytest.raises(TypeError, match=r'str'):
        folder = 123
        exports = {
            "xdmf": {
                "functions": ['solute', '1'],
                "labels":  ['a', 'b'],
                "folder": folder
            }
            }
        FESTIM.export.define_xdmf_files(exports)

    # Test an empty string for folder
    with pytest.raises(ValueError, match=r'empty string'):
        folder = ''
        exports = {
            "xdmf": {
                "functions": ['solute', '1'],
                "labels":  ['a', 'b'],
                "folder": folder
            }
            }
        FESTIM.export.define_xdmf_files(exports)


def test_export_xdmf(tmpdir):
    mesh = fenics.UnitSquareMesh(3, 3)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    d = tmpdir.mkdir("Solution_Test")
    exports = {
        "xdmf": {
            "functions": ['solute', 'retention'],
            "labels":  ['a', 'b'],
            "folder": str(Path(d))
        }
        }
    files = [fenics.XDMFFile(str(Path(d.join("a.xdmf")))),
             fenics.XDMFFile(str(Path(d.join("b.xdmf"))))]

    assert FESTIM.export.export_xdmf([fenics.Function(V), fenics.Function(V)],
                                     exports, files, 20, append=True) is None

    exports["xdmf"]["functions"] = ['solute', 'blabla']

    with pytest.raises(KeyError, match=r'blabla'):
        FESTIM.export.export_xdmf(
            [fenics.Function(V), fenics.Function(V)],
            exports, files, 20, append=True)

    exports["xdmf"]["functions"] = ['solute', '13']
    with pytest.raises(KeyError, match=r'13'):
        FESTIM.export.export_xdmf(
            [fenics.Function(V), fenics.Function(V)],
            exports, files, 20, append=True)


def test_create_flux_functions():
    '''
    Test the function FESTIM.create_flux_functions()
    '''
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    materials = [
        {
            "D_0": 2,
            "E_diff": 3,
            "thermal_cond": 4,
            "id": 1
            },
        {
            "D_0": 3,
            "E_diff": 4,
            "thermal_cond": 5,
            "id": 2
            }
            ]
    mf = fenics.MeshFunction("size_t", mesh, 1, 0)
    for cell in fenics.cells(mesh):
        x = cell.midpoint().x()
        if x < 0.5:
            mf[cell] = 1
        else:
            mf[cell] = 2
    A, B, C = FESTIM.post_processing.create_flux_functions(mesh, materials, mf)
    for cell in fenics.cells(mesh):
        cell_no = cell.index()
        assert A.vector()[cell_no] == mf[cell]+1
        assert B.vector()[cell_no] == mf[cell]+2
        assert C.vector()[cell_no] == mf[cell]+3


def test_derived_quantities():
    '''
    Test the function FESTIM.derived_quantities()
    '''
    # Create Functions
    mesh = fenics.UnitIntervalMesh(10000)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.Expression("2*x[0]*x[0]", degree=3)
    u = fenics.interpolate(u, V)
    T = fenics.Expression("2*x[0]*x[0] + 1", degree=3)
    T = fenics.interpolate(T, V)

    surface_markers = fenics.MeshFunction("size_t", mesh, 0, 1)
    domain = fenics.CompiledSubDomain('x[0] > 0.99999999')
    domain.mark(surface_markers, 2)

    volume_markers = fenics.MeshFunction("size_t", mesh, 1, 1)
    domain = fenics.CompiledSubDomain('x[0] > 0.75')
    domain.mark(volume_markers, 2)
    # Set parameters for derived quantities
    parameters = {
        "exports": {
            "derived_quantities": {
                "surface_flux": [
                    {
                        "field": 'solute',
                        "surfaces": [2]
                    },
                    {
                        "field": 'T',
                        "surfaces": [2]
                    },
                ],
                "average_volume": [
                    {
                        "field": 'T',
                        "volumes": [1]
                    }
                ],
                "total_volume": [
                    {
                        "field": 'solute',
                        "volumes": [1, 2]
                    }
                ],
                "total_surface": [
                    {
                        "field": 'solute',
                        "surfaces": [2]
                    }
                ],
                "maximum_volume": [
                    {
                        "field": 'T',
                        "volumes": [1]
                    }
                ],
                "minimum_volume": [
                    {
                        "field": 'solute',
                        "volumes": [2]
                    }
                ],
                "file": "derived_quantities",
                "folder": "",
            }
        }
    }
    # Expected result
    expected = [4, 4, 11/8, 9/8, 17/8, 9/32, 37/96, 2]
    # Compute
    tab = FESTIM.post_processing.derived_quantities(
        parameters, [u, u, T], [1, 1],
        [volume_markers, surface_markers])
    # Compare
    assert len(tab) == len(expected)
    for i in range(0, len(tab)):
        assert abs(tab[i] - expected[i])/expected[i] < 1e-3


def test_header_derived_quantities():
    # Set parameters for derived quantities
    parameters = {
        "exports": {
            "derived_quantities": {
                "surface_flux": [
                    {
                        "field": 'solute',
                        "surfaces": [2]
                    },
                    {
                        "field": 'T',
                        "surfaces": [2]
                    },
                ],
                "average_volume": [
                    {
                        "field": 'T',
                        "volumes": [1]
                    }
                ],
                "total_volume": [
                    {
                        "field": 'solute',
                        "volumes": [1, 2]
                    }
                ],
                "total_surface": [
                    {
                        "field": 'solute',
                        "surfaces": [2]
                    }
                ],
                "maximum_volume": [
                    {
                        "field": 'T',
                        "volumes": [1]
                    }
                ],
                "minimum_volume": [
                    {
                        "field": 'solute',
                        "volumes": [2]
                    }
                ],
                "file": "derived_quantities",
                "folder": "",
            }
        }
    }

    tab = FESTIM.post_processing.header_derived_quantities(parameters)
    expected = ["t(s)",
                "Flux surface 2: solute", "Flux surface 2: T",
                "Average T volume 1",
                "Minimum solute volume 2", "Maximum T volume 1",
                "Total solute volume 1", "Total solute volume 2",
                "Total solute surface 2"]
    assert len(tab) == len(expected)
    for i in range(0, len(tab)):
        assert tab[i] == expected[i]
