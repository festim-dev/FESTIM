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
            "H": {
                "free_enthalpy": 5,
                "entropy": 6
            },
            "id": 1
            },
        {
            "D_0": 3,
            "E_diff": 4,
            "thermal_cond": 5,
            "H": {
                "free_enthalpy": 6,
                "entropy": 7
            },
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
    A, B, C, D, E = \
        FESTIM.post_processing.create_flux_functions(mesh, materials, mf)
    for cell in fenics.cells(mesh):
        cell_no = cell.index()
        assert A.vector()[cell_no] == mf[cell]+1
        assert B.vector()[cell_no] == mf[cell]+2
        assert C.vector()[cell_no] == mf[cell]+3
        assert D.vector()[cell_no] == mf[cell]+4
        assert E.vector()[cell_no] == mf[cell]+5


def test_derived_quantities():
    '''
    Test the function FESTIM.derived_quantities()
    '''
    T = 2*FESTIM.x**2 + 1
    u = 2*FESTIM.x**2
    D = 2*T
    thermal_cond = T**2
    Q = 2*T + 3
    R = 8.314

    # Create Functions
    mesh = fenics.UnitIntervalMesh(10000)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    u_ = fenics.Expression(sp.printing.ccode(u), degree=3)
    u_ = fenics.interpolate(u_, V)
    T_ = fenics.Expression(sp.printing.ccode(T), degree=3)
    T_ = fenics.interpolate(T_, V)

    surface_markers = fenics.MeshFunction("size_t", mesh, 0, 1)
    domain = fenics.CompiledSubDomain('x[0] > 0.99999999')
    domain.mark(surface_markers, 2)

    volume_markers = fenics.MeshFunction("size_t", mesh, 1, 1)
    domain = fenics.CompiledSubDomain('x[0] > 0.75')
    domain.mark(volume_markers, 2)
    # Set parameters for derived quantities
    parameters = {
        "temperature": {},
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

                "minimum_volume": [
                    {
                        "field": 'solute',
                        "volumes": [2]
                    }
                ],
                "maximum_volume": [
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
                "file": "derived_quantities",
                "folder": "",
            }
        }
    }

    # Expected result
    flux_u = D*sp.diff(u, FESTIM.x)
    flux_T = thermal_cond*(sp.diff(T, FESTIM.x))
    average_T_1 = (sp.integrate(T, FESTIM.x).subs(FESTIM.x, 0.75) -
                   sp.integrate(T, FESTIM.x).subs(FESTIM.x, 0))/0.75
    total_u_1 = sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0.75) - \
        sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0)
    total_u_2 = sp.integrate(u, FESTIM.x).subs(FESTIM.x, 1) - \
        sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0.75)
    total_surf_u_2 = u.subs(FESTIM.x, 1)
    max_T_1 = T.subs(FESTIM.x, 0.75)
    min_u_2 = u.subs(FESTIM.x, 0.75)
    expected = [
        flux_u.subs(FESTIM.x, 1),
        flux_T.subs(FESTIM.x, 1),
        average_T_1,
        min_u_2,
        max_T_1,
        total_u_1,
        total_u_2,
        total_surf_u_2]

    # Compute
    D = fenics.Expression(sp.printing.ccode(D), degree=3)
    D = fenics.interpolate(D, V)
    thermal_cond = fenics.Expression(sp.printing.ccode(thermal_cond), degree=3)
    thermal_cond = fenics.interpolate(thermal_cond, V)
    tab = FESTIM.post_processing.derived_quantities(
        parameters, [u_, u_, T_], [D, thermal_cond],
        [volume_markers, surface_markers])

    # Compare
    assert len(tab) == len(expected)
    for i in range(0, len(tab)):
        assert abs(tab[i] - expected[i])/expected[i] < 1e-3


def test_derived_quantities_soret():
    '''
    Test the function FESTIM.derived_quantities()
    with soret effect
    '''
    T = 2*FESTIM.x**2 + 1
    u = 2*FESTIM.x**2
    D = 2*T
    thermal_cond = T**2
    Q = 2*T + 3
    R = 8.314

    # Create Functions
    mesh = fenics.UnitIntervalMesh(10000)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    u_ = fenics.Expression(sp.printing.ccode(u), degree=3)
    u_ = fenics.interpolate(u_, V)
    T_ = fenics.Expression(sp.printing.ccode(T), degree=3)
    T_ = fenics.interpolate(T_, V)

    surface_markers = fenics.MeshFunction("size_t", mesh, 0, 1)
    domain = fenics.CompiledSubDomain('x[0] > 0.99999999')
    domain.mark(surface_markers, 2)

    volume_markers = fenics.MeshFunction("size_t", mesh, 1, 1)
    domain = fenics.CompiledSubDomain('x[0] > 0.75')
    domain.mark(volume_markers, 2)
    # Set parameters for derived quantities
    parameters = {
        "temperature": {
            "soret": True,
        },
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

                "minimum_volume": [
                    {
                        "field": 'solute',
                        "volumes": [2]
                    }
                ],
                "maximum_volume": [
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
                "file": "derived_quantities",
                "folder": "",
            }
        }
    }

    # Expected result
    flux_u = D*(sp.diff(u, FESTIM.x) + Q*u/(R*T**2)*sp.diff(T, FESTIM.x))
    flux_T = thermal_cond*(sp.diff(T, FESTIM.x))
    average_T_1 = (sp.integrate(T, FESTIM.x).subs(FESTIM.x, 0.75) -
                   sp.integrate(T, FESTIM.x).subs(FESTIM.x, 0))/0.75
    total_u_1 = sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0.75) - \
        sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0)
    total_u_2 = sp.integrate(u, FESTIM.x).subs(FESTIM.x, 1) - \
        sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0.75)
    total_surf_u_2 = u.subs(FESTIM.x, 1)
    max_T_1 = T.subs(FESTIM.x, 0.75)
    min_u_2 = u.subs(FESTIM.x, 0.75)
    expected = [
        flux_u.subs(FESTIM.x, 1),
        flux_T.subs(FESTIM.x, 1),
        average_T_1,
        min_u_2,
        max_T_1,
        total_u_1,
        total_u_2,
        total_surf_u_2]

    # Compute
    D = fenics.Expression(sp.printing.ccode(D), degree=3)
    D = fenics.interpolate(D, V)
    thermal_cond = fenics.Expression(sp.printing.ccode(thermal_cond), degree=3)
    thermal_cond = fenics.interpolate(thermal_cond, V)
    Q = fenics.Expression(sp.printing.ccode(Q), degree=3)
    Q = fenics.interpolate(Q, V)
    tab = FESTIM.post_processing.derived_quantities(
        parameters, [u_, u_, T_], [D, thermal_cond, Q],
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


def test_header_derived_quantities_wrong_key():
    # Set parameters for derived quantities
    parameters_quantity = {
        "exports": {
            "derived_quantities": {
                "FOO": [
                    {
                        "field": 'solute',
                        "surfaces": [2]
                    },
                ],
                "file": "derived_quantities",
                "folder": "",
            }
        }
    }
    parameters_field = {
        "exports": {
            "derived_quantities": {
                "surface_flux": [
                    {
                        "field": 'foo',
                        "surfaces": [2]
                    },
                ],
                "file": "derived_quantities",
                "folder": "",
            }
        }
    }
    parameters_surface = {
        "exports": {
            "derived_quantities": {
                "FOO": [
                    {
                        "field": 'solute',
                        "surfaces": [20]
                    },
                ],
                "file": "derived_quantities",
                "folder": "",
            }
        }
    }
    with pytest.raises(ValueError, match=r'quantity'):
        tab = FESTIM.post_processing.header_derived_quantities(
            parameters_quantity)
    with pytest.raises(ValueError, match=r'field'):
        tab = FESTIM.post_processing.header_derived_quantities(
            parameters_field)
