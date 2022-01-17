import FESTIM
from FESTIM.export import define_xdmf_files, export_profiles, export_xdmf
from FESTIM.post_processing import run_post_processing, derived_quantities, \
    create_properties
import fenics
import pytest
import sympy as sp
import numpy as np
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
    assert len(expected) == len(define_xdmf_files(exports))

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
        define_xdmf_files(exports)

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
        define_xdmf_files(exports)


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

    assert export_xdmf([fenics.Function(V), fenics.Function(V)],
                       exports, files, 20, append=True) is None

    exports["xdmf"]["functions"] = ['solute', 'foo']

    with pytest.raises(ValueError, match=r'foo'):
        export_xdmf(
            [fenics.Function(V), fenics.Function(V)],
            exports, files, 20, append=True)

    exports["xdmf"]["functions"] = ['solute', '13']
    with pytest.raises(ValueError, match=r'13'):
        export_xdmf(
            [fenics.Function(V), fenics.Function(V)],
            exports, files, 20, append=True)


def test_export_profiles(tmpdir):
    '''
    Test the function export.export_profiles()
    '''
    mesh = fenics.UnitIntervalMesh(3)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    d = tmpdir.mkdir("Solution_Test")
    functions = [fenics.Function(V), fenics.Function(V)]
    exports = {
        "txt": {
            "functions": ['1', 'retention'],
            "times": [2.5, 1, 3.2],
            "labels":  ['a', 'b'],
            "folder": str(Path(d))
        }
        }

    t = 0
    dt = fenics.Constant(2)
    while t < 4:
        dt_old = dt
        dt = export_profiles(functions, exports, t, dt, V)
        # Test that dt is not changed if not on time
        if np.allclose(t, exports["txt"]["times"]):
            assert np.isclose(float(dt_old), float(dt)) 
        # Test that dt has the right value
        elif t < 2:
            assert np.isclose(float(dt), 1) 
        elif t < 3:
            assert np.isclose(float(dt), 0.5) 
        else:
            assert np.isclose(float(dt), 0.2) 
        t += float(dt)

    # Test that a ValueError is raised if wrong function
    t = 1
    exports["txt"]["functions"][0] = "foo"
    with pytest.raises(ValueError):
        export_profiles(functions, exports, t, dt, V)


def test_export_profiles_with_vectors(tmpdir):
    '''
    Test the function export.export_profiles() with a vector space
    '''
    mesh = fenics.UnitIntervalMesh(3)
    solute = fenics.FiniteElement('CG', mesh.ufl_cell(), 1)
    traps = fenics.FiniteElement('DG', mesh.ufl_cell(), 1)
    element = [solute] + [traps]*1
    V = fenics.FunctionSpace(mesh, fenics.MixedElement(element))
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)

    d = tmpdir.mkdir("Solution_Test")
    u = fenics.Function(V)
    functions = list(u.split())
    exports = {
        "txt": {
            "functions": ['1', 'solute'],
            "times": [2.5, 1, 3.2],
            "labels":  ['a', 'b'],
            "folder": str(Path(d))
        }
        }

    t = 0
    dt = fenics.Constant(2)
    while t < 4:
        dt_old = dt
        dt = export_profiles(functions, exports, t, dt, V_DG1)
        # Test that dt is not changed if not on time
        if np.allclose(t, exports["txt"]["times"]):
            assert np.isclose(float(dt_old), float(dt)) 
        # Test that dt has the right value
        elif t < 2:
            assert np.isclose(float(dt), 1) 
        elif t < 3:
            assert np.isclose(float(dt), 0.5)
        else:
            assert np.isclose(float(dt), 0.2) 
        t += float(dt)

    # Test that a ValueError is raised if wrong function
    t = 1
    exports["txt"]["functions"][0] = "foo"
    with pytest.raises(ValueError):
        export_profiles(functions, exports, t, dt, V)


def test_create_properties():
    '''
    Test the function FESTIM.create_properties()
    '''
    mesh = fenics.UnitIntervalMesh(10)
    DG_1 = fenics.FunctionSpace(mesh, 'DG', 1)
    mat_1 = FESTIM.Material(1, D_0=1, E_D=0, S_0=7, E_S=0, thermal_cond=4, heat_capacity=5, rho=6, H={"free_enthalpy": 5, "entropy": 6})
    mat_2 = FESTIM.Material(2, D_0=2, E_D=0, S_0=8, E_S=0, thermal_cond=5, heat_capacity=6, rho=7, H={"free_enthalpy": 6, "entropy": 6})
    materials = FESTIM.Materials([mat_1, mat_2])
    mf = fenics.MeshFunction("size_t", mesh, 1, 0)
    for cell in fenics.cells(mesh):
        x = cell.midpoint().x()
        if x < 0.5:
            mf[cell] = 1
        else:
            mf[cell] = 2
    T = fenics.Expression("1", degree=1)
    D, thermal_cond, cp, rho, H, S = \
        create_properties(mesh, materials, mf, T)
    D = fenics.interpolate(D, DG_1)
    thermal_cond = fenics.interpolate(thermal_cond, DG_1)
    cp = fenics.interpolate(cp, DG_1)
    rho = fenics.interpolate(rho, DG_1)
    H = fenics.interpolate(H, DG_1)
    S = fenics.interpolate(S, DG_1)

    for cell in fenics.cells(mesh):
        assert D(cell.midpoint().x()) == mf[cell]
        assert thermal_cond(cell.midpoint().x()) == mf[cell] + 3
        assert cp(cell.midpoint().x()) == mf[cell] + 4
        assert rho(cell.midpoint().x()) == mf[cell] + 5
        assert H(cell.midpoint().x()) == mf[cell] + 10
        assert S(cell.midpoint().x()) == mf[cell] + 6


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
                        "field": 'retention',
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
    tab = derived_quantities(
        parameters, [u_, u_, T_], [volume_markers, surface_markers],
        [D, thermal_cond, None, None])

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
    tab = derived_quantities(
        parameters, [u_, u_, T_], [volume_markers, surface_markers],
        [D, thermal_cond, Q, None])

    # Compare
    assert len(tab) == len(expected)
    for i in range(0, len(tab)):
        assert abs(tab[i] - expected[i])/expected[i] < 1e-3


def test_derived_quantities_chemical_pot():
    '''
    Test the function FESTIM.derived_quantities()
    with soret effect
    '''
    T = 2*FESTIM.x**2 + 1
    u = 2*FESTIM.x**2
    D = 2*T
    thermal_cond = T**2
    S = 2 + T
    R = 8.314
    theta = u/S
    # Create Functions
    mesh = fenics.UnitIntervalMesh(10000)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    u_ = fenics.Expression(sp.printing.ccode(u), degree=3)
    u_ = fenics.interpolate(u_, V)
    theta_ = fenics.Expression(sp.printing.ccode(theta), degree=3)
    theta_ = fenics.interpolate(theta_, V)
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
        },
        "exports": {
            "derived_quantities": {
                "surface_flux": [
                    {
                        "field": 'solute',
                        "surfaces": [2]
                    },
                ],
                "minimum_volume": [
                    {
                        "field": 'solute',
                        "volumes": [2]
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
    flux_u = D*(sp.diff(u, FESTIM.x))
    total_u_1 = sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0.75) - \
        sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0)
    total_u_2 = sp.integrate(u, FESTIM.x).subs(FESTIM.x, 1) - \
        sp.integrate(u, FESTIM.x).subs(FESTIM.x, 0.75)
    total_surf_u_2 = u.subs(FESTIM.x, 1)
    min_u_2 = u.subs(FESTIM.x, 0.75)
    expected = [
        flux_u.subs(FESTIM.x, 1),
        min_u_2,
        total_u_1,
        total_u_2,
        total_surf_u_2]

    # Compute
    D = fenics.Expression(sp.printing.ccode(D), degree=3)
    D = fenics.interpolate(D, V)
    thermal_cond = fenics.Expression(sp.printing.ccode(thermal_cond), degree=3)
    thermal_cond = fenics.interpolate(thermal_cond, V)
    S = fenics.Expression(sp.printing.ccode(S), degree=3)
    S = fenics.interpolate(S, V_DG1)
    tab = derived_quantities(
        parameters, [fenics.project(S*theta_, V_DG1), theta_, T_],
        [volume_markers, surface_markers],
        [D, thermal_cond, None])

    # Compare
    assert len(tab) == len(expected)
    for i in range(0, len(tab)):
        print(i)
        assert abs(tab[i] - expected[i])/expected[i] < 1e-3


def test_run_post_processing_export_xdmf_chemical_pot(tmpdir):
    """this test checks that the computation of retention is correctly made
    with conservation of chemical pot
    """
    # build
    d = tmpdir.mkdir("out")
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    val_theta = 2
    theta_out = fenics.interpolate(fenics.Constant(val_theta), V)
    files = [
        fenics.XDMFFile(str(Path(d)) + "/retention.xdmf"),
        fenics.XDMFFile(str(Path(d)) + "/solute.xdmf")
    ]
    exports = {
            "xdmf": {
                "functions": ['retention', 'solute'],
                "labels": ['retention', 'solute'],
                "folder": str(Path(d))
            }
        }
    val_S = 3  # solubility
    S = fenics.Constant(val_S)

    parameters = {
        "exports": exports,
        "temperature": {
            "type": "expression",
            "value": 500}
    }

    # run
    my_sim = FESTIM.Simulation(parameters)
    my_sim.transient = True
    my_sim.u = theta_out
    my_temp = FESTIM.Temperature("expression", 500)
    my_temp.create_functions(V, FESTIM.Materials(), None, None, None)
    my_sim.T = my_temp
    my_sim.volume_markers, my_sim.surface_markers = None, None
    my_sim.V_CG1, my_sim.V_DG1 = V, V_DG1
    my_sim.t = 0
    my_sim.dt = 1
    my_sim.files = files
    my_sim.append = False
    my_sim.D, my_sim.thermal_cond, my_sim.cp, my_sim.rho, \
        my_sim.H, my_sim.S = *[None]*5, S
    my_sim.derived_quantities_global = []
    my_sim.chemical_pot = True
    run_post_processing(my_sim)

    # check
    u_in = fenics.Function(V)
    files[0].read_checkpoint(u_in, "retention", -1)
    for i in range(10):
        assert np.isclose(u_in(i/10), val_theta*val_S)

    files[1].read_checkpoint(u_in, "solute", -1)
    for i in range(10):
        assert np.isclose(u_in(i/10), val_theta*val_S)


def test_run_post_processing_export_xdmf_chemical_pot(tmpdir):
    """this test checks that the computation of retention is correctly made
    with conservation of chemical pot and temperature of type "solve_transient"
    """
    # build
    d = tmpdir.mkdir("out")
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    V_DG1 = fenics.FunctionSpace(mesh, 'DG', 1)
    val_theta = 2
    theta_out = fenics.interpolate(fenics.Constant(val_theta), V)
    files = [
        fenics.XDMFFile(str(Path(d)) + "/retention.xdmf"),
        fenics.XDMFFile(str(Path(d)) + "/solute.xdmf")
    ]
    exports = {
            "xdmf": {
                "functions": ['retention', 'solute'],
                "labels": ['retention', 'solute'],
                "folder": str(Path(d))
            }
        }
    val_S = 3  # solubility
    S = fenics.Constant(val_S)

    parameters = {
        "exports": exports,
        "boundary_conditions": [],
        "temperature": {
            "type": "solve_transient",
            "boundary_conditions": [
                {
                    "type": "dc",
                    "value": 500,
                    "surfaces": [1, 2]
                }
            ],
            "initial_condition": 500
            }
    }

    # run
    my_sim = FESTIM.Simulation(parameters)
    my_sim.transient = True
    my_sim.u = theta_out
    my_temp = FESTIM.Temperature("expression", 500)
    my_temp.create_functions(V, FESTIM.Materials(), None, None, None)
    my_sim.T = my_temp
    my_sim.volume_markers, my_sim.surface_markers = None, None
    my_sim.V_CG1, my_sim.V_DG1 = V, V_DG1
    my_sim.t = 0
    my_sim.dt = 1
    my_sim.files = files
    my_sim.append = False
    my_sim.D, my_sim.thermal_cond, my_sim.cp, my_sim.rho, \
        my_sim.H, my_sim.S = *[None]*5, S
    my_sim.derived_quantities_global = []
    my_sim.chemical_pot = True
    run_post_processing(my_sim)

    # check
    u_in = fenics.Function(V)
    files[0].read_checkpoint(u_in, "retention", -1)
    for i in range(10):
        assert np.isclose(u_in(i/10), val_theta*val_S)

    files[1].read_checkpoint(u_in, "solute", -1)
    for i in range(10):
        assert np.isclose(u_in(i/10), val_theta*val_S)
