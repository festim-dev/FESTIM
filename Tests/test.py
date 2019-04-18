from FESTIM import FESTIM
import fenics
import pytest
import sympy as sp


# Unit tests

def test_mesh_and_refine_meets_refinement_conditions():
    '''
    Test that function mesh_and_refine() gives the right
    refinement conditions
    '''
    def create_subdomains(x1, x2):
        class domain(FESTIM.SubDomain):
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
        mesh = FESTIM.mesh_and_refine(param)

        mf1 = FESTIM.MeshFunction('size_t', mesh, 1)
        mf2 = FESTIM.MeshFunction('size_t', mesh, 1)
        subdomain1 = create_subdomains(0, refinements[i][1][1])
        subdomain1.mark(mf1, 1)
        subdomain2 = create_subdomains(0, refinements[i][1][0])
        subdomain2.mark(mf2, 2)
        nb_cell_1 = 0
        nb_cell_2 = 0
        for cell in FESTIM.cells(mesh):
            cell_no = cell.index()
            if mf1.array()[cell_no] == 1:
                nb_cell_1 += 1
            if mf2.array()[cell_no] == 2:
                nb_cell_2 += 1
        assert nb_cell_1 >= refinements[i][0][1]
        assert nb_cell_2 >= refinements[i][0][0]


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
    assert len(expected) == len(FESTIM.define_xdmf_files(exports))

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
        FESTIM.define_xdmf_files(exports)

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
        FESTIM.define_xdmf_files(exports)


def test_export_xdmf():
    mesh = fenics.UnitSquareMesh(3, 3)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    folder = "Solution"
    exports = {
        "xdmf": {
            "functions": ['solute', 'retention'],
            "labels":  ['a', 'b'],
            "folder": folder
        }
        }
    files = [fenics.XDMFFile(folder + "/" + "a.xdmf"),
             fenics.XDMFFile(folder + "/" + "b.xdmf")]
    assert FESTIM.export_xdmf(
        [fenics.Function(V), fenics.Function(V)],
        exports, files, 20) is None

    exports["xdmf"]["functions"] = ['solute', 'blabla']

    with pytest.raises(KeyError, match=r'blabla'):
        FESTIM.export_xdmf(
            [fenics.Function(V), fenics.Function(V)],
            exports, files, 20)

    exports["xdmf"]["functions"] = ['solute', '13']
    with pytest.raises(KeyError, match=r'13'):
        FESTIM.export_xdmf(
            [fenics.Function(V), fenics.Function(V)],
            exports, files, 20)


def test_apply_boundary_conditions():
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, 'P', 1)

    surface_markers = fenics.MeshFunction(
        "size_t", mesh, mesh.topology().dim()-1, 0)
    surface_markers.set_all(0)
    i = 0
    for f in fenics.facets(mesh):
        i += 1
        x0 = f.midpoint()
        surface_markers[f] = 0
        if fenics.near(x0.x(), 0):
            surface_markers[f] = 1
        if fenics.near(x0.x(), 1):
            surface_markers[f] = 2

    boundary_conditions = [
        {
            "surface": [1],
            "value": 0,
            "component": 0,
            "type": "dc"
        },
        {
            "surface": [2],
            "value": 1,
            "type": "dc"
        }
        ]
    bcs, expressions = FESTIM.apply_boundary_conditions(
        boundary_conditions, V, surface_markers, 1, 300)
    assert len(bcs) == 2
    assert len(expressions) == 2

    u = fenics.Function(V)
    for bc in bcs:
        bc.apply(u.vector())
    assert abs(u(0)-0) < 1e-15
    assert abs(u(1)-1) < 1e-15


def test_formulation_no_trap_1_material():
    '''
    Test function formulation() with 1 intrinsic trap
    and 1 material
    '''
    dt = 1
    traps = []
    materials = [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            }]
    extrinsic_traps = []
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 1)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)

    solutions = list(fenics.split(u))
    previous_solutions = list(fenics.split(u_n))
    testfunctions = list(fenics.split(v))

    mf = fenics.MeshFunction('size_t', mesh, 1, 1)
    dx = fenics.dx(subdomain_data=mf)
    temp = fenics.Expression("300", degree=0)
    flux_ = fenics.Expression("1", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_)
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    expected_form += -flux_*testfunctions[0]*dx

    assert expected_form.equals(F) is True


def test_formulation_1_trap_1_material():
    '''
    Test function formulation() with 1 intrinsic trap
    and 1 material
    '''
    dt = 1
    traps = [{
        "energy": 1,
        "density": 2,
        "materials": [1]
        }]
    materials = [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            }]
    extrinsic_traps = []
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)

    solutions = list(fenics.split(u))
    previous_solutions = list(fenics.split(u_n))
    testfunctions = list(fenics.split(v))

    mf = fenics.MeshFunction('size_t', mesh, 1, 1)
    dx = fenics.dx(subdomain_data=mf)
    temp = fenics.Expression("300", degree=0)
    flux_ = fenics.Expression("1", degree=0)
    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_)

    # take density Expression() from formulation()
    density = expressions[2]
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    expected_form += -flux_*testfunctions[0]*dx + \
        ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[1]*dx
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (density - solutions[1]) * \
        testfunctions[1]*dx(1)
    expected_form += 1e13*fenics.exp(-1/8.6e-5/temp)*solutions[1] * \
        testfunctions[1]*dx(1)
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[0]*dx
    assert expected_form.equals(F) is True


def test_formulation_2_traps_1_material():
    '''
    Test function formulation() with 2 intrinsic traps
    and 1 material
    '''
    # Set parameters
    dt = 1
    traps = [{
        "energy": 1,
        "density": 2 + FESTIM.x,
        "materials": [1]
        },
        {
        "energy": 1,
        "density": 3,
        "materials": [1]
        }]
    materials = [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            }]
    extrinsic_traps = []

    # Prepare
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, len(traps)+1)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)

    solutions = list(fenics.split(u))
    previous_solutions = list(fenics.split(u_n))
    testfunctions = list(fenics.split(v))

    mf = fenics.MeshFunction('size_t', mesh, 1, 1)
    dx = fenics.dx(subdomain_data=mf)
    temp = fenics.Expression("300", degree=0)
    flux_ = fenics.Expression("1", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_)

    # Densities from formulation()
    density1 = expressions[2]
    density2 = expressions[3]
    # Transient sol
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    # Diffusion sol
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    # Source sol
    expected_form += -flux_*testfunctions[0]*dx
    # Transient trap 1
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[1]*dx
    # Trapping trap 1
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (density1 - solutions[1]) * \
        testfunctions[1]*dx(1)
    # Detrapping trap 1
    expected_form += 1e13*fenics.exp(-1/8.6e-5/temp)*solutions[1] * \
        testfunctions[1]*dx(1)
    # Source detrapping sol
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[0]*dx

    # Transient trap 2
    expected_form += ((solutions[2] - previous_solutions[2]) / dt) * \
        testfunctions[2]*dx
    # Trapping trap 2
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (density2 - solutions[2]) * \
        testfunctions[2]*dx(1)
    # Detrapping trap 2
    expected_form += 1e13*fenics.exp(-1/8.6e-5/temp)*solutions[2] * \
        testfunctions[2]*dx(1)
    # Source detrapping 2 sol
    expected_form += ((solutions[2] - previous_solutions[2]) / dt) * \
        testfunctions[0]*dx

    assert expected_form.equals(F) is True


def test_formulation_1_trap_2_materials():
    '''
    Test function formulation() with 1 intrinsic trap
    and 2 materials
    '''
    def create_subdomains(x1, x2):
        class domain(FESTIM.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= x1 and x[0] <= x2
        domain = domain()
        return domain
    dt = 1
    traps = [{
        "energy": 1,
        "density": 2 + FESTIM.x**2,
        "materials": [1, 2]
        }]
    materials = [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 0.5],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            },
            {
            "alpha": 2,
            "beta": 3,
            "density": 4,
            "borders": [0.5, 1],
            "E_diff": 5,
            "D_0": 6,
            "id": 2
            }]
    extrinsic_traps = []
    mesh = fenics.UnitIntervalMesh(10)
    mf = fenics.MeshFunction("size_t", mesh, 1, 1)
    mat1 = create_subdomains(0, 0.5)
    mat2 = create_subdomains(0.5, 1)
    mat1.mark(mf, 1)
    mat2.mark(mf, 2)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)

    solutions = list(fenics.split(u))
    previous_solutions = list(fenics.split(u_n))
    testfunctions = list(fenics.split(v))

    mf = fenics.MeshFunction('size_t', mesh, 1, 1)
    dx = fenics.dx(subdomain_data=mf)
    temp = fenics.Expression("300", degree=0)
    flux_ = fenics.Expression("1", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_)

    # Density from formulation()
    density = expressions[2]
    # Transient sol
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    # Diffusion sol mat 1
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    # Diffusion sol mat 2
    expected_form += 6 * fenics.exp(-5/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(2)
    # Source sol
    expected_form += -flux_*testfunctions[0]*dx
    # Transient trap 1
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[1]*dx
    # Trapping trap 1 mat 1
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (density - solutions[1]) * \
        testfunctions[1]*dx(1)
    # Trapping trap 1 mat 2
    expected_form += - 6 * fenics.exp(-5/8.6e-5/temp)/2/2/3 * \
        solutions[0] * (density - solutions[1]) * \
        testfunctions[1]*dx(2)
    # Detrapping trap 1 mat 1
    expected_form += 1e13*fenics.exp(-1/8.6e-5/temp)*solutions[1] * \
        testfunctions[1]*dx(1)
    # Detrapping trap 1 mat 2
    expected_form += 1e13*fenics.exp(-1/8.6e-5/temp)*solutions[1] * \
        testfunctions[1]*dx(2)
    # Source detrapping sol
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[0]*dx

    assert expected_form.equals(F) is True


def test_formulation_1_extrap_1_material():
    '''
    Test function formulation() with 1 extrinsic trap
    and 1 material
    '''
    dt = 1
    traps = [{
        "energy": 1,
        "materials": [1],
        "type": "extrinsic"
        }]
    materials = [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            }]

    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    W = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)
    n = fenics.interpolate(fenics.Expression('1', degree=0), W)
    solutions = list(fenics.split(u))
    previous_solutions = list(fenics.split(u_n))
    testfunctions = list(fenics.split(v))
    extrinsic_traps = [n]
    mf = fenics.MeshFunction('size_t', mesh, 1, 1)
    dx = fenics.dx(subdomain_data=mf)
    temp = fenics.Expression("300", degree=0)
    flux_ = fenics.Expression("10000", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_)
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    expected_form += -flux_*testfunctions[0]*dx + \
        ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[1]*dx
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (extrinsic_traps[0] - solutions[1]) * \
        testfunctions[1]*dx(1)
    expected_form += 1e13*fenics.exp(-1/8.6e-5/temp)*solutions[1] * \
        testfunctions[1]*dx(1)
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[0]*dx

    assert expected_form.equals(F) is True


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
    A, B, C = FESTIM.create_flux_functions(mesh, materials, mf)
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
    tab = FESTIM.derived_quantities(parameters, [u, u, T], [1, 1],
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

    tab = FESTIM.header_derived_quantities(parameters)
    expected = ["t(s)",
                "Flux surface 2: solute", "Flux surface 2: T",
                "Average T volume 1",
                "Minimum solute volume 2", "Maximum T volume 1",
                "Total solute volume 1", "Total solute volume 2",
                "Total solute surface 2"]
    assert len(tab) == len(expected)
    for i in range(0, len(tab)):
        assert tab[i] == expected[i]


# Integration tests


def test_run_temperature_stationary():
    '''
    Check that the temperature module works well in 1D stationary
    '''
    u = 1 + 2*FESTIM.x**2
    size = 1
    parameters = {
        "materials": [
            {
                "thermal_cond": 1,
                "alpha": 1.1e-10,
                "beta": 6*6.3e28,
                "density": 6.3e28,
                "borders": [0, size],
                "E_diff": 0.39,
                "D_0": 4.1e-7,
                "id": 1
                }
                ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": size,
                "refinements": [
                ],
            },
        "boundary_conditions": [
                    {
                        "surface": [1],
                        "value": 1,
                        "component": 0,
                        "type": "dc"
                    }
            ],
        "temperature": {
            "type": "solve_stationary",
            "boundary_conditions": [
                {
                    "type": "dirichlet",
                    "value": u,
                    "surface": [1, 2]
                }
                ],
            "source_term": [
                {
                    "value": -4,
                    "volume": 1
                }
            ],
            "initial_condition": u
        },
        "source_term": {
            'flux': 0
            },
        "solving_parameters": {
            "final_time": 30,
            "num_steps": 60,
            "adaptative_time_step": {
                "stepsize_change_ratio": 1,
                "t_stop": 40,
                "stepsize_stop_max": 0.5,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_it": 50,
            }
            },
        "exports": {
            "txt": {
                "functions": ['retention'],
                "times": [100],
                "labels": ['retention']
            },
            "xdmf": {
                "functions": [],
                "labels":  [],
                "folder": "Coucou"
            },
            "TDS": {
                "file": "desorption",
                "TDS_time": 450
                }
            },

    }
    output = FESTIM.run(parameters)
    # temp at the middle
    T_computed = output["temperature"][1][1]
    assert abs(T_computed - (1+2*(size/2)**2)) < 1e-9


def test_run_temperature_transient():
    '''
    Check that the temperature module works well in 1D transient
    '''
    u = 1 + 2*FESTIM.x**2+FESTIM.t
    size = 1
    parameters = {
        "materials": [
            {
                "thermal_cond": 1,
                "rho": 1,
                "heat_capacity": 1,
                "alpha": 1.1e-10,  # lattice constant ()
                "beta": 6*6.3e28,  # number of solute sites per atom (6 for W)
                "density": 6.3e28,
                "borders": [0, size],
                "E_diff": 0.39,
                "D_0": 4.1e-7,
                "id": 1
            }
            ],
        "traps": [
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": size,
                "refinements": [
                ],
            },
        "boundary_conditions": [
                    {
                        "surface": [1],
                        "value": 1,
                        "component": 0,
                        "type": "dc"
                    }
            ],
        "temperature": {
            "type": "solve_transient",
            "boundary_conditions": [
                {
                    "type": "dirichlet",
                    "value": u,
                    "surface": [1, 2]
                }
                ],
            "source_term": [
                {
                    "value": sp.diff(u, FESTIM.t) - sp.diff(u, FESTIM.x, 2),
                    "volume": 1
                }
            ],
            "initial_condition": u
        },
        "source_term": {
            'flux': 0
            },
        "solving_parameters": {
            "final_time": 30,
            "num_steps": 60,
            "adaptative_time_step": {
                "stepsize_change_ratio": 1,
                "t_stop": 40,
                "stepsize_stop_max": 0.5,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_it": 50,
            }
            },
        "exports": {
            "txt": {
                "functions": ['retention'],
                "times": [100],
                "labels": ['retention']
            },
            "xdmf": {
                "functions": [],
                "labels":  [],
                "folder": "Coucou"
            },
            "TDS": {
                "file": "desorption",
                "TDS_time": 450
                }
            },

    }
    output = FESTIM.run(parameters)
    # temp at the middle
    T_computed = output["temperature"][1][1]
    error = []
    u_D = fenics.Expression(sp.printing.ccode(u), t=0, degree=4)
    for i in range(1, len(output["temperature"])):
        t = output["temperature"][i][0]
        T = output["temperature"][i][1]
        u_D.t = t
        error.append(abs(T - u_D(size/2)))
    assert max(error) < 1e-9


def test_run_MMS():
    '''
    Test function run() for several refinements
    '''

    u = 1 + sp.exp(-4*fenics.pi**2*FESTIM.t)*sp.cos(2*fenics.pi*FESTIM.x)
    v = 1 + sp.exp(-4*fenics.pi**2*FESTIM.t)*sp.cos(2*fenics.pi*FESTIM.x)

    def parameters(h, dt, final_time, u, v):
        size = 1
        folder = 'Solution_Test'
        v_0 = 1e13
        E_t = 1.5
        T = 700
        density = 1 * 6.3e28
        beta = 6*density
        alpha = 1.1e-10
        n_trap = 1e-1*density
        E_diff = 0.39
        D_0 = 4.1e-7
        k_B = 8.6e-5
        D = D_0 * fenics.exp(-E_diff/k_B/T)
        v_i = v_0 * fenics.exp(-E_t/k_B/T)
        v_m = D/alpha/alpha/beta

        f = sp.diff(u, FESTIM.t) + sp.diff(v, FESTIM.t) - \
            D * sp.diff(u, FESTIM.x, 2)
        g = sp.diff(v, FESTIM.t) + v_i*v - v_m * u * (n_trap-v)
        parameters = {
            "materials": [
                {
                    "alpha": alpha,  # lattice constant ()
                    "beta": beta,  # number of solute sites per atom (6 for W)
                    "density": density,
                    "borders": [0, size],
                    "E_diff": E_diff,
                    "D_0": D_0,
                    "id": 1
                    }
                    ],
            "traps": [
                {
                    "energy": E_t,
                    "density": n_trap,
                    "materials": 1,
                    "source_term": g
                }
                ],
            "initial_conditions": [
                {
                    "value": u,
                    "component": 0
                },
                {
                    "value": v,
                    "component": 1
                }
            ],

            "mesh_parameters": {
                    "initial_number_of_cells": round(size/h),
                    "size": size,
                    "refinements": [
                    ],
                },
            "boundary_conditions": [
                    {
                        "surface": [1, 2],
                        "value": u,
                        "component": 0,
                        "type": "dc"
                    },
                    {
                        "surface": [1, 2],
                        "value": v,
                        "component": 1,
                        "type": "dc"
                    }
                ],
            "temperature": {
                    'type': "expression",
                    'value': T
                },
            "source_term": {
                'flux': f
                },
            "solving_parameters": {
                "final_time": final_time,
                "num_steps": round(1/dt),
                "adaptative_time_step": {
                    "stepsize_change_ratio": 1,
                    "t_stop": 0,
                    "stepsize_stop_max": dt,
                    "dt_min": 1e-5
                    },
                "newton_solver": {
                    "absolute_tolerance": 1e-10,
                    "relative_tolerance": 1e-9,
                    "maximum_it": 50,
                }
                },
            "exports": {
                "txt": {
                    "functions": [],
                    "times": [],
                    "labels": [],
                    "folder": folder
                },
                "xdmf": {
                    "functions": [],
                    "labels":  [],
                    "folder": folder
                },
                "TDS": {
                    "file": "desorption",
                    "TDS_time": 0,
                    "folder": folder
                    },
                "error": [
                    {
                        "exact_solution": [u, v],
                        "norm": 'error_max',
                        "degree": 4
                    }
                ]
                },
        }
        return parameters

    tol_u = 1e-7
    tol_v = 1e-1
    sizes = [1/1600, 1/1700]
    dt = 1/50
    final_time = 0.1
    for h in sizes:
        output = FESTIM.run(parameters(h, dt, final_time, u, v))
        error_max_u = output["error"][0][1]
        error_max_v = output["error"][0][2]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h) + '\n \
            with dt = ' + str(dt)
        print(msg)
        assert output["temperature"][1][1] == 700
        assert output["temperature"][5][1] == 700
        assert error_max_u < tol_u and error_max_v < tol_v
