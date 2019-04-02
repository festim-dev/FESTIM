import FESTIM
import fenics
import pytest
import sympy as sp


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
    f = fenics.Expression("1", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_,
        f)
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    expected_form += -flux_*f*testfunctions[0]*dx

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
    f = fenics.Expression("1", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_,
        f)
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    expected_form += -flux_*f*testfunctions[0]*dx + \
        ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[1]*dx
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (2 - solutions[1]) * \
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
        "density": 2,
        "materials": [1]
        },
        {
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
    f = fenics.Expression("1", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_,
        f)
    # Transient sol
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    # Diffusion sol
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    # Source sol
    expected_form += -flux_*f*testfunctions[0]*dx
    # Transient trap 1
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[1]*dx
    # Trapping trap 1
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (2 - solutions[1]) * \
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
        solutions[0] * (2 - solutions[2]) * \
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
        "density": 2,
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
    f = fenics.Expression("1", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_,
        f)

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
    expected_form += -flux_*f*testfunctions[0]*dx
    # Transient trap 1
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[1]*dx
    # Trapping trap 1 mat 1
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (2 - solutions[1]) * \
        testfunctions[1]*dx(1)
    # Trapping trap 1 mat 2
    expected_form += - 6 * fenics.exp(-5/8.6e-5/temp)/2/2/3 * \
        solutions[0] * (2 - solutions[1]) * \
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
    f = fenics.Expression("1", degree=0)

    F, expressions = FESTIM.formulation(
        traps, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, materials, temp, flux_,
        f)
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx
    expected_form += 5 * fenics.exp(-4/8.6e-5/temp) * \
        fenics.dot(
            fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    expected_form += -flux_*f*testfunctions[0]*dx + \
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
            "boundary_conditions": {
                "dc": [
                    {
                        "surface": [1, 2],
                        "value": u,
                        "component": 0
                    },
                    {
                        "surface": [1, 2],
                        "value": v,
                        "component": 1
                    }
                ],
                "solubility": [  # "surface", "S_0", "E_S", "pressure", "density"
                ]
                },
            "temperature": {
                    'type': "expression",
                    'value': T
                },
            "source_term": {
                'flux': f,
                'distribution': 1
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
        assert error_max_u < tol_u and error_max_v < tol_v
