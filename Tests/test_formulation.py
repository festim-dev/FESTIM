# Unit tests formulation
import FESTIM
import fenics
import pytest
import sympy as sp
from ufl.core.multiindex import Index


def test_fluxes():
    Kr_0 = 2
    E_Kr = 3
    order = 2
    k_B = 8.6e-5
    T = 1000
    boundary_conditions = [

        {
            "type": "recomb",
            "Kr_0": Kr_0,
            "E_Kr": E_Kr,
            "order": order,
            "surface": 1,
            },
        {
           "type": "flux",
           "value": 2*FESTIM.x + FESTIM.t,
           "surface": [1, 2],
        },
    ]
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    u = fenics.Function(V)
    v = fenics.TestFunction(V)

    u = fenics.interpolate(fenics.Expression(('1', '1'), degree=1), V)

    solutions = list(fenics.split(u))
    testfunctions = list(fenics.split(v))
    sol = solutions[0]
    test_sol = testfunctions[0]
    F, expressions = FESTIM.boundary_conditions.apply_fluxes(
        {"boundary_conditions": boundary_conditions}, solutions,
        testfunctions, fenics.ds, T)
    expected_form = 0
    expected_form += -test_sol * (-Kr_0 * fenics.exp(-E_Kr/k_B/T) *
                                  sol**order)*fenics.ds(1)
    expected_form += -test_sol*expressions[0]*fenics.ds(1)
    expected_form += -test_sol*expressions[0]*fenics.ds(2)

    assert expected_form.equals(F) is True


def test_formulation_no_trap_1_material():
    '''
    Test function formulation() with 0 intrinsic trap
    and 1 material
    '''
    Index._globalcount = 8
    dt = 1
    parameters = {
        "materials": [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            }],
        "traps": [],
        "source_term": {"value": "1"},
    }
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

    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, temp, transient=True)

    Index._globalcount = 8
    flux_ = expressions[0]
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx(1)
    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp) * fenics.grad(solutions[0]),
        fenics.grad(testfunctions[0]))*dx(1)
    expected_form += -flux_*testfunctions[0]*dx

    assert expected_form.equals(F) is True


def test_formulation_1_trap_1_material():
    '''
    Test function formulation() with 1 intrinsic trap
    and 1 material
    '''
    Index._globalcount = 8
    dt = 1
    parameters = {
        "traps": [{
            "energy": 1,
            "density": 2,
            "materials": [1]
            }],
        "materials": [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            }],
        "source_term": {"value": 1},
    }
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
    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions,
        testfunctions, previous_solutions, dt, dx, temp, transient=True)
    flux_ = expressions[0]
    Index._globalcount = 8
    # take density Expression() from formulation()
    density = expressions[2]
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx(1)
    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp) * fenics.grad(solutions[0]),
        fenics.grad(testfunctions[0]))*dx(1)
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
    Index._globalcount = 8
    # Set parameters
    dt = 1
    extrinsic_traps = []
    parameters = {
        "traps": [{
            "energy": 1,
            "density": 2 + FESTIM.x,
            "materials": [1]
            },
            {
            "energy": 1,
            "density": 3,
            "materials": [1]
            }],
        "materials": [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            }],
        "source_term": {"value": 1},
    }
    # Prepare
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, len(parameters["traps"])+1)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)

    solutions = list(fenics.split(u))
    previous_solutions = list(fenics.split(u_n))
    testfunctions = list(fenics.split(v))

    mf = fenics.MeshFunction('size_t', mesh, 1, 1)
    dx = fenics.dx(subdomain_data=mf)
    temp = fenics.Expression("300", degree=0)

    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, temp, transient=True)
    flux_ = expressions[0]

    Index._globalcount = 8
    # Densities from formulation()
    density1 = expressions[2]
    density2 = expressions[3]
    # Transient sol
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx(1)
    # Diffusion sol
    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp) * fenics.grad(solutions[0]),
        fenics.grad(testfunctions[0]))*dx(1)
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
    Index._globalcount = 8

    def create_subdomains(x1, x2):
        class domain(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= x1 and x[0] <= x2
        domain = domain()
        return domain
    dt = 1
    parameters = {
        "traps": [
            {
                "energy": 1,
                "density": 2 + FESTIM.x**2,
                "materials": [1, 2]
            }],
        "materials": [{
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
                }],
        "source_term": {"value": 1},
    }
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

    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, temp, transient=True)
    flux_ = expressions[0]
    Index._globalcount = 8
    # Density from formulation()
    density = expressions[2]
    # Transient sol
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx(1)
    expected_form += ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx(2)
    # Diffusion sol mat 1
    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp)*fenics.grad(solutions[0]),
        fenics.grad(testfunctions[0]))*dx(1)
    # Diffusion sol mat 2
    expected_form += fenics.dot(
            6 * fenics.exp(-5/8.6e-5/temp) * fenics.grad(solutions[0]),
            fenics.grad(testfunctions[0]))*dx(2)
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
    Index._globalcount = 8
    dt = 1
    parameters = {
        "traps": [{
            "energy": 1,
            "materials": [1],
            "type": "extrinsic"
            }],
        "materials": [{
                "alpha": 1,
                "beta": 2,
                "density": 3,
                "borders": [0, 1],
                "E_diff": 4,
                "D_0": 5,
                "id": 1
                }],
        "source_term": {"value": 10000},
    }

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

    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, temp, transient=True)
    flux_ = expressions[0]
    Index._globalcount = 8
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx(1)
    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp) * fenics.grad(solutions[0]),
        fenics.grad(testfunctions[0]))*dx(1)
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


def test_formulation_steady_state():
    '''
    Test function formulation() with 1 intrinsic trap
    and 1 material in steady state
    '''
    Index._globalcount = 8
    parameters = {
        "traps": [
            {
              "energy": 1,
              "density": 2,
              "materials": [1]
            }
            ],
        "materials": [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "id": 1
            }],
        "source_term": {"value": 1},
    }
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
    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions,
        testfunctions, previous_solutions, 0, dx, temp, transient=False)
    Index._globalcount = 8
    print(F)
    flux_ = expressions[0]
    density = expressions[2]
    expected_form = -flux_*testfunctions[0]*dx
    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp) * fenics.grad(solutions[0]),
        fenics.grad(testfunctions[0]))*dx(1)
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        solutions[0] * (density - solutions[1]) * \
        testfunctions[1]*dx(1)
    expected_form += 1e13*fenics.exp(-1/8.6e-5/temp)*solutions[1] * \
        testfunctions[1]*dx(1)
    print(expected_form)
    assert expected_form.equals(F) is True


def test_formulation_heat_transfer():
    '''
    Test function define_variational_problem_heat_transfers
    '''

    def thermal_cond(a):
        return a**2

    Index._globalcount = 8
    u = 1 + 2*FESTIM.x**2
    parameters = {
        "materials": [{
            "borders": [0, 1],
            "thermal_cond": thermal_cond,
            "rho": 5,
            "heat_capacity": 4,
            "id": 1
            }],
        "temperature": {
            "type": "solve_transient",
            "boundary_conditions": [
                {
                    "type": "dirichlet",
                    "value": u,
                    "surface": [1]
                },
                {
                    "type": "neumann",
                    "value": 2,
                    "surface": [2]
                },                
                ],
            "source_term": [
                {
                    "value": -4,
                    "volume": 1
                }
            ],
        },
    }
    dt = 2
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.FunctionSpace(mesh, 'P', 1)

    T = fenics.Function(V)
    T_n = fenics.Function(V)
    v = fenics.TestFunction(V)
    functions = [T, v, T_n]

    # create mesh functions
    surface_markers = fenics.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    surface_markers.set_all(0)
    for f in fenics.facets(mesh):
        x0 = f.midpoint()
        if fenics.near(x0.x(), 0):
            surface_markers[f] = 1
        if fenics.near(x0.x(), 1):
            surface_markers[f] = 2
    volume_markers = fenics.MeshFunction('size_t', mesh, 1, 1)
    ds = fenics.Measure('ds', domain=mesh, subdomain_data=surface_markers)
    dx = fenics.Measure('dx', domain=mesh, subdomain_data=volume_markers)
    # Run function
    F, expressions = \
        FESTIM.formulations.define_variational_problem_heat_transfers(
            parameters, functions, [dx, ds], dt=dt)
    Index._globalcount = 8
    source = expressions[0]
    expected_form = 5*4*(T - T_n)/dt * v * dx(1) + fenics.dot(thermal_cond(T)*fenics.grad(T), fenics.grad(v))*dx(1) 
    expected_form += - source*v*dx(1)

    neumann_flux = expressions[1]
    expected_form += -neumann_flux * v * ds(2)

    assert expected_form.equals(F)


def test_formulation_soret():
    Index._globalcount = 8

    def create_subdomains(x1, x2):
        class domain(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= x1 and x[0] <= x2
        domain = domain()
        return domain
    parameters = {
        "traps": [],
        "materials": [{
                "alpha": 1,
                "beta": 2,
                "density": 3,
                "borders": [0, 0.5],
                "E_diff": 4,
                "D_0": 5,
                "H":{
                    "free_enthalpy": 4,
                    "entropy": 3
                },
                "id": 1
                },
                {
                "alpha": 1,
                "beta": 2,
                "density": 3,
                "borders": [0, 0.5],
                "E_diff": 4,
                "D_0": 5,
                "H":{
                    "free_enthalpy": 4,
                    "entropy": 3
                },
                "id": 2
                }
                ],
        "source_term": {"value": 1},
        "temperature": {
            "soret": True
        }
    }
    extrinsic_traps = []
    mesh = fenics.UnitIntervalMesh(10)
    mf = fenics.MeshFunction("size_t", mesh, 1, 1)
    mat1 = create_subdomains(0, 0.5)
    mat2 = create_subdomains(0.5, 1)
    mat1.mark(mf, 1)
    mat2.mark(mf, 2)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)

    solutions = list(fenics.split(u))
    previous_solutions = list(fenics.split(u_n))
    testfunctions = list(fenics.split(v))

    mf = fenics.MeshFunction('size_t', mesh, 1, 1)
    dx = fenics.dx(subdomain_data=mf)
    temp = fenics.Expression("300", degree=0)
    temp = fenics.interpolate(temp, V)  # temp must be a function and not an expression in that case
    dt = 2
    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, temp, transient=True)
    flux_ = expressions[0]
    Index._globalcount = 8

    # Transient sol
    expected_form = ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx(1)
    expected_form += ((solutions[0] - previous_solutions[0]) / dt) * \
        testfunctions[0]*dx(2)
    # Diffusion sol mat 1
    expected_form += fenics.dot(5 * fenics.exp(-4/FESTIM.k_B/temp)*fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(1)
    expected_form += fenics.dot(5 * fenics.exp(-4/FESTIM.k_B/temp)*(4*temp + 3)*solutions[0]/(FESTIM.R*temp**2)*fenics.grad(temp), fenics.grad(testfunctions[0]))*dx(1)
    # Diffusion sol mat 2
    expected_form += fenics.dot(5 * fenics.exp(-4/FESTIM.k_B/temp) * fenics.grad(solutions[0]), fenics.grad(testfunctions[0]))*dx(2)
    expected_form += fenics.dot(5 * fenics.exp(-4/FESTIM.k_B/temp) * (4*temp + 3)*solutions[0]/(FESTIM.R*temp**2)*fenics.grad(temp), fenics.grad(testfunctions[0]))*dx(2)
    # Source sol
    expected_form += -flux_*testfunctions[0]*dx

    assert expected_form.equals(F) is True


def test_formulation_no_trap_1_material_chemical_pot():
    '''
    Test function formulation() with 0 intrinsic trap
    and 1 material with chemical potential conservation
    '''
    Index._globalcount = 8
    dt = 1
    parameters = {
        "materials": [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 1],
            "E_diff": 4,
            "D_0": 5,
            "S_0": 2,
            "E_S": 2,
            "id": 1
            }],
        "traps": [],
        "source_term": {"value": "1"},
    }
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
    temp_n = fenics.Expression("200", degree=0)

    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions, testfunctions,
        previous_solutions, dt, dx, temp, T_n=temp_n, transient=True)

    Index._globalcount = 8
    flux_ = expressions[0]
    theta = solutions[0]*2*fenics.exp(-2/8.6e-5/temp)
    theta_n = previous_solutions[0]*2*fenics.exp(-2/8.6e-5/temp_n)
    expected_form = ((theta - theta_n) / dt) * testfunctions[0]*dx(1)
    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp) * fenics.grad(theta),
        fenics.grad(testfunctions[0]))*dx(1)
    expected_form += -flux_*testfunctions[0]*dx

    assert expected_form.equals(F) is True


def test_formulation_1_trap_1_material_chemical_pot():
    '''
    Test function formulation() with 1 intrinsic trap
    and 1 material with chemical potential conservation
    '''
    Index._globalcount = 8
    dt = 1
    parameters = {
        "traps": [
            {
            "energy": 1,
            "density": 2,
            "materials": [1]
            }
        ],
        "materials": [{
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0, 0.5],
            "E_diff": 4,
            "D_0": 5,
            "S_0": 2,
            "E_S": 2,
            "id": 1
            },
            {
            "alpha": 1,
            "beta": 2,
            "density": 3,
            "borders": [0.5, 1],
            "E_diff": 4,
            "D_0": 5,
            "S_0": 3,
            "E_S": 3,
            "id": 2
            },
            ],
    }
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
    temp_n = fenics.Expression("200", degree=0)
    F, expressions = FESTIM.formulations.formulation(
        parameters, extrinsic_traps, solutions,
        testfunctions, previous_solutions, dt, dx, temp, temp_n, transient=True)
    Index._globalcount = 8
    # take density Expression() from formulation()
    print(expressions)
    density = expressions[1]

    theta1 = solutions[0]*2*fenics.exp(-2/8.6e-5/temp)
    theta1_n = previous_solutions[0]*2*fenics.exp(-2/8.6e-5/temp_n)
    theta2 = solutions[0]*3*fenics.exp(-3/8.6e-5/temp)
    theta2_n = previous_solutions[0]*3*fenics.exp(-3/8.6e-5/temp_n)

    expected_form = ((theta1 - theta1_n) / dt) * \
        testfunctions[0]*dx(1)
    expected_form += ((theta2 - theta2_n) / dt) * \
        testfunctions[0]*dx(2)

    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp) * fenics.grad(theta1),
        fenics.grad(testfunctions[0]))*dx(1)
    expected_form += fenics.dot(
        5 * fenics.exp(-4/8.6e-5/temp) * fenics.grad(theta2),
        fenics.grad(testfunctions[0]))*dx(2)

    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[1]*dx
    expected_form += - 5 * fenics.exp(-4/8.6e-5/temp)/1/1/2 * \
        theta1 * (density - solutions[1]) * \
        testfunctions[1]*dx(1)
    expected_form += 1e13*fenics.exp(-1/8.6e-5/temp)*solutions[1] * \
        testfunctions[1]*dx(1)
    expected_form += ((solutions[1] - previous_solutions[1]) / dt) * \
        testfunctions[0]*dx
    assert expected_form.equals(F) is True
