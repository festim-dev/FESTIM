# Unit tests formulation
from FESTIM import FESTIM
import fenics
import pytest
import sympy as sp


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
