import FESTIM
from FESTIM.boundary_conditions import apply_boundary_conditions, \
    apply_fluxes
import fenics
import pytest
import sympy as sp
import numpy as np


def test_fluxes_chemical_pot():
    '''
    This test that the function boundary_conditions.apply_fluxes()
    returns the correct formulation in the case of conservation
    of chemical potential
    '''
    Kr_0 = 2
    E_Kr = 3
    S_0 = 2
    E_S = 3
    order = 2
    k_B = FESTIM.k_B
    T = 1000
    parameters = {
        "materials": [
            {
                "S_0": S_0,
                "E_S": E_S,
                "id": 1
            },
            {
                "S_0": S_0,
                "E_S": E_S,
                "id": 2
            }
        ],
        "boundary_conditions": [
            {
                "type": "recomb",
                "Kr_0": Kr_0,
                "E_Kr": E_Kr,
                "order": order,
                "surfaces": 1,
            },
            {
                "type": "flux",
                "value": 2*FESTIM.x + FESTIM.t,
                "surfaces": [1, 2],
            },
        ]
    }
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    u = fenics.Function(V)
    v = fenics.TestFunction(V)

    solutions = list(fenics.split(u))
    testfunctions = list(fenics.split(v))
    sol = solutions[0]
    test_sol = testfunctions[0]

    S = S_0*fenics.exp(-E_S/k_B/T)
    Kr = -Kr_0 * fenics.exp(-E_Kr/k_B/T)
    F, expressions = apply_fluxes(
        parameters, u, v, fenics.ds, T, S)
    expected_form = 0
    expected_form += -test_sol * (Kr*(sol*S)**order)*fenics.ds(1)
    expected_form += -test_sol*expressions[0]*fenics.ds(1)
    expected_form += -test_sol*expressions[0]*fenics.ds(2)
    assert expected_form.equals(F) is True


def test_fluxes():
    Kr_0 = 2
    E_Kr = 3
    order = 2
    k_B = FESTIM.k_B
    T = 1000
    boundary_conditions = [

        {
            "type": "recomb",
            "Kr_0": Kr_0,
            "E_Kr": E_Kr,
            "order": order,
            "surfaces": 1,
            },
        {
           "type": "flux",
           "value": 2*FESTIM.x + FESTIM.t,
           "surfaces": [1, 2],
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
    F, expressions = apply_fluxes(
        {"boundary_conditions": boundary_conditions}, u,
        v, fenics.ds, T)
    expected_form = 0
    expected_form += -test_sol * (-Kr_0 * fenics.exp(-E_Kr/k_B/T) *
                                  sol**order)*fenics.ds(1)
    expected_form += -test_sol*expressions[0]*fenics.ds(1)
    expected_form += -test_sol*expressions[0]*fenics.ds(2)

    assert expected_form.equals(F) is True

    # Test error raise
    with pytest.raises(NameError, match=r'Unknown boundary condition type'):
        boundary_conditions[0].update({"type": "foo"})
        apply_fluxes(
            {"boundary_conditions": boundary_conditions}, u,
            v, fenics.ds, T)


def test_apply_boundary_conditions_theta():
    '''
    Test the function apply_boundary_condition()
    when conservation of chemical potential is
    required.
    Meaning that the BC value given by the user must
    be multiplied by S(T) in apply_boundary_condition()
    - multi material
    - transient bc
    - transient T
    - space dependent T
    '''
    S_01 = 2
    S_02 = 3
    E_S1 = 0.1
    E_S2 = 0.2
    parameters = {
        "materials": [
            {
                "S_0": S_01,
                "E_S": E_S1,
                "id": 1
                },
            {
                "S_0": S_02,
                "E_S": E_S2,
                "id": 2
                }
        ],
        "boundary_conditions": [
            {
                "type": "dc",
                "value": 200 + FESTIM.t,
                "surfaces": [1, 2]
            },
        ]
    }

    mesh = fenics.UnitSquareMesh(4, 4)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    u = fenics.Function(V)
    v = fenics.TestFunction(V)
    temp = fenics.Expression("200 + (x[0] + 1)*t", t=1, degree=1)

    vm = fenics.MeshFunction("size_t", mesh, 2, 1)
    left = fenics.CompiledSubDomain('x[0] < 0.5')
    right = fenics.CompiledSubDomain('x[0] >= 0.5')
    left.mark(vm, 1)
    right.mark(vm, 2)

    sm = fenics.MeshFunction("size_t", mesh, 1, 0)
    left = fenics.CompiledSubDomain('x[0] < 0.0001')
    left.mark(sm, 1)
    right = fenics.CompiledSubDomain('x[0] > 0.99999999')
    right.mark(sm, 2)
    bcs, expressions = \
        apply_boundary_conditions(
            parameters, V, [vm, sm], temp)

    F = fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx

    for i in range(0, 3):
        temp.t = i
        expressions[0].t = i
        expressions[1].t = i

        # Test that the expression is correct at vertices
        expr = fenics.interpolate(expressions[1], V)
        assert np.isclose(
            expr(0.25, 0.5),
            (200 + i)/(S_01*np.exp(-E_S1/FESTIM.k_B/temp(0.25, 0.5))))
        assert np.isclose(
            expr(0.75, 0.5),
            (200 + i)/(S_02*np.exp(-E_S2/FESTIM.k_B/temp(0.75, 0.5))))

        # Test that the BCs can be applied to a problem
        # and gives the correct values
        fenics.solve(F == 0, u, bcs[0])
        assert np.isclose(
            u(0.25, 0.5),
            (200 + i)/(S_01*np.exp(-E_S1/FESTIM.k_B/temp(0, 0.5))))
        fenics.solve(F == 0, u, bcs[1])
        assert np.isclose(
            u(0.75, 0.5),
            (200 + i)/(S_02*np.exp(-E_S2/FESTIM.k_B/temp(1, 0.5))))


def test_apply_boundary_conditions_fail():
    boundary_conditions = [
        {
            "type": "foo"
        }
    ]
    mesh = fenics.UnitIntervalMesh(10)
    V = fenics.VectorFunctionSpace(mesh, 'P', 1, 2)
    T = fenics.Expression("300", degree=0)
    markers = ["foo", "foo"]  # not needed here since only failing is testing
    with pytest.raises(KeyError, match=r'Missing boundary condition type key'):
        apply_boundary_conditions(
            {"boundary_conditions": [{}]}, V,
            markers, T)

    with pytest.raises(NameError, match=r'Unknown boundary condition type'):
        apply_boundary_conditions(
            {"boundary_conditions": boundary_conditions}, V,
            markers, T)


def test_bc_recomb():
    """Test the function boundary_conditions.apply_boundary_conditions
    with bc type dc_imp
    """
    phi = 3 + 10*FESTIM.t
    R_p = 5 + FESTIM.x
    D_0 = 2
    E_D = 0.5
    K_0 = 2
    E_K = 0.35
    parameters = {
        "materials": [
            {
                "id": 1
                },
            {
                "id": 2
                }
        ],
        "boundary_conditions": [
            {
                "type": "dc_imp",
                "implanted_flux": phi,
                "implantation_depth": R_p,
                "D_0": D_0,
                "E_D": E_D,
                "K_0": K_0,
                "E_K": E_K,
                "surfaces": [1, 2]
            },
        ]
    }

    mesh = fenics.UnitSquareMesh(4, 4)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    T_expr = 500 + (FESTIM.x + 1)*100*FESTIM.t
    temp = fenics.Expression(sp.printing.ccode(T_expr), t=0, degree=1)

    sm = fenics.MeshFunction("size_t", mesh, 1, 0)
    left = fenics.CompiledSubDomain('x[0] < 0.0001')
    left.mark(sm, 1)
    right = fenics.CompiledSubDomain('x[0] > 0.99999999')
    right.mark(sm, 2)

    bcs, expressions = \
        apply_boundary_conditions(
            parameters, V, [None, sm], temp)
    for current_time in range(0, 3):
        temp.t = current_time
        expressions[0].t = current_time
        expressions[1].t = current_time

        for x_ in [0, 1]:
            T = float(T_expr.subs(FESTIM.t, current_time).subs(FESTIM.x, x_))
            D = D_0*np.exp(-E_D/FESTIM.k_B/T)
            K = K_0*np.exp(-E_K/FESTIM.k_B/T)
            # Test that the expression is correct at vertices
            val_phi = phi.subs(FESTIM.t, current_time).subs(FESTIM.x, x_)
            val_R_p = R_p.subs(FESTIM.t, current_time).subs(FESTIM.x, x_)
            assert np.isclose(
                expressions[2](x_, 0.5),
                float(val_phi*val_R_p/D +
                      (val_phi/K)**0.5))


def test_bc_recomb_instant_recomb():
    """Test the function boundary_conditions.apply_boundary_conditions
    with bc type dc_imp (with instantaneous recombination)
    """
    phi = 3 + 10*FESTIM.t
    R_p = 5 + FESTIM.x
    D_0 = 200
    E_D = 0.25
    parameters = {
        "materials": [
            {
                "id": 1
                },
            {
                "id": 2
                }
        ],
        "boundary_conditions": [
            {
                "type": "dc_imp",
                "implanted_flux": phi,
                "implantation_depth": R_p,
                "D_0": D_0,
                "E_D": E_D,
                "surfaces": [1, 2]
            },
        ]
    }
    # Set up
    mesh = fenics.UnitSquareMesh(4, 4)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    T_expr = 500 + (FESTIM.x + 1)*100*FESTIM.t
    temp = fenics.Expression(sp.printing.ccode(T_expr), t=0, degree=1)

    sm = fenics.MeshFunction("size_t", mesh, 1, 0)
    left = fenics.CompiledSubDomain('x[0] < 0.0001')
    left.mark(sm, 1)
    right = fenics.CompiledSubDomain('x[0] > 0.99999999')
    right.mark(sm, 2)

    bcs, expressions = \
        apply_boundary_conditions(
            parameters, V, [None, sm], temp)
    for current_time in range(0, 3):
        temp.t = current_time
        expressions[0].t = current_time
        expressions[1].t = current_time

        for x_ in [0, 1]:
            T = float(T_expr.subs(FESTIM.t, current_time).subs(FESTIM.x, x_))
            D = D_0*np.exp(-E_D/FESTIM.k_B/T)
            # Test that the expression is correct at vertices
            val_phi = phi.subs(FESTIM.t, current_time).subs(FESTIM.x, x_)
            val_R_p = R_p.subs(FESTIM.t, current_time).subs(FESTIM.x, x_)
            assert np.isclose(
                expressions[2](x_, 0.5),
                float(val_phi*val_R_p/D))


def test_bc_recomb_chemical_pot():
    """Tests the function boundary_conditions.apply_boundary_conditions()
    with type dc_imp and conservation of chemical potential
    """
    phi = 3
    R_p = 5
    D_0 = 2
    E_D = 0.5
    K_0 = 2
    E_K = 0.5
    S_01 = 2
    S_02 = 3
    E_S1 = 0.1
    E_S2 = 0.2
    parameters = {
        "materials": [
            {
                "S_0": S_01,
                "E_S": E_S1,
                "id": 1
                },
            {
                "S_0": S_02,
                "E_S": E_S2,
                "id": 2
                }
        ],
        "boundary_conditions": [
            {
                "type": "dc_imp",
                "implanted_flux": phi,
                "implantation_depth": R_p,
                "D_0": D_0,
                "E_D": E_D,
                "K_0": K_0,
                "E_K": E_K,
                "surfaces": [1, 2]
            },
        ]
    }

    mesh = fenics.UnitSquareMesh(4, 4)
    V = fenics.FunctionSpace(mesh, 'P', 1)

    temp = fenics.Expression("200 + (x[0] + 1)*t", t=1, degree=1)

    vm = fenics.MeshFunction("size_t", mesh, 2, 1)
    left = fenics.CompiledSubDomain('x[0] < 0.5')
    right = fenics.CompiledSubDomain('x[0] >= 0.5')
    left.mark(vm, 1)
    right.mark(vm, 2)

    sm = fenics.MeshFunction("size_t", mesh, 1, 0)
    left = fenics.CompiledSubDomain('x[0] < 0.0001')
    left.mark(sm, 1)
    right = fenics.CompiledSubDomain('x[0] > 0.99999999')
    right.mark(sm, 2)

    bcs, expressions = \
        apply_boundary_conditions(
            parameters, V, [vm, sm], temp)

    # Set up formulation
    u = fenics.Function(V)
    v = fenics.TestFunction(V)
    F = fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx

    for i in range(0, 3):
        temp.t = i
        expressions[0].t = i
        expressions[1].t = i

        T_left = 200 + i
        T_right = 200 + 2*i
        D_left = D_0*np.exp(-E_D/FESTIM.k_B/T_left)
        D_right = D_0*np.exp(-E_D/FESTIM.k_B/T_right)
        K_left = K_0*np.exp(-E_K/FESTIM.k_B/T_left)
        K_right = K_0*np.exp(-E_K/FESTIM.k_B/T_right)
        S_left = S_01*np.exp(-E_S1/FESTIM.k_B/temp(0, 0.5))
        S_right = S_02*np.exp(-E_S2/FESTIM.k_B/temp(1, 0.5))

        # Test that the BCs can be applied to a problem
        # and gives the correct values
        fenics.solve(F == 0, u, bcs[0])
        assert np.isclose(
            u(0.25, 0.5),
            (phi*R_p/D_left + (phi/K_left)**0.5)/S_left)
        fenics.solve(F == 0, u, bcs[1])
        assert np.isclose(
            u(0.25, 0.5),
            (phi*R_p/D_right + (phi/K_right)**0.5)/S_right)
