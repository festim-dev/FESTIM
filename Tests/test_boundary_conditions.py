import FESTIM
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
    F, expressions = FESTIM.boundary_conditions.apply_fluxes(
        parameters, solutions, testfunctions, fenics.ds, T, S)
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
    F, expressions = FESTIM.boundary_conditions.apply_fluxes(
        {"boundary_conditions": boundary_conditions}, solutions,
        testfunctions, fenics.ds, T)
    expected_form = 0
    expected_form += -test_sol * (-Kr_0 * fenics.exp(-E_Kr/k_B/T) *
                                  sol**order)*fenics.ds(1)
    expected_form += -test_sol*expressions[0]*fenics.ds(1)
    expected_form += -test_sol*expressions[0]*fenics.ds(2)

    assert expected_form.equals(F) is True


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
        FESTIM.boundary_conditions.apply_boundary_conditions(
            parameters, V, [vm, sm], temp)

    F = fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx

    for i in range(0, 3):
        temp.t = i
        expressions[0]._bci.t = i

        # Test that the expression is correct at vertices
        expr = fenics.interpolate(expressions[0], V)
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
