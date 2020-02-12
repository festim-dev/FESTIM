import FESTIM
import fenics
import pytest
import sympy as sp
import numpy as np


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
                "surface": [1, 2]
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
