from FESTIM.solving import adaptive_stepsize, solve_it
import fenics
import pytest


def test_adaptive_stepsize():
    dt = fenics.Constant(1e-8)
    with pytest.raises(SystemExit):
        adaptive_stepsize(2, False, dt, 1, 1, 2)

    val = 1e-8
    dt2 = fenics.Constant(val)

    adaptive_stepsize(6, True, dt2, 1, 2, 2)
    assert float(dt2) == val/2


def test_default_dt_min_value():
    """
    Tests that the adaptive stepsize works with a default value and that no
    error is raised
    """

    # build
    mesh = fenics.UnitIntervalMesh(8)
    V = fenics.FunctionSpace(mesh, "CG", 1)
    u = fenics.Function(V)
    v = fenics.TestFunction(V)
    F = fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx

    du = fenics.TrialFunction(u.function_space())
    J = fenics.derivative(F, u, du)  # Define the Jacobian
    bcs = []
    t = 0
    dt = fenics.Constant(1)
    solving_parameters = {
        "final_time": 500,
        "initial_stepsize": 0.5,
        "adaptive_stepsize": {
            "stepsize_change_ratio": 1.1,
            "t_stop": 430,
            "stepsize_stop_max": 0.5,
        },
        "newton_solver": {
            "absolute_tolerance": 1e10,
            "relative_tolerance": 1e-9,
            "maximum_iterations": 50,
        }
    }
    # run & test
    solve_it(F, u, J, bcs, t, dt, solving_parameters)
