from FESTIM.solving import solve_it, solve_once
import FESTIM
import fenics
import pytest


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
    dt = FESTIM.Stepsize(initial_value=0.5, stepsize_change_ratio=1.1, t_stop=430, stepsize_stop_max=0.5)

    my_settings = FESTIM.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        final_time=500,
        )
    # run & test
    solve_it(F, u, bcs, t, dt, my_settings, J=J)


def test_solve_once_jacobian_is_none():
    """Checks that solve_once() works when the jacobian (J) is None (defaults)
    """
    # build
    mesh = fenics.UnitIntervalMesh(8)
    V = fenics.FunctionSpace(mesh, "CG", 1)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)
    F = (u-u_n)*v*fenics.dx + 1*v*fenics.dx + \
        fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx

    bcs = []
    my_settings = FESTIM.Settings(absolute_tolerance=1e-10, relative_tolerance=1e-10, maximum_iterations=50)
    # run
    nb_it, converged = solve_once(F, u, bcs, my_settings)

    # test
    assert converged


def test_solve_once_returns_false():
    """Checks that solve_once() returns False when didn't converge
    """
    # build
    mesh = fenics.UnitIntervalMesh(8)
    V = fenics.FunctionSpace(mesh, "CG", 1)
    u = fenics.Function(V)
    u_n = fenics.Function(V)
    v = fenics.TestFunction(V)
    F = (u-u_n)*v*fenics.dx + 1*v*fenics.dx + \
        fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx

    bcs = []
    my_settings = FESTIM.Settings(absolute_tolerance=1e-20, relative_tolerance=1e-20, maximum_iterations=1)
    # run
    nb_it, converged = solve_once(F, u, bcs, my_settings, J=None)

    # test
    assert not converged
