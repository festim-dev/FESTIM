import festim
import fenics as f


def test_default_dt_min_value():
    """
    Tests that the adaptive stepsize works with a default value and that no
    error is raised
    """

    # build
    mesh = f.UnitIntervalMesh(8)
    V = f.FunctionSpace(mesh, "CG", 1)

    t = 0
    dt = festim.Stepsize(
        initial_value=0.5, stepsize_change_ratio=1.1, t_stop=430, stepsize_stop_max=0.5
    )

    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-9,
        maximum_iterations=50,
        final_time=500,
    )
    my_problem = festim.HTransportProblem(
        festim.Mobile(), festim.Traps(), festim.Temperature(200), my_settings, []
    )
    my_problem.u = f.Function(V)
    my_problem.u_n = f.Function(V)
    my_problem.v = f.TestFunction(V)
    my_problem.F = f.dot(f.grad(my_problem.u), f.grad(my_problem.v)) * f.dx

    du = f.TrialFunction(my_problem.u.function_space())
    my_problem.J = f.derivative(my_problem.F, my_problem.u, du)  # Define the Jacobian
    # run & test
    my_problem.update(t, dt)


def test_solve_once_jacobian_is_none():
    """Checks that solve_once() works when the jacobian (J) is None (defaults)"""
    # build
    mesh = f.UnitIntervalMesh(8)
    V = f.FunctionSpace(mesh, "CG", 1)

    my_settings = festim.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, maximum_iterations=50
    )
    my_problem = festim.HTransportProblem(
        festim.Mobile(), festim.Traps(), festim.Temperature(200), my_settings, []
    )
    my_problem.u = f.Function(V)
    my_problem.u_n = f.Function(V)
    my_problem.v = f.TestFunction(V)
    my_problem.F = (
        (my_problem.u - my_problem.u_n) * my_problem.v * f.dx
        + 1 * my_problem.v * f.dx
        + f.dot(f.grad(my_problem.u), f.grad(my_problem.v)) * f.dx
    )
    # run
    nb_it, converged = my_problem.solve_once()

    # test
    assert converged


def test_solve_once_returns_false():
    """Checks that solve_once() returns False when didn't converge"""
    # build
    mesh = f.UnitIntervalMesh(8)
    V = f.FunctionSpace(mesh, "CG", 1)

    my_settings = festim.Settings(
        absolute_tolerance=1e-20, relative_tolerance=1e-20, maximum_iterations=1
    )
    my_problem = festim.HTransportProblem(
        festim.Mobile(), festim.Traps(), festim.Temperature(200), my_settings, []
    )
    my_problem.u = f.Function(V)
    my_problem.u_n = f.Function(V)
    my_problem.v = f.TestFunction(V)
    my_problem.F = (
        (my_problem.u - my_problem.u_n) * my_problem.v * f.dx
        + 1 * my_problem.v * f.dx
        + f.dot(f.grad(my_problem.u), f.grad(my_problem.v)) * f.dx
    )
    # run
    nb_it, converged = my_problem.solve_once()

    # test
    assert not converged


def test_solve_once_linear_solver_mumps():
    """Checks that solve_once() works when an alternative linear solver is used rather than the default"""
    # build
    mesh = f.UnitIntervalMesh(8)
    V = f.FunctionSpace(mesh, "CG", 1)

    my_settings = festim.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        maximum_iterations=50,
        linear_solver="mumps",
    )
    my_problem = festim.HTransportProblem(
        festim.Mobile(), festim.Traps(), festim.Temperature(200), my_settings, []
    )
    my_problem.u = f.Function(V)
    my_problem.u_n = f.Function(V)
    my_problem.v = f.TestFunction(V)
    my_problem.F = (
        (my_problem.u - my_problem.u_n) * my_problem.v * f.dx
        + 1 * my_problem.v * f.dx
        + f.dot(f.grad(my_problem.u), f.grad(my_problem.v)) * f.dx
    )
    # run
    nb_it, converged = my_problem.solve_once()

    # test
    assert converged
