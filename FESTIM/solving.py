from FESTIM import *
from fenics import *


def solve_it(F, u, bcs, t, dt, settings, J=None):
    """Solves the problem during time stepping.

    Args:
        F (fenics.Form()): Formulation to be solved (F=0)
        u (fenics.Function()): Function for concentrations
        bcs (list): contains boundary conditions (list of fenics.DirichletBC())
        t (float): time
        dt (FESTIM.Stepsize): stepsize
        settings (FESTIM.Settings): the simulation settings
        J (fenics.Function(), optional): The jacobian. Defaults to None.
    """
    converged = False
    u_ = Function(u.function_space())
    u_.assign(u)
    while converged is False:
        u.assign(u_)
        nb_it, converged = solve_once(F, u, bcs, settings, J=J)
        if dt.adaptive_stepsize is not None:
            dt.adapt(t, nb_it, converged)
    return


def solve_once(F, u, bcs, settings, J=None):
    """Solves non linear problem

    Args:
        F (fenics.Form()): Formulation to be solved (F=0)
        u (fenics.Function()): Function for concentrations
        bcs (list): contains boundary conditions (list of fenics.DirichletBC())
        settings (FESTIM.Settings): the simulation settings
        J (fenics.Function(), optional): The jacobian. If None, it will be
            computed. Defaults to None.

    Returns:
        int, bool: number of iterations for reaching convergence, True if
            converged else False
    """

    if J is None:  # Define the Jacobian
        du = TrialFunction(u.function_space())
        J = derivative(F, u, du)
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    solver.parameters["newton_solver"]["absolute_tolerance"] = \
        settings.absolute_tolerance
    solver.parameters["newton_solver"]["relative_tolerance"] = \
        settings.relative_tolerance
    solver.parameters["newton_solver"]["maximum_iterations"] = \
        settings.maximum_iterations
    nb_it, converged = solver.solve()

    return nb_it, converged
