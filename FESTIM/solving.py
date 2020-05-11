from FESTIM import *
from fenics import *


def solve_it(F, u, J, bcs, t, dt, solving_parameters):
    """Solves the problem during time stepping.

    Arguments:
        F {fenics.Form()} -- Formulation to be solved
        u {fenics.Function()} -- Function for concentrations
        J {fenics.Function()} -- Jacobian
        bcs {list} -- contains boundary conditions (fenics.DirichletBC())
        t {float} -- time
        dt {fenics.Constant()} -- stepsize
        solving_parameters {dict} -- solving parameters

    Returns:
        fenics.Function() -- function for concentrations
        fenics.Constant() -- stepsize
    """
    converged = False
    u_ = Function(u.function_space())
    u_.assign(u)
    while converged is False:
        u.assign(u_)
        u, nb_it, converged = solve_once(F, u, J, bcs, solving_parameters)
        if "adaptive_stepsize" in solving_parameters.keys():
            stepsize_change_ratio = \
                solving_parameters[
                    "adaptive_stepsize"]["stepsize_change_ratio"]
            dt_min = solving_parameters["adaptive_stepsize"]["dt_min"]
            adaptive_stepsize(
                nb_it=nb_it, converged=converged, dt=dt,
                stepsize_change_ratio=stepsize_change_ratio,
                dt_min=dt_min, t=t)
            if "t_stop" in solving_parameters["adaptive_stepsize"].keys():
                t_stop = solving_parameters["adaptive_stepsize"]["t_stop"]
                stepsize_stop_max = \
                    solving_parameters["adaptive_stepsize"][
                            "stepsize_stop_max"]
                if t >= t_stop:
                    if float(dt) > stepsize_stop_max:
                        dt.assign(stepsize_stop_max)
    return u, dt


def solve_once(F, u, J, bcs, solving_parameters):
    """Solves non linear problem

    Arguments:
        F {fenics.Form()} -- Formulation to be solved
        u {fenics.Function()} -- Function for concentrations
        J {fenics.Function()} -- Jacobian
        bcs {list} -- contains boundary conditions (fenics.DirichletBC())
        solving_parameters {dict} -- solving parameters

    Returns:
        fenics.Function() -- function for concentrations
        int -- number of iterations for reaching convergence
        bool -- True if converged, else False
    """
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    solver.parameters["newton_solver"]["absolute_tolerance"] = \
        solving_parameters['newton_solver']['absolute_tolerance']
    solver.parameters["newton_solver"]["relative_tolerance"] = \
        solving_parameters['newton_solver']['relative_tolerance']
    solver.parameters["newton_solver"]["maximum_iterations"] = \
        solving_parameters['newton_solver']['maximum_iterations']
    nb_it, converged = solver.solve()

    return u, nb_it, converged


def adaptive_stepsize(nb_it, converged, dt, dt_min,
                      stepsize_change_ratio, t):
    """Adapts the stepsize as function of the number of iterations of the
    solver.

    Arguments:
        nb_it {int} -- number of iterations
        converged {bool} -- True if converged, else False
        dt {fenics.Constant()} -- stepsize
        dt_min {float} -- minimum stepsize
        stepsize_change_ratio {float} -- stepsize change ratio for adaptive
            stepsize
        t {float} -- time
    """

    if converged is False:
        dt.assign(float(dt)/stepsize_change_ratio)
        if float(dt) < dt_min:
            sys.exit('Error: stepsize reached minimal value')

    if nb_it < 5:
        dt.assign(float(dt)*stepsize_change_ratio)
    else:
        dt.assign(float(dt)/stepsize_change_ratio)
    return
