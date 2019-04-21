from FESTIM import *
from fenics import *


def solve_u(F, u, bcs, t, dt, solving_parameters):
    du = TrialFunction(u.function_space())
    J = derivative(F, u, du)  # Define the Jacobian
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    solver.parameters["newton_solver"]["absolute_tolerance"] = \
        solving_parameters['newton_solver']['absolute_tolerance']
    solver.parameters["newton_solver"]["relative_tolerance"] = \
        solving_parameters['newton_solver']['relative_tolerance']
    nb_it, converged = solver.solve()

    t_stop = solving_parameters["adaptive_stepsize"]["t_stop"]
    stepsize_stop_max = \
        solving_parameters["adaptive_stepsize"]["stepsize_stop_max"]
    stepsize_change_ratio = \
        solving_parameters["adaptive_stepsize"]["stepsize_change_ratio"]
    dt_min = solving_parameters["adaptive_stepsize"]["dt_min"]
    dt = adaptive_stepsize(
        converged=converged, nb_it=nb_it, dt=dt,
        stepsize_change_ratio=stepsize_change_ratio,
        dt_min=dt_min, t=t, t_stop=t_stop,
        stepsize_stop_max=stepsize_stop_max)
    return u, dt


def adaptive_stepsize(converged, nb_it, dt, dt_min,
                      stepsize_change_ratio, t, t_stop,
                      stepsize_stop_max):
    '''
    Adapts the stepsize as function of the number of iterations of the
    solver.
    Arguments:
    - converged : bool, determines if the time step has converged.
    - nb_it : int, number of iterations
    - dt : Constant(), fenics object
    - dt_min : float, stepsize minimum value
    - stepsize_change_ration : float, stepsize change ratio
    - t : float, time
    - t_stop : float, time where adaptive time step stops
    - stepsize_stop_max : float, maximum stepsize after stop
    Returns:
    - dt : Constant(), fenics object
    '''
    while converged is False:
        dt.assign(float(dt)/stepsize_change_ratio)
        nb_it, converged = solver.solve()
        if float(dt) < dt_min:
            sys.exit('Error: stepsize reached minimal value')
    if t > t_stop:
        if float(dt) > stepsize_stop_max:
            dt.assign(stepsize_stop_max)
    else:
        if nb_it < 5:
            dt.assign(float(dt)*stepsize_change_ratio)
        else:
            dt.assign(float(dt)/stepsize_change_ratio)
    return dt
