from FESTIM import *
from fenics import *
import numpy as np


def solve_it(F, u, J, bcs, t, dt, solving_parameters):
    converged = False
    u_ = Function(u.function_space())
    u_.assign(u)
    while converged is False:
        u.assign(u_)
        u, nb_it, converged = solve_once(F, u, J, bcs, solving_parameters)
        if "adaptive_stepsize" in solving_parameters.keys():
            t_stop = solving_parameters["adaptive_stepsize"]["t_stop"]
            stepsize_stop_max = \
                solving_parameters["adaptive_stepsize"]["stepsize_stop_max"]
            stepsize_change_ratio = \
                solving_parameters[
                    "adaptive_stepsize"]["stepsize_change_ratio"]
            dt_min = solving_parameters["adaptive_stepsize"]["dt_min"]
            adaptive_stepsize(
                nb_it=nb_it, converged=converged, dt=dt,
                stepsize_change_ratio=stepsize_change_ratio,
                dt_min=dt_min, t=t, t_stop=t_stop,
                stepsize_stop_max=stepsize_stop_max)
    if "times" in solving_parameters.keys():
        times = np.array(sorted(solving_parameters['times']))
        if t < times[len(times) - 1]:
            index_closest = (np.abs(times-t)).argmin()
            if t >= times[index_closest]:
                next_time = times[index_closest+1]
            else:
                next_time = times[index_closest]
            if t + float(dt) > next_time:
                dt.assign(next_time - t)
    return u, dt


def solve_once(F, u, J, bcs, solving_parameters):
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
                      stepsize_change_ratio, t, t_stop,
                      stepsize_stop_max):
    '''
    Adapts the stepsize as function of the number of iterations of the
    solver.
    Arguments:
    - solver : FEniCS NonlinearVariationalSolver
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
    if converged is False:
        dt.assign(float(dt)/stepsize_change_ratio)
        if float(dt) < dt_min:
            sys.exit('Error: stepsize reached minimal value')

    if nb_it < 5:
        dt.assign(float(dt)*stepsize_change_ratio)
    else:
        dt.assign(float(dt)/stepsize_change_ratio)
    if t > t_stop:
        if float(dt) > stepsize_stop_max:
            dt.assign(stepsize_stop_max)
    return
