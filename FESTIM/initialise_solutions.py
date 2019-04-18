from fenics import *
import sympy as sp
import FESTIM


def initialising_solutions(V, initial_conditions):
    '''
    Returns the prievious solutions Function() objects for formulation
    and initialise them (0 by default).
    Arguments:
    - V: FunctionSpace(), function space of concentrations
    - initial_conditions: list, contains values and components
    '''
    print('Defining initial values')
    u_n, components = FESTIM.functionspaces_and_functions.define_functions(V)
    # initial conditions are 0 by default
    expression = ['0'] * len(components)
    for ini in initial_conditions:
        value = ini["value"]
        value = sp.printing.ccode(value)
        expression[ini["component"]] = value
    if len(expression) == 1:
        expression = expression[0]
    else:
        expression = tuple(expression)
    ini_u = Expression(expression, degree=3, t=0)
    u_n = interpolate(ini_u, V)
    components = split(u_n)
    return u_n, components


def initialising_extrinsic_traps(W, number_of_traps):
    '''
    Returns a list of Function(W)
    Arguments:
    - W: FunctionSpace, functionspace of the extrinsic traps
    - number_of_traps: int, number of traps
    '''
    previous_solutions = []
    for i in range(number_of_traps):
        ini = Expression("0", degree=2)
        previous_solutions.append(interpolate(ini, W))
    return previous_solutions
