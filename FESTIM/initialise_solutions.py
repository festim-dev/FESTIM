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
        if type(ini['value']) == str and ini['value'].endswith(".xdmf"):
            comp = Function(V.sub(ini["component"]).collapse())
            if "label" not in ini.keys():
                raise KeyError("label key not found")
            if "time_step" not in ini.keys():
                raise KeyError("time_step key not found")
            with XDMFFile(ini["value"]) as file:
                file.read_checkpoint(comp, ini["label"], ini["time_step"])
            #  only works if meshes are the same
        else:
            value = ini["value"]
            value = sp.printing.ccode(value)
            comp = Expression(value, degree=3, t=0)
        if V.dim() > 1:
            comp = interpolate(comp, V.sub(ini["component"]).collapse())
        else:
            comp = interpolate(comp, V)
        assign(u_n.sub(ini["component"]), comp)
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
