from fenics import *
import sympy as sp
import FESTIM


def initialising_solutions(parameters, V, S=None):
    '''
    Returns the prievious solutions Function() objects for formulation
    and initialise them (0 by default).
    Arguments:
    - parameters: list, contains values and components
    - V: fenics.FunctionSpace(), function space of concentrations
    - S=None: fenics.UserExpression(), solubility
    Returns:
    - u_n: fenics.Function(), previous solution
    - components: list, components of u_n
    '''
    print('Defining initial values')
    u_n, components = FESTIM.functionspaces_and_functions.define_functions(V)
    if "initial_conditions" in parameters.keys():
        initial_conditions = parameters["initial_conditions"]
    else:
        initial_conditions = []
    check_no_duplicates(initial_conditions)

    for ini in initial_conditions:
        if 'component' not in ini.keys():
            ini["component"] = 0
        if type(ini['value']) == str and ini['value'].endswith(".xdmf"):

            if V.num_sub_spaces() > 0:
                comp = Function(V.sub(ini["component"]).collapse())
            else:
                comp = Function(V)
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
        chemical_pot = False
        if S is not None:  # Is multiplication by S needed ?
            for mat in parameters["materials"]:
                if "S_0" in mat.keys() or "E_S" in mat.keys():
                    chemical_pot = True
        if ini["component"] == 0 and chemical_pot is True:
            comp = comp/S  # variable change
        if V.num_sub_spaces() > 0:
            if ini["component"] == 0 and chemical_pot is True:
                # Product must be projected
                comp = project(comp, V.sub(ini["component"]).collapse())
            else:
                comp = interpolate(comp, V.sub(ini["component"]).collapse())
            assign(u_n.sub(ini["component"]), comp)
        else:
            if ini["component"] == 0 and chemical_pot is True:
                u_n = project(comp, V)
            else:
                u_n = interpolate(comp, V)

    components = split(u_n)
    return u_n, components


def initialising_extrinsic_traps(W, number_of_traps):
    '''
    Returns a list of Function(W)
    Arguments:
    - W: FunctionSpace, functionspace of the extrinsic traps
    - number_of_traps: int, number of traps
    Returns:
    - previous_solutions: list, contains fenics.Function()
    '''
    previous_solutions = []
    for i in range(number_of_traps):
        ini = Expression("0", degree=2)
        previous_solutions.append(interpolate(ini, W))
    return previous_solutions


def check_no_duplicates(initial_conditions):
    components = []
    for e in initial_conditions:
        if "component" not in e:
            comp = 0
        else:
            comp = e["component"]
        if comp in components:
            raise ValueError("Duplicate component " + str(comp))
        else:
            components.append(comp)
    return
