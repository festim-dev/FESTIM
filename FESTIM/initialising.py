from fenics import *
import sympy as sp
import FESTIM


def initialise_solutions(parameters, V, S=None):
    """Returns the prievious solutions Function() objects for formulation
    and initialise them (0 by default)

    Arguments:
        parameters {dict} -- main parameters dict
        V {fenics.FunctionSpace} -- function space of concentrations

    Keyword Arguments:
        S {fenics.UserExpression} -- solubility (default: {None})

    Raises:
        KeyError: if label key is not found
        KeyError: if time_step key is not found

    Returns:
        fenics.Function -- previous solution
        list -- components of the previous solution
    """

    print('Defining initial values')
    u_n = Function(V)
    components = list(split(u_n))
    if "initial_conditions" in parameters.keys():
        initial_conditions = parameters["initial_conditions"]
    else:
        initial_conditions = []
    check_no_duplicates(initial_conditions)

    for ini in initial_conditions:
        if 'component' not in ini.keys():
            ini["component"] = 0
        if type(ini['value']) == str and ini['value'].endswith(".xdmf"):
            comp = read_from_xdmf(ini, V)
        else:
            value = ini["value"]
            value = sp.printing.ccode(value)
            comp = Expression(value, degree=3, t=0)

        chemical_pot = False
        if S is not None:
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

    return u_n


def read_from_xdmf(ini, V):
    """Reads component from XDMF

    Arguments:
        ini {dict} -- contains XDMF file info ("value", "label", "time_step")
            and "component"
        V {fenics.FunctionSpace} -- solution's functionspace

    Raises:
        KeyError: if label key is not found
        KeyError: if time_step key is not found

    Returns:
        [fenics.Function] -- fenics.Function(V) that will be projected
    """
    if V.num_sub_spaces() > 0:
        comp = Function(V.sub(ini["component"]).collapse())
    else:
        comp = Function(V)
    if "label" not in ini.keys():
        raise KeyError("label key not found")
    if "time_step" not in ini.keys():
        raise KeyError("time_step key not found")
    with XDMFFile(ini["value"]) as f:
        f.read_checkpoint(comp, ini["label"], ini["time_step"])
        #  only works if meshes are the same
    return comp


def initialise_extrinsic_traps(W, number_of_traps):
    """Returns a list of fenics.Function(W)

    Arguments:
        W {fenics.FunctionSpace} -- functionspace of the extrinsic traps
            densities
        number_of_traps {int} -- number of traps

    Returns:
        list -- contains fenics.Function
    """

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
