from fenics import *
import sympy as sp
import FESTIM


def read_from_xdmf(filename, label, time_step, V):
    """Reads function from XDMF

    Args:
        filename (str): the filename
        label (str): the label of the function in the file
        time_step (int): the timestep at which the file is read
        V (fenics.FunctionSpace): The function space of the function

    Returns:
        [fenics.Function]: the function
    """
    comp = Function(V)
    with XDMFFile(filename) as f:
        f.read_checkpoint(comp, label, time_step)
        #  only works if meshes are the same
    return comp


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
