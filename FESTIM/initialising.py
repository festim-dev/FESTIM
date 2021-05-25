from fenics import *
import sympy as sp
import FESTIM


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
