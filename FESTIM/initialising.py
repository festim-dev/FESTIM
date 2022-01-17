from fenics import *
import sympy as sp


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
