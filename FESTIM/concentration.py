from fenics import *
from FESTIM import read_from_xdmf
import sympy as sp


class Concentration:
    """Class for concentrations (solute or traps) with attributed
    fenics.Function objects for the solution and the previous solution and a
    fenics.TestFunction

    Args:
        solution (fenics.Function or ufl.Indexed): Solution for "current"
            timestep
        previous_solution (fenics.Function or ufl.Indexed): Solution for
            "previous" timestep
        test_function (fenics.TestFunction or ufl.Indexed): test function
    """
    def __init__(self, solution=None, previous_solution=None, test_function=None):
        self.solution = solution
        self.previous_solution = previous_solution
        self.test_function = test_function
        self.sub_expressions = []
        self.F = None
        self.post_processing_solution = None  # used for post treatment

    def initialise(self, initial_condition, V):
        comp = self.get_comp(initial_condition, V)
        comp = interpolate(comp, V)
        assign(self.previous_solution, comp)

    def get_comp(self, initial_condition, V):
        if type(initial_condition['value']) == str and initial_condition['value'].endswith(".xdmf"):
            comp = read_from_xdmf(
                initial_condition['value'],
                initial_condition["label"],
                initial_condition["time_step"],
                V)
        else:
            value = initial_condition["value"]
            value = sp.printing.ccode(value)
            comp = Expression(value, degree=3, t=0)
        return comp

    def read_from_xdmf(filename, timestep, label, V):
        comp = Function(V)
        with XDMFFile(ini["value"]) as f:
            f.read_checkpoint(comp, label, timestep)