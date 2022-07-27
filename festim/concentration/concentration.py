from fenics import *
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

    def initialise(self, V, value, label=None, time_step=None):
        """Assign a value to self.previous_solution

        Args:
            V (fenics.FunctionSpace): the function space
            value (sp.Add, float, int, str): the value of the initialisation.
            label (str, optional): the label in the XDMF file. Defaults to
                None.
            time_step (int, optional): the time step to read in the XDMF file.
                Defaults to None.
        """
        # TODO : do we need V here? can it be retrieved from self.solution?
        # not if indexed (several components)
        comp = self.get_comp(V, value, label=label, time_step=time_step)
        comp = interpolate(comp, V)
        assign(self.previous_solution, comp)

    def get_comp(self, V, value, label=None, time_step=None):
        if type(value) == str and value.endswith(".xdmf"):
            comp = Function(V)
            with XDMFFile(value) as f:
                f.read_checkpoint(comp, label, time_step)
        else:
            value = sp.printing.ccode(value)
            comp = Expression(value, degree=3, t=0)
        return comp
