import FESTIM
import sympy as sp
from fenics import *


class Temperature:
    """
    Description of Temperature

    Attributes:
        type (str): the type of temperature
        T (fenics.Function): the function attributed with temperature
        T_n (fenics.Function): the previous function
        v_T (fenics.TestFunction): the test function
        value (sp.Add, int, float): the expression of temperature
        expression (fenics.Expression): the expression of temperature as a
            fenics object
        initial_value (sp.Add, int, float): the initial value
        sub_expressions (list): contains time dependent fenics.Expression to
            be updated
        F (fenics.Form): the variational form of the heat transfer problem
        sources (list): contains FESTIM.Source objects for volumetric heat
            sources
        boundary_conditions (list): contains FESTIM.BoundaryConditions
    """
    def __init__(self, value=None) -> None:
        """Inits Temperature

        Args:
            type (str): type of temperature in "expression",
                "solve_stationary", "solve_transient"
            value (sp.Add, int, float, optional): The value of the temperature.
                Only needed if type is not "expression". Defaults to None.
            initial_value (sp.Add, int, float, optional): The initial value.
                Only needed if type is not "expression". Defaults to None.
        """
        # self.type = type
        self.T = None
        self.T_n = None
        self.value = value
        self.expression = None

    def create_functions(self, V):
        """Creates functions self.T, self.T_n

        Args:
            V (fenics.FunctionSpace): the function space of Temperature
        """
        self.T = Function(V, name="T")
        self.T_n = Function(V, name="T_n")
        self.expression = Expression(
            sp.printing.ccode(self.value), t=0, degree=2)
        self.T.assign(interpolate(self.expression, V))
        self.T_n.assign(self.T)
