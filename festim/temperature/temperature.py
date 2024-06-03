import sympy as sp
import fenics as f


class Temperature:
    """
    Class for Temperature in FESTIM

    Args:
        value (sp.Add, int, float, optional): The value of the temperature.
            Defaults to None.

    Attributes:
        T (fenics.Function): the function attributed with temperature
        T_n (fenics.Function): the previous function
        value (sp.Add, int, float): the expression of temperature
        expression (fenics.Expression): the expression of temperature as a
            fenics object

    Usage:
        >>> import festim as F
        >>> my_model = F.Simulation(...)
        >>> my_model.T = F.Temperature(300 + 10 * F.x + F.t)
    """

    def __init__(self, value=None) -> None:
        self.T = None
        self.T_n = None
        self.value = value
        self.expression = None

    def create_functions(self, mesh):
        """Creates functions self.T, self.T_n

        Args:
            mesh (festim.Mesh): the mesh
        """
        V = f.FunctionSpace(mesh.mesh, "CG", 1)
        self.T = f.Function(V, name="T")
        self.T_n = f.Function(V, name="T_n")
        self.expression = f.Expression(sp.printing.ccode(self.value), t=0, degree=2)
        self.T.assign(f.interpolate(self.expression, V))
        self.T_n.assign(self.T)

    def update(self, t):
        """Updates T_n, expression, and T with respect to time

        Args:
            t (float): the time
        """
        self.T_n.assign(self.T)
        self.expression.t = t
        self.T.assign(f.interpolate(self.expression, self.T.function_space()))

    def is_steady_state(self):
        return "t" not in sp.printing.ccode(self.value)
