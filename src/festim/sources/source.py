from fenics import Constant, Expression, Function, UserExpression
import sympy as sp


class Source:
    """
    Volumetric source term.

    Args:
        value (sympy.Expr, float, int, fenics.Expression, fenics.UserExpression, fenics.Function): the value of the
            volumetric source term
        volume (int): the volume in which the source is applied
        field (str): the field on which the source is applied ("0",
            "solute", "1", "T")

    Attributes:
        value (fenics.Expression, fenics.UserExpression, fenics.Constant): the
            value of the volumetric source term
        volume (int): the volume in which the source is applied
        field (str): the field on which the source is applied ("0", "solute",
            "1", "T")
    """

    def __init__(self, value, volume, field) -> None:
        self.volume = volume
        self.field = field

        if isinstance(value, (float, int)):
            self.value = Constant(value)
        elif isinstance(value, sp.Expr):
            self.value = Expression(sp.printing.ccode(value), t=0, degree=2)
        elif isinstance(value, (Expression, UserExpression, Function)):
            self.value = value
