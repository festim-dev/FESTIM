from festim import BoundaryCondition
import sympy as sp
import fenics as f


class FluxBC(BoundaryCondition):
    """
    Boundary condition ensuring the gradient of the solution
    so that:
    -D * grad(c) * n = f  or -lambda * grad(T) * n = f
    depending if applied to mobile concentration or temperature

    Args:
        surfaces (list or int): the surfaces of the BC
        value (sp.Expr or float, optional): value of the flux. Defaults to
            None.
    """

    def __init__(self, surfaces, value=None, **kwargs) -> None:
        super().__init__(surfaces=surfaces, **kwargs)
        self.value = value

    def create_form(self, T, solute):
        """Creates the form for the flux

        Args:
            T (f.Function or f.Expression): Temperature
            solute (f.Function): mobile concentration of hydrogen
        """
        form = sp.printing.ccode(self.value)
        self.form = f.Expression(form, t=0, degree=2)
        self.sub_expressions.append(self.form)
