from FESTIM import BoundaryCondition
import sympy as sp
import fenics as f


class FluxBC(BoundaryCondition):
    def __init__(self, surfaces, value=None, type="flux", **kwargs) -> None:
        super().__init__(type=type, surfaces=surfaces, value=value, **kwargs)

    def create_form(self, T, solute):
        form = sp.printing.ccode(self.value)
        form = f.Expression(form, t=0, degree=2)
        self.sub_expressions.append(form)
        self.form = form
        return form
