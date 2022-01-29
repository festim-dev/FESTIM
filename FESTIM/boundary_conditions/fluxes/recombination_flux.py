from FESTIM import FluxBC, k_B
import fenics as f
import sympy as sp


class RecombinationFlux(FluxBC):
    def __init__(self, Kr_0, E_Kr, order, surfaces) -> None:
        self.Kr_0 = Kr_0
        self.E_Kr = E_Kr
        self.order = order
        super().__init__(surfaces=surfaces)

    def create_form(self, T, solute):
        Kr_0_expr = f.Expression(sp.printing.ccode(self.Kr_0),
                                   t=0,
                                   degree=1)
        E_Kr_expr = f.Expression(sp.printing.ccode(self.E_Kr),
                                   t=0,
                                   degree=1)

        Kr = Kr_0_expr*f.exp(-E_Kr_expr/k_B/T)
        self.form = -Kr*solute**self.order
        self.sub_expressions = [Kr_0_expr, E_Kr_expr]
