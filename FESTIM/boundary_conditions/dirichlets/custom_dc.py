from FESTIM import DirichletBC, BoundaryConditionExpression
import fenics as f
import sympy as sp


class CustomDirichlet(DirichletBC):
    def __init__(self, surfaces, function, component=0, **prms) -> None:
        super().__init__(surfaces, component=component)
        self.function = function
        self.prms = prms
        self.convert_prms()

    def create_expression(self, T):
        value_BC = BoundaryConditionExpression(
            T, self.function,
            **self.prms,
        )
        self.expression = value_BC
        self.sub_expressions = self.prms.values()

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.prms.items():
            if isinstance(value, (int, float)):
                self.prms[key] = f.Constant(value)
            else:
                self.prms[key] = f.Expression(sp.printing.ccode(value),
                                       t=0,
                                       degree=1)
