import FESTIM
import fenics as f
import sympy as sp


class BoundaryCondition:
    def __init__(self, surfaces, value=None, function=None, component=0, **kwargs) -> None:

        if not isinstance(surfaces, list):
            surfaces = [surfaces]
        self.surfaces = surfaces

        self.value = value
        self.function = function
        self.component = component
        self.prms = kwargs
        self.expression = None
        self.sub_expressions = []
        self.convert_prms()

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.prms.items():
            if isinstance(value, (int, float)):
                self.prms[key] = f.Constant(value)
            else:
                self.prms[key] = f.Expression(sp.printing.ccode(value),
                                       t=0,
                                       degree=1)
