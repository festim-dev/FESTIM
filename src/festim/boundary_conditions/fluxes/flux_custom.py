from festim import FluxBC
import sympy as sp
import fenics as f


class CustomFlux(FluxBC):
    """
    FluxBC subclass allowing the use of a user-defined function.

    Args:
        surfaces (list or int): the surfaces of the BC
        function (callable): the function.

    Example::

        def fun(T, solute, param1):
            return 2*T + solute - param1

        my_bc = CustomFlux(
            surfaces=[1, 2],
            function=fun,
            param1=2*festim.x + festim.t
        )
    """

    def __init__(self, surfaces, function, field, **prms) -> None:
        super().__init__(surfaces=surfaces, field=field)
        self.function = function
        self.prms = prms
        self.convert_prms()

    def create_form(self, T, solute):
        self.form = self.function(T, solute, **self.prms)
        self.sub_expressions += [expression for expression in self.prms.values()]

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.prms.items():
            if isinstance(value, (int, float)):
                self.prms[key] = f.Constant(value)
            else:
                self.prms[key] = f.Expression(sp.printing.ccode(value), t=0, degree=1)
