from FESTIM import DirichletBC, BoundaryConditionExpression, k_B
import fenics as f
import sympy as sp


def sieverts_law(T, S_0, E_S, pressure):
    S = S_0*f.exp(-E_S/k_B/T)
    return S*pressure**0.5


class SievertsBC(DirichletBC):
    def __init__(self, surfaces, S_0, E_S, pressure) -> None:
        super().__init__(surfaces, component=0)
        self.S_0 = S_0
        self.E_S = E_S
        self.pressure = pressure

    def create_expression(self, T):
        pressure = f.Expression(
            sp.printing.ccode(self.pressure),
            t=0, degree=1)
        value_BC = BoundaryConditionExpression(
            T, sieverts_law,
            S_0=self.S_0, E_S=self.E_S,
            pressure=pressure,
        )
        self.expression = value_BC
        self.sub_expressions = [pressure]
