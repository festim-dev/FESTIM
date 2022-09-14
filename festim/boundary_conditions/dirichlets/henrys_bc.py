from festim import DirichletBC, BoundaryConditionExpression, k_B
import fenics as f
import sympy as sp


def henrys_law(T, H_0, E_H, pressure):
    H = H_0 * f.exp(-E_H / k_B / T)
    return H * pressure


class HenrysBC(DirichletBC):
    """Subclass of DirichletBC for Henry's law: cm = H*pressure

    Args:
        surfaces (list or int): the surfaces on which the BC is applied
        H_0 (float): Henry's constant pre-exponential factor (m-3.Pa-1)
        E_H (float): Henry's constant solution energy (eV)
        pressure (float or sp.Expr): hydrogen partial pressure (Pa)
    """

    def __init__(self, surfaces, H_0, E_H, pressure) -> None:
        super().__init__(surfaces, field=0, value=None)
        self.H_0 = H_0
        self.E_H = E_H
        self.pressure = pressure

    def create_expression(self, T):
        pressure = f.Expression(sp.printing.ccode(self.pressure), t=0, degree=1)
        value_BC = BoundaryConditionExpression(
            T,
            henrys_law,
            H_0=self.H_0,
            E_H=self.E_H,
            pressure=pressure,
        )
        self.expression = value_BC
        self.sub_expressions = [pressure]
