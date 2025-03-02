from festim import FluxBC, k_B
import fenics as f
import sympy as sp


class DissociationFlux(FluxBC):
    """
    FluxBC subclass for hydrogen dissociation flux.
    -D(T) * grad(c) * n = Kd(T) * P

    Args:
        Kd_0 (float or sp.Expr): dissociation coefficient pre-exponential
            factor (m-2 s-1 Pa-1)
        E_Kd (float or sp.Expr): dissociation coefficient activation
            energy (eV)
        P (float or sp.Expr): partial pressure of H (Pa)
        surfaces (list or int): the surfaces of the BC
    """

    def __init__(self, Kd_0, E_Kd, P, surfaces) -> None:
        self.Kd_0 = Kd_0
        self.E_Kd = E_Kd
        self.P = P
        super().__init__(surfaces=surfaces, field=0)

    def create_form(self, T, solute):
        Kd_0_expr = f.Expression(sp.printing.ccode(self.Kd_0), t=0, degree=1)
        E_Kd_expr = f.Expression(sp.printing.ccode(self.E_Kd), t=0, degree=1)
        P_expr = f.Expression(sp.printing.ccode(self.P), t=0, degree=1)

        Kd = Kd_0_expr * f.exp(-E_Kd_expr / k_B / T)
        self.form = Kd * P_expr
        self.sub_expressions = [Kd_0_expr, E_Kd_expr, P_expr]
