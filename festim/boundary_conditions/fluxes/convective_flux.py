from festim import FluxBC, k_B
import fenics as f
import sympy as sp


class ConvectiveFlux(FluxBC):
    """
    FluxBC subclass for convective heat flux
    -lambda * grad(T) * n = h_coeff * (T - T_ext)

    Args:
        h_coeff (float or sp.Expr): heat exchange coefficient (W/m2/K)
        T_ext (float or sp.Expr): fluid temperature (K)
        surfaces (list or int): the surfaces of the BC
    """

    def __init__(self, h_coeff, T_ext, surfaces) -> None:
        self.h_coeff = h_coeff
        self.T_ext = T_ext
        super().__init__(surfaces=surfaces, field="T")

    def create_form(self, T, solute):
        h_coeff = f.Expression(sp.printing.ccode(self.h_coeff), t=0, degree=1)
        T_ext = f.Expression(sp.printing.ccode(self.T_ext), t=0, degree=1)

        self.form = -h_coeff * (T - T_ext)
        self.sub_expressions = [h_coeff, T_ext]
