from festim import FluxBC, k_B
import fenics as f
import sympy as sp


class MassFlux(FluxBC):
    """
    FluxBC subclass for advective mass flux
    -D * grad(c) * n = h_mass * (c - c_ext)

    Args:
        h_mass (float or sp.Expr): mass transfer coefficient (m/s)
        c_ext (float or sp.Expr): external concentration (1/m3)
        surfaces (list or int): the surfaces of the BC

    Reference: Bergman, T. L., Bergman, T. L., Incropera, F. P., Dewitt, D. P.,
    & Lavine, A. S. (2011). Fundamentals of heat and mass transfer. John Wiley & Sons.
    """

    def __init__(self, h_coeff, c_ext, surfaces) -> None:
        self.h_coeff = h_coeff
        self.c_ext = c_ext
        super().__init__(surfaces=surfaces, field=0)

    def create_form(self, T, solute):
        h_coeff = f.Expression(sp.printing.ccode(self.h_coeff), t=0, degree=1)
        c_ext = f.Expression(sp.printing.ccode(self.c_ext), t=0, degree=1)

        self.form = -h_coeff * (solute - c_ext)
        self.sub_expressions = [h_coeff, c_ext]
