from FESTIM import FluxBC, k_B
import fenics as f
import sympy as sp


class ConvectiveFlux(FluxBC):
    def __init__(self, h_coeff, T_ext, surfaces) -> None:
        self.h_coeff = h_coeff
        self.T_ext = T_ext
        super().__init__(surfaces=surfaces, component="T")

    def create_form(self, T, solute):
        h_coeff = f.Expression(sp.printing.ccode(self.h_coeff),
                                   t=0,
                                   degree=1)
        T_ext = f.Expression(sp.printing.ccode(self.T_ext),
                                   t=0,
                                   degree=1)

        # TODO check the sign here....
        self.form = h_coeff * (T - T_ext)
        self.sub_expressions = [h_coeff, T_ext]
