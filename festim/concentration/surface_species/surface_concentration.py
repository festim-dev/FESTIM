from festim import Concentration, k_B
from fenics import *
import sympy as sp


class SurfaceConcentration(Concentration):
    """ """

    def __init__(
        self,
        k_sb,
        E_sb,
        k_bs,
        E_bs,
        l_abs,
        N_s,
        N_b,
        J_vs,
        surfaces,
        initial_condition,
        **prms
    ):
        super().__init__()
        self.k_sb = k_sb
        self.E_sb = E_sb
        self.k_bs = k_bs
        self.E_bs = E_bs
        self.J_vs = J_vs
        self.l_abs = l_abs
        self.N_s = N_s
        self.N_b = N_b
        if not isinstance(surfaces, list):
            surfaces = [surfaces]
        self.J_vs = J_vs
        self.surfaces = surfaces
        self.initial_condition = initial_condition
        self.prms = prms
        self.convert_prms()

        self.F = 0

    def create_form(self, mobile, T, ds, dt):
        solution = self.solution
        prev_solution = self.previous_solution
        test_function = self.test_function
        solute = mobile.solution

        k_sb = self.k_sb
        E_sb = self.E_sb
        k_bs = self.k_bs
        E_bs = self.E_bs
        l_abs = self.l_abs
        N_s = self.N_s
        N_b = self.N_b
        J_vs = self.J_vs
        if callable(J_vs):
            J_vs = J_vs(solution, T.T, **self.prms)
        if callable(E_sb):
            E_sb = E_sb(solution)
        if callable(E_bs):
            E_bs = E_bs(solution)

        J_sb = k_sb * solution * (1 - solute / N_b) * exp(-E_sb / k_B / T.T)
        J_bs = k_bs * (solute * l_abs) * (1 - solution / N_s) * exp(-E_bs / k_B / T.T)

        surf_form = (solution - prev_solution) / dt.value - (
            J_vs + J_bs - J_sb
        )  # Only time-dependent?

        for surf in self.surfaces:
            self.F += surf_form * test_function * ds(surf)
        self.sub_expressions += [expression for expression in self.prms.values()]

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.prms.items():
            if isinstance(value, (int, float)):
                self.prms[key] = Constant(value)
            else:
                self.prms[key] = Expression(sp.printing.ccode(value), t=0, degree=1)
