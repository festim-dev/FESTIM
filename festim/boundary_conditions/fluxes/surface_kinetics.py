from festim import FluxBC
from fenics import *
import sympy as sp


class SurfaceKinetics(FluxBC):
    """ """

    def __init__(
        self, k_sb, k_bs, l_abs, N_s, N_b, J_vs, surfaces, initial_condition, **prms
    ):
        super().__init__(surfaces=surfaces, field=0)
        self.k_sb = k_sb
        self.k_bs = k_bs
        self.J_vs = J_vs
        self.l_abs = l_abs
        self.N_s = N_s
        self.N_b = N_b
        self.J_vs = J_vs
        self.initial_condition = initial_condition
        self.prms = prms
        self.convert_prms()

        self.solutions = [None] * len(self.surfaces)
        self.previous_solutions = [None] * len(self.surfaces)
        self.test_functions = [None] * len(self.surfaces)
        self.post_processing_solutions = [None] * len(self.surfaces)

    def create_form(self, solute, solute_prev, solute_test_function, T, ds, dt):
        """Creates the general form associated with the surface species
        d c_s/ dt = k_bs l_abs c_m (1 - c_s/N_s) - k_sb c_s (1 - c_b/N_b) + J_vs

        Args:
            solution (fenics.Function or ufl.Indexed): mobile solution for "current"
                timestep
            previous_solution (fenics.Function or ufl.Indexed): mobile solution for
                "previous" timestep
            test_function (fenics.TestFunction or ufl.Indexed): mobile test function
            T (festim.Temperature): the temperature of the simulation
            ds (fenics.Measure): the ds measure of the sim
            dt (festim.Stepsize): the step-size
        """

        l_abs = self.l_abs
        N_s = self.N_s
        N_b = self.N_b
        self.F = 0

        for i, surf in enumerate(self.surfaces):

            J_vs = self.J_vs
            if callable(J_vs):
                J_vs = J_vs(self.solutions[i], T.T, **self.prms)
            k_sb = self.k_sb
            if callable(k_sb):
                k_sb = k_sb(self.solutions[i], T.T, **self.prms)
            k_bs = self.k_bs
            if callable(k_bs):
                k_bs = k_bs(self.solutions[i], T.T, **self.prms)

            J_sb = k_sb * self.solutions[i] * (1 - solute / N_b)
            J_bs = k_bs * (solute * l_abs) * (1 - self.solutions[i] / N_s)

            if dt is not None:
                # Surface concentration form
                self.F += (
                    (self.solutions[i] - self.previous_solutions[i])
                    / dt.value
                    * self.test_functions[i]
                    * ds(surf)
                )
                # Flux to solute species
                self.F += (
                    -l_abs
                    * (solute - solute_prev)
                    / dt.value
                    * solute_test_function
                    * ds(surf)
                )

            self.F += -(J_vs + J_bs - J_sb) * self.test_functions[i] * ds(surf)
            self.F += (J_bs - J_sb) * solute_test_function * ds(surf)

        self.sub_expressions += [expression for expression in self.prms.values()]

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.prms.items():
            if isinstance(value, (int, float)):
                self.prms[key] = Constant(value)
            else:
                self.prms[key] = Expression(sp.printing.ccode(value), t=0, degree=1)
