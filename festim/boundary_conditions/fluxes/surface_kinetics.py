from festim import FluxBC
from fenics import *
import sympy as sp


class SurfaceKinetics(FluxBC):
    """
    FluxBC subclass allowing to include surface processes on in 1D H transport simulations.

    d c_s / dt = k_bs l_abs c_m (1 - c_s/N_s) - k_sb c_s (1 - c_b/N_b) + J_vs

    -D * grad(c) * n = -l_abs * d c / dt - k_bs l_abs c_m (1 - c_s/N_s) + k_sb c_s (1 - c_b/N_b)

    .. warning::

        The SurfaceKinetics boundary condition can be used only in 1D simulations

    Args:
        k_sb (float, callable): attempt frequency for surface-to-subsurface transition (s-1)
        k_bs (float, callable): attempt frequency for subsurface-to-surface transition (s-1)
        l_abs (float): characteristic distance between surface and subsurface sites (m)
        N_s (float): surface concentration of adsorption sites (m-2)
        N_b (float): bulk concentration of interstitial sites (m-3)
        J_vs (float, callable): the net adsorption flux from vacuum to surface (m-2 s-1),
            can accept additional parameters (see example)
        surfaces (int, list): the surfaces for which surface processes are considered
        initial_condition (int, float): the initial value of the H surface concentration (m-2)

    Example::

        def K_sb(T, surf_conc, prm1):
            return 1e13 * f.exp(-2.0/F.k_B/T)

        def K_bs(T, surf_conc, prm1):
            return 1e13 * f.exp(-0.2/F.k_B/T)

        def J_vs(T, surf_conc, prm1):
            return (1-surf_conc / 5) ** 2 * fenics.exp(-2 / T) + prm1

        my_SurfConc = SurfaceKinetics(
            k_sb = K_sb,
            k_bs = K_bs,
            l_abs = 110e-12,
            N_s = 2e19,
            N_b = 6e28,
            J_vs = J_vs,
            surfaces = [1, 2],
            initial_condition = 0,
            prm1=2e16
        )
    """

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
        """
        Creates the general form associated with the surface species

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
                J_vs = J_vs(T.T, self.solutions[i], **self.prms)
            k_sb = self.k_sb
            if callable(k_sb):
                k_sb = k_sb(T.T, self.solutions[i], **self.prms)
            k_bs = self.k_bs
            if callable(k_bs):
                k_bs = k_bs(T.T, self.solutions[i], **self.prms)

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
