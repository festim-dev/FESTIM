from festim import Concentration, k_B
from fenics import *
import sympy as sp


class SurfaceConcentration(Concentration):
    """
    The concetration of adsorbed H species

    Args:
        k_sb (float): pre-exponential factor for surface-to-subsurface transition (m-2 s-1)
        E_sb (float, callable): activation energy for surface-to-subsurface transition (eV)
        k_bs (float): pre-exponential factor for subsurface-to-surface transition (m-2 s-1)
        E_bs (float, callable): activation energy for subsurface-to-surface transition (eV)
        l_abs (float): characteristic distance between surface and subsurface sites (m)
        N_s (float): surface concentration of adsorption sites (m-2)
        N_b (float): bulk concentration of interstitial sites (m-3)
        J_vs (float, callable): the net adsorption flux from vacuum to surface (m-2 s-1),
            can accept additional parameters (see example)
        surfaces (int, list): the surfaces for which surface processes are considered
        
    Example::

        def E_sb(surf_conc):
            return 2 * (1 - surf_conc / 5)

        def E_bs(surf_conc):
            return 2 * (1 - surf_conc / 5)

        def J_vs(surf_conc, T, prm1):
            return (1-surf_conc / 5) ** 2 * fenics.exp(-2 / T) + prm1

        my_SurfConc = festim.SurfaceConcentration(
            k_sb = 1e13,
            E_sb = E_sb,
            k_bs = 1e13,
            E_bs = E_bs,
            l_abs = 110e-12,
            N_s = 2e19,
            N_b = 6e28,
            J_vs = J_vs,
            surfaces=[1, 2],
            prm1=2e16
        )
    """

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
        """Creates the general form associated with the surface species
        d c_s/ dt = K_bs c_m (1 - c_s/N_s) - K_sb c_s (1 - c_b/N_b) + J_vs

        Args:
            mobile (festim.Mobile): the mobile concentration of the simulation
            T (festim.Temperature): the temperature of the simulation
            ds (fenics.Measure): the ds measure of the sim
            dt (festim.Stepsize): the step-size
        """

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
