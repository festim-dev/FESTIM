from festim import FluxBC
from fenics import *
import sympy as sp


class SurfaceKinetics(FluxBC):
    r"""
    FluxBC subclass allowing to include surface processes in 1D H transport simulations:

    .. math::

        \dfrac{d c_{\mathrm{s}}}{dt} = J_{\mathrm{bs}} - J_{\mathrm{sb}} + J_{\mathrm{vs}};

    .. math::

        -D \nabla c_\mathrm{m} \cdot \mathbf{n} = \lambda_{\mathrm{IS}} \dfrac{\partial c_{\mathrm{m}}}{\partial t}
        + J_{\mathrm{bs}} - J_{\mathrm{sb}},

    where :math:`c_{\mathrm{m}}` is the concentration of mobile hydrogen (:math:`\mathrm{H \ m}^{-3}`),
    :math:`c_{\mathrm{s}}` is the surface concentration of adsorbed hydrogen (:math:`\mathrm{H \ m}^{-2}`),
    the H flux from subsurface to surface :math:`J_{\mathrm{bs}}` (in :math:`\mathrm{m}^{-2} \ \mathrm{s}^{-1}`) is:

    .. math::

         J_{\mathrm{bs}} = k_{\mathrm{bs}} c_{\mathrm{m}} \lambda_{\mathrm{abs}} \left(1 - \dfrac{c_\mathrm{s}}{n_{\mathrm{surf}}}\right),

    the H flux from surface to subsurface :math:`J_{\mathrm{sb}}` (in :math:`\mathrm{m}^{-2} \ \mathrm{s}^{-1}`) is:

    .. math::

         J_{\mathrm{sb}} = k_{\mathrm{sb}} c_{\mathrm{s}} \left(1 - \dfrac{c_{\mathrm{m}}}{n_\mathrm{IS}}\right),

    :math:`\lambda_{\mathrm{abs}}=n_{\mathrm{surf}}/n_{\mathrm{IS}}` is the characteristic distance between surface and
    subsurface sites (:math:`\mathrm{m}`).

    For more details see:
        E.A. Hodille et al 2017 Nucl. Fusion 57 056002; Y. Hamamoto et al 2020 Nucl. Mater. Energy 23 100751

    .. warning::

        The SurfaceKinetics boundary condition can be used only in 1D simulations!

    Args:
        k_sb (float or callable): rate constant for the surface-to-subsurface transition (:math:`\mathrm{s}^{-1}`),
            can accept additional parameters (see example)
        k_bs (float or callable): rate constant for the subsurface-to-surface transition (:math:`\mathrm{s}^{-1}`),
            can accept additional parameters (see example)
        lambda_IS (float): characteristic distance between two iterstitial sites (:math:`\mathrm{m}`)
        n_surf (float): surface concentration of adsorption sites (:math:`\mathrm{m}^{-2}`)
        n_IS (float): bulk concentration of interstitial sites (:math:`\mathrm{m}^{-3}`)
        J_vs (float or callable): the net adsorption flux from vacuum to surface (:math:`\mathrm{m}^{-2} \ \mathrm{s}^{-1}`),
            can accept additional parameters (see example)
        surfaces (int or list): the surfaces for which surface processes are considered
        initial_condition (int or float): the initial value of the H surface concentration (:math:`\mathrm{m}^{-2}`)

    Attributes:
        previous_solutions (list): list containing solutions (fenics.Function or ufl.Indexed)
            on each surface for "previous" timestep
        test_functions (list): list containing test functions (fenics.TestFunction or ufl.Indexed)
            for each surface
        post_processing_solutions (list): list containing solutions (fenics.Function or ufl.Indexed)
            on each surface used for post-processing

    Example::

        def K_sb(T, surf_conc, prm1, prm2):
            return 1e13 * f.exp(-2.0/F.k_B/T)

        def K_bs(T, surf_conc, prm1, prm2):
            return 1e13 * f.exp(-0.2/F.k_B/T)

        def J_vs(T, surf_conc, prm1, prm2):
            return (1-surf_conc/5) ** 2 * fenics.exp(-2/F.k_B/T) + prm1 * prm2

        my_surf_model = SurfaceKinetics(
            k_sb=K_sb,
            k_bs=K_bs,
            lambda_IS=110e-12,
            n_surf=2e19,
            n_IS=6e28,
            J_vs=J_vs,
            surfaces=[1, 2],
            initial_condition=0,
            prm1=2e16,
            prm2=F.t
        )
    """

    def __init__(
        self,
        k_sb,
        k_bs,
        lambda_IS,
        n_surf,
        n_IS,
        J_vs,
        surfaces,
        initial_condition,
        **prms
    ) -> None:
        super().__init__(surfaces=surfaces, field=0)
        self.k_sb = k_sb
        self.k_bs = k_bs
        self.J_vs = J_vs
        self.lambda_IS = lambda_IS
        self.n_surf = n_surf
        self.n_IS = n_IS
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
            solute (fenics.Function or ufl.Indexed): mobile solution for "current"
                timestep
            solute_prev (fenics.Function or ufl.Indexed): mobile solution for
                "previous" timestep
            solute_test_function (fenics.TestFunction or ufl.Indexed): mobile test function
            T (festim.Temperature): the temperature of the simulation
            ds (fenics.Measure): the ds measure of the sim
            dt (festim.Stepsize): the step-size
        """

        lambda_IS = self.lambda_IS
        n_surf = self.n_surf
        n_IS = self.n_IS
        lambda_abs = (
            n_surf / n_IS
        )  # characteristic distance between surface and subsurface sites
        self.form = 0

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

            J_sb = k_sb * self.solutions[i] * (1 - solute / n_IS)
            J_bs = k_bs * (solute * lambda_abs) * (1 - self.solutions[i] / n_surf)

            if dt is not None:
                # Surface concentration form
                self.form += (
                    (self.solutions[i] - self.previous_solutions[i])
                    / dt.value
                    * self.test_functions[i]
                    * ds(surf)
                )
                # Flux to solute species
                self.form += (
                    lambda_IS
                    * (solute - solute_prev)
                    / dt.value
                    * solute_test_function
                    * ds(surf)
                )

            self.form += -(J_vs + J_bs - J_sb) * self.test_functions[i] * ds(surf)
            self.form += (J_bs - J_sb) * solute_test_function * ds(surf)

        self.sub_expressions += [expression for expression in self.prms.values()]

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.prms.items():
            if isinstance(value, (int, float)):
                self.prms[key] = Constant(value)
            else:
                self.prms[key] = Expression(sp.printing.ccode(value), t=0, degree=1)
