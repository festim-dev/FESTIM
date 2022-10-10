from festim import ExtrinsicTrapBase, k_B
import fenics as f


class NeutronInducedTrap(ExtrinsicTrapBase):
    """
    Class for neutron induced trap creation with annealing.
    The temporal evolution of the trap density is given by

    dn_t/dt = phi*K*(1 - n_t/n_max) + A_0*exp(-E_A/(k_B*T))*n_t

    Args:
        k_0 (float, list): trapping pre-exponential factor (m3 s-1)
        E_k (float, list): trapping activation energy (eV)
        p_0 (float, list): detrapping pre-exponential factor (s-1)
        E_p (float, list): detrapping activation energy (eV)
        materials (list or int): the materials ids the trap is living in
        phi (float, sympy.Expr, f.Expression, f.UserExpression):
            damage rate (dpa s-1),
        K (float, sympy.Expr, f.Expression, f.UserExpression):
            trap creation factor (m-3 dpa-1),
        n_max (float, sympy.Expr, f.Expression, f.UserExpression):
            maximum trap density (m-3),
        A_0 (float, sympy.Expr, f.Expression, f.UserExpression):
            trap_annealing_factor (s-1),
        E_A (float, sympy.Expr, f.Expression, f.UserExpression):
            annealing activation energy (eV).
        id (int, optional): The trap id. Defaults to None.
    """

    def __init__(self, k_0, E_k, p_0, E_p, materials, phi, K, n_max, A_0, E_A, id=None):
        super().__init__(
            k_0,
            E_k,
            p_0,
            E_p,
            materials,
            phi=phi,
            K=K,
            n_max=n_max,
            A_0=A_0,
            E_A=E_A,
            id=id,
        )

    def create_form_density(self, dx, dt, T):
        """
        Creates the variational formulation for the extrinsic trap density.

        Args:
            dx (fenics.Measure): the dx measure of the sim
            dt (festim.Stepsize): the stepsize of the simulation.
            T (festim.Temperature): the temperature of the
                simulation
        """
        density = self.density[0]
        T = T.T

        F = (
            ((density - self.density_previous_solution) / dt.value)
            * self.density_test_function
            * dx
        )
        F += (
            -self.phi
            * self.K
            * (1 - (density / self.n_max))
            * self.density_test_function
            * dx
        )
        F += (
            self.A_0
            * f.exp(-self.E_A / (k_B * T))
            * density
            * self.density_test_function
            * dx
        )

        self.form_density = F
