from festim import Trap, as_constant_or_expression


class ExtrinsicTrapBase(Trap):
    def __init__(self, k_0, E_k, p_0, E_p, materials, id=None, **kwargs):
        """Inits ExtrinsicTrap

        Args:
            E_k (float, list): trapping pre-exponential factor (m3 s-1)
            k_0 (float, list): trapping activation energy (eV)
            p_0 (float, list): detrapping pre-exponential factor (s-1)
            E_p (float, list): detrapping activation energy (eV)
            materials (list, int): the materials ids the trap is living in
            id (int, optional): The trap id. Defaults to None.
        """
        super().__init__(k_0, E_k, p_0, E_p, materials, density=None, id=id)
        for name, val in kwargs.items():
            setattr(self, name, as_constant_or_expression(val))
        self.density_previous_solution = None
        self.density_test_function = None


class ExtrinsicTrap(ExtrinsicTrapBase):
    """
    For details in the forumation see
    http://www.sciencedirect.com/science/article/pii/S2352179119300547

    Args:
        E_k (float, list): trapping pre-exponential factor (m3 s-1)
        k_0 (float, list): trapping activation energy (eV)
        p_0 (float, list): detrapping pre-exponential factor (s-1)
        E_p (float, list): detrapping activation energy (eV)
        materials (list, int): the materials ids the trap is living in
        id (int, optional): The trap id. Defaults to None.
    """

    def __init__(
        self,
        k_0,
        E_k,
        p_0,
        E_p,
        materials,
        phi_0,
        n_amax,
        n_bmax,
        eta_a,
        eta_b,
        f_a,
        f_b,
        id=None,
    ):

        super().__init__(
            k_0,
            E_k,
            p_0,
            E_p,
            materials,
            phi_0=phi_0,
            n_amax=n_amax,
            n_bmax=n_bmax,
            eta_a=eta_a,
            eta_b=eta_b,
            f_a=f_a,
            f_b=f_b,
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

        Notes:
            T is an argument, although is not used in the formulation of
            extrinsic traps, but potential for subclasses of extrinsic traps
        """
        density = self.density[0]
        F = (
            ((density - self.density_previous_solution) / dt.value)
            * self.density_test_function
            * dx
        )
        F += (
            -self.phi_0
            * (
                (1 - density / self.n_amax) * self.eta_a * self.f_a
                + (1 - density / self.n_bmax) * self.eta_b * self.f_b
            )
            * self.density_test_function
            * dx
        )
        self.form_density = F
