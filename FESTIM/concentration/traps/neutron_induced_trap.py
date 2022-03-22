from FESTIM import ExtrinsicTrap, k_B
from fenics import *


class NeutronInducedTrap(ExtrinsicTrap):
    """
    Class for neutron induced trap creation with annealing.
    The temporal evolution of the trap density is given by:

    dn_t/dt = phi*K*(1 - n_t/n_max) + A_0*exp(-E_A/(k_B*T))*n_t

    """
    def __init__(self, k_0, E_k, p_0, E_p, materials, form_parameters,
                 id=None):
        """
        Inits NeutronInducedTrap

        Args:
            k_0 (float, list): trapping pre-exponential factor (m3 s-1)
            E_k (float, list): trapping activation energy (eV)
            p_0 (float, list): detrapping pre-exponential factor (s-1)
            E_p (float, list): detrapping activation energy (eV)
            materials (list or int): the materials ids the trap is living in
            form_parameters (dict): dict with keys ["phi", "K", "n_max",
                "A_0", "E_A"].
                phi: damage rate (dpa s-1),
                K: trap creation factor (m-3 dpa-1),
                n_max: maximum trap density (m-3),
                A_0: trap_annealing_factor (s-1),
                E_A: annealing activation energy (eV).
                All variables in form_parameters dict can be floats or
                sympy.Expr
            id (int, optional): The trap id. Defaults to None.
        """
        super().__init__(k_0, E_k, p_0, E_p, materials, form_parameters,
                         id=id)

    def create_form_density(self, dx, dt, T):
        """
        Creates the variational formulation for the extrinsic trap density.

        Args:
            dx (fenics.Measure): the dx measure of the sim
            dt (FESTIM.Stepsize): the stepsize of the simulation.
            T (FESTIM.Temperature): the temperature of the
                simulation
        """
        phi = self.form_parameters["phi"]
        K = self.form_parameters["K"]
        n_max = self.form_parameters["n_max"]
        A_0 = self.form_parameters["A_0"]
        E_A = self.form_parameters["E_A"]
        density = self.density[0]
        T = T.T

        F = ((density - self.density_previous_solution)/dt.value) * \
            self.density_test_function*dx
        F += -phi*K*(1 - (density/n_max)) * self.density_test_function*dx
        F += A_0*exp(-E_A/(k_B*T))*density * self.density_test_function*dx

        self.form_density = F
