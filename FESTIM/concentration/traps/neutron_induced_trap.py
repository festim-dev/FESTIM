from FESTIM import ExtrinsicTrap, k_B
from fenics import *
import sympy as sp


class NeutronInducedTrap(ExtrinsicTrap):
    def __init__(self, k_0, E_k, p_0, E_p, materials, form_parameters, id=None,
                 type=None):
        """
        Inits Neutron induced traps

        Args:
            E_k (float, list): trapping pre-exponential factor (m2/s)
            k_0 (float, list): trapping activation energy (eV)
            p_0 (float, list): detrapping pre-exponential factor (m2/s)
            E_p (float, list): detrapping activation energy (eV)
            materials (list or int): the materials ids the trap is living in
            density_parameters (dict): dict with keys ["phi", "K", "n_max",
                "A_0", "E_A", "n_0"]
            id (int, optional): The trap id. Defaults to None.

        Notes:
        For an analytical solution for trap density with the following
        formulation:

        n_t = F*n_max/(F + A*n_max) + (F*n_0 - F*n_max + A*n_0*n_max) *
            np.exp(t*(-F/n_max - A))/(F + A*n_max)
        Where:
        F = phi*K
        A = A_0*np.exp(-E_A/(k_B*T))

            phi (float, sympy.Expr): damage rate (dpa s-1)
            K (float): trap creation factor (m-3 s-1)
            n_max (float): maximum trap density (m-3)
            A_0 (float): trap_annealing_factor (s-1)
            E_A (float): annealing activation energy (eV)
            n_0 (float): number of initial traps (m-3)
        """
        super().__init__(k_0, E_k, p_0, E_p, materials, form_parameters,
                         id=None, type=None)
        self.form_parameters = form_parameters
        self.density_previous_solution = None
        self.density_test_function = None

    def create_form_density(self, dx, dt, T):
        """
        Creates the variational formulation for the extrinsic trap density.

        Args:
            dx (fenics.Measure): the dx measure of the sim
            dt (FESTIM.Stepsize): If None assuming steady state.
            T (FESTIM.Temperature): the temperature of the
                simulation
        """
        phi = self.form_parameters["phi"]
        K = self.form_parameters["K"]
        n_max = self.form_parameters["n_max"]
        A_0 = self.form_parameters["A_0"]
        E_A = self.form_parameters["E_A"]
        n_0 = self.form_parameters["n_0"]
        density = self.density[0]
        T = T.T

        F = ((density - self.density_previous_solution)/dt.value) * \
            self.density_test_function*dx
        F += -(phi*K*n_max/(phi*K + A_0*exp(-E_A/(k_B*T))*n_max) +
               (phi*K*n_0 - phi*K*n_max + A_0*exp(-E_A/(k_B*T))*n_0*n_max) *
               exp(dt.value*(-phi*K/n_max - (A_0*exp(-E_A/(k_B*T))))) /
               (phi*K + A_0*exp(-E_A/(k_B*T))*n_max)) *\
            self.density_test_function*dx

        self.form_density = F
