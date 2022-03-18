from FESTIM import ExtrinsicTrap
from fenics import *
import sympy as sp


class CustomTrap(ExtrinsicTrap):
    def __init__(self, k_0, E_k, p_0, E_p, materials, form_parameters, id=None, type=None):
        """Inits ExtrinsicTrap
        Args:
            E_k (float): trapping pre-exponential factor
            k_0 (float): trapping activation energy
            p_0 (float): detrapping pre-exponential factor
            E_p (float): detrapping activation energy
            materials (list or int): the materials ids the trap is living in
            form_parameters (dict): dict with keys ["prm1", "prm2"]
            id (int, optional): The trap id. Defaults to None.
        """
        super().__init__(k_0, E_k, p_0, E_p, materials, form_parameters, id=None, type=None)

    def create_form_density(self, dx, dt):
        prm1 = self.form_parameters["prm1"]
        prm2 = self.form_parameters["prm2"]
        density = self.density[0]
        F = ((density - self.density_previous_solution)/dt.value) * \
            self.density_test_function*dx
        F += -prm1*(T + prm2) * self.density_test_function*dx

        self.form_density = F
