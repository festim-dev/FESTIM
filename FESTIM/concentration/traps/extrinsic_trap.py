from FESTIM import Trap
from fenics import *
import sympy as sp


class ExtrinsicTrap(Trap):
    def __init__(self, k_0, E_k, p_0, E_p, materials, form_parameters,
                 id=None):
        """Inits ExtrinsicTrap

        Args:
            E_k (float, list): trapping pre-exponential factor (m3 s-1)
            k_0 (float, list): trapping activation energy (eV)
            p_0 (float, list): detrapping pre-exponential factor (s-1)
            E_p (float, list): detrapping activation energy (eV)
            materials (list, int): the materials ids the trap is living in
            form_parameters (dict): dict with keys ["phi_0", "n_amax",
                "n_bmax", "eta_a", "eta_b", "f_a", "f_b"]
            id (int, optional): The trap id. Defaults to None.
        """
        super().__init__(k_0, E_k, p_0, E_p, materials, density=None, id=id)
        self.form_parameters = form_parameters
        self.convert_prms()
        self.density_previous_solution = None
        self.density_test_function = None

    def convert_prms(self):
        """Converts all the form parameters into fenics.Expression or
        fenics.Constant
        """
        # create Expressions or Constant for all parameters
        for key, value in self.form_parameters.items():
            if isinstance(value, (int, float)):
                self.form_parameters[key] = Constant(value)
            else:
                self.form_parameters[key] = Expression(sp.printing.ccode(value),
                                                       t=0, degree=1)
                self.sub_expressions.append(self.form_parameters[key])

    def create_form_density(self, dx, dt, T):
        """
        Creates the variational formulation for the extrinsic trap density.

        Args:
            dx (fenics.Measure): the dx measure of the sim
            dt (FESTIM.Stepsize): the stepsize of the simulation.
            T (FESTIM.Temperature): the temperature of the
                simulation

        Notes:
            T is an argument, although is not used in the formulation of
            extrinsic traps, but potential for subclasses of extrinsic traps
        """
        phi_0 = self.form_parameters["phi_0"]
        n_amax = self.form_parameters["n_amax"]
        n_bmax = self.form_parameters["n_bmax"]
        eta_a = self.form_parameters["eta_a"]
        eta_b = self.form_parameters["eta_b"]
        f_a = self.form_parameters["f_a"]
        f_b = self.form_parameters["f_b"]
        density = self.density[0]
        F = ((density - self.density_previous_solution)/dt.value) * \
            self.density_test_function*dx
        F += -phi_0*(
            (1 - density/n_amax)*eta_a*f_a +
            (1 - density/n_bmax)*eta_b*f_b) \
            * self.density_test_function*dx
        self.form_density = F
