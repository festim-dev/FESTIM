import FESTIM
from fenics import *


class ExtrinsicTrap(FESTIM.Trap):
    def __init__(self, k_0, E_k, p_0, E_p, materials, form_parameters, id=None, type=None):
        super().__init__(k_0, E_k, p_0, E_p, materials, density=None, id=id)
        self.form_parameters = form_parameters
        self.density_previous_solution = None
        self.density_test_function = None
        self.type = type

    def convert_prms(self):
        # create Expressions or Constant for all parameters
        for key, value in self.form_parameters.items():
            if isinstance(value, (int, float)):
                self.prms[key] = Constant(value)
            else:
                self.prms[key] = Expression(sp.printing.ccode(value),
                                       t=0,
                                       degree=1)
                self.sub_expressions.append(self.prms[key])

    def create_form_density(self, dx, dt):
        phi_0 = self.form_parameters["phi_0"]
        n_amax = self.form_parameters["n_amax"]
        n_bmax = self.form_parameters["n_bmax"]
        eta_a = self.form_parameters["eta_a"]
        eta_b = self.form_parameters["eta_b"]
        f_a = self.form_parameters["f_a"]
        f_b = self.form_parameters["f_b"]
        density = self.density[0]
        F = ((density - self.density_previous_solution)/dt) * \
            self.density_test_function*dx
        F += -phi_0*(
            (1 - density/n_amax)*eta_a*f_a +
            (1 - density/n_bmax)*eta_b*f_b) \
            * self.density_test_function*dx
        self.form_density = F
