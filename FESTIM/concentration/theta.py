from FESTIM import Mobile, k_B
import fenics as f


class Theta(Mobile):
    """Class representing the "chemical potential" c/S where S is the
    solubility of the metal
    """
    def __init__(self):
        """Inits Theta
        """
        super().__init__()
        self.S = None

    def initialise(self, V, value, label=None, time_step=None):
        """Assign a value to self.previous_solution

        Args:
            V (fenics.FunctionSpace): the function space
            value (sp.Add, float, int, str): the value of the initialisation.
            label (str, optional): the label in the XDMF file. Defaults to
                None.
            time_step (int, optional): the time step to read in the XDMF file.
                Defaults to None.
        """
        comp = self.get_comp(V, value, label=label, time_step=time_step)
        # TODO this needs changing for Henry
        comp = (comp/self.S)**2
        # Product must be projected
        comp = f.project(comp, V)
        f.assign(self.previous_solution, comp)

    def get_concentration_for_a_given_material(self, material, T):
        # TODO this needs changing for Henry
        E_S = material.E_S
        S_0 = material.S_0
        # here f.DOLFIN_EPS is needed for cases when self.solution = 0
        # fenics doesn't like 0**0.5 so instead we do (0 + 1e-16)**0.5.
        # for more details see:
        # https://fenicsproject.discourse.group/t/square-root-and-natural-logarithm-not-working/5894/2
        c_0 = (self.solution + f.DOLFIN_EPS)**(1/2)*S_0*f.exp(-E_S/k_B/T.T)
        c_0_n = (self.previous_solution + f.DOLFIN_EPS)**(1/2)*S_0*f.exp(-E_S/k_B/T.T_n)
        return c_0, c_0_n

    def mobile_concentration(self):
        # TODO this needs changing for Henry
        return self.S*(self.solution)**0.5

    def post_processing_solution_to_concentration(self):
        # TODO this needs changing for Henry
        self.post_processing_solution = self.S * self.post_processing_solution**0.5
