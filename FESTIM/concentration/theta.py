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
        comp = comp/self.S
        # Product must be projected
        comp = f.project(comp, V)
        f.assign(self.previous_solution, comp)

    def get_concentration_for_a_given_material(self, material, T):
        """Returns the concentration (and previous concentration) for a given
        material

        Args:
            material (FESTIM.Material): the material with attributes S_0 and
                E_S
            T (FESTIM.Temperature): the temperature with attributest T and T_n

        Returns:
            fenics.Product, fenics.Product: the current concentration and
                previous concentration
        """
        # TODO this needs changing for Henry
        E_S = material.E_S
        S_0 = material.S_0
        c_0 = self.solution*S_0*f.exp(-E_S/k_B/T.T)
        c_0_n = self.previous_solution*S_0*f.exp(-E_S/k_B/T.T_n)
        return c_0, c_0_n

    def mobile_concentration(self):
        """Returns the hydrogen concentration as c=theta*S
        Where S is FESTIM.ArheniusCoeff defines on all materials.
        This is needed when adding neuman or robin BCs to the form.

        Returns:
            fenics.Product: the hydrogen mobile concentration
        """
        # TODO this needs changing for Henry
        return self.solution*self.S

    def post_processing_solution_to_concentration(self):
        """Converts the post_processing_solution from theta to mobile
        concentration.
        c = theta * S.
        The attribute post_processing_solution is fenics.Product (if self.S is
        FESTIM.ArheniusCoeff)
        """
        # TODO this needs changing for Henry
        self.post_processing_solution *= self.S
