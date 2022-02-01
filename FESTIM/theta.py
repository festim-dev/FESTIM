from FESTIM import Mobile, k_B
import fenics as f


class Theta(Mobile):
    def __init__(self):
        super().__init__()

    def get_concentration_for_a_given_material(self, material, T):
        # TODO this needs changing for Henry
        E_S = material.E_S
        S_0 = material.S_0
        c_0 = self.solution*S_0*f.exp(-E_S/k_B/T.T)
        c_0_n = self.previous_solution*S_0*f.exp(-E_S/k_B/T.T_n)
        return c_0, c_0_n

    def get_solute_concentration(self, materials):
        # TODO this needs changing for Henry
        return self.solution*materials.S

    def convert_theta_to_concentration(self, theta, S):
        # TODO this needs changing for Henry
        return theta*S
