import FESTIM


class Material:
    def __init__(self, id, D_0, E_D, S_0=None, E_S=None, thermal_cond=None, heat_capacity=None, rho=None, borders=[], H=None) -> None:
        self.id = id
        self.D_0 = D_0
        self.E_D = E_D
        self.S_0 = S_0
        self.E_S = E_S
        self.thermal_cond = thermal_cond
        self.heat_capacity = heat_capacity
        self.rho = rho
        self.borders = borders
        self.H = H
        if H is not None:
            self.free_enthalpy = H["free_enthalpy"]
            self.entropy = H["entropy"]
