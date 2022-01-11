import FESTIM


class Material:
    def __init__(self, id, D_0, E_D, S_0=None, E_S=None, thermal_conductivity=None, heat_capacity=None, density=None, borders=[]) -> None:
        self.id = id
        self.D_0 = D_0
        self.E_D = E_D
        self.S_0 = S_0
        self.E_S = E_S
        self.thermal_conductivity = thermal_conductivity
        self.heat_capacity = heat_capacity
        self.density = density
        self.borders = borders
