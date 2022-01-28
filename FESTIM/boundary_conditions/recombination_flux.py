from FESTIM import FluxBC, k_B
import fenics as f


class RecombinationFlux(FluxBC):
    def __init__(self, Kr_0, E_Kr, order, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Kr_0 = Kr_0
        self.E_Kr = E_Kr
        self.order = order

    def create_form(self, T, solute):
        Kr = self.Kr_0*f.exp(-self.E_Kr/k_B/T)
        self.form = -Kr*solute**self.order
