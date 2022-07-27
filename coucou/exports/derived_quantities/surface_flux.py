from festim import SurfaceQuantity, R
import fenics as f


class SurfaceFlux(SurfaceQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)
        self.title = "Flux surface {}: {}".format(self.surface, self.field)

    def compute(self, soret=False):
        field_to_prop = {
            "0": self.D,
            "solute": self.D,
            0: self.D,
            "T": self.thermal_cond,
        }
        self.prop = field_to_prop[self.field]
        flux = f.assemble(
            self.prop * f.dot(f.grad(self.function), self.n) * self.ds(self.surface)
        )
        if soret and self.field in [0, "0", "solute"]:
            flux += f.assemble(
                self.prop
                * self.function
                * self.H
                / (R * self.T**2)
                * f.dot(f.grad(self.T), self.n)
                * self.ds(self.surface)
            )
        return flux
