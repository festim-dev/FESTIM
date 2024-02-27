from festim import SurfaceQuantity, k_B
import fenics as f


class SurfaceFlux(SurfaceQuantity):
    def __init__(self, field, surface) -> None:
        """
        Object to compute the flux J of a field u through a surface
        J = integral(-prop * grad(u) . n ds)
        where prop is the property of the field (D, thermal conductivity, etc)
        u is the field
        n is the normal vector of the surface
        ds is the surface measure.

        Note: this will probably won't work correctly for axisymmetric meshes.
        Use only with cartesian coordinates.

        Args:
            field (str, int):  the field ("solute", 0, 1, "T", "retention")
            surface (int): the surface id
        """
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
                * self.Q
                / (k_B * self.T**2)
                * f.dot(f.grad(self.T), self.n)
                * self.ds(self.surface)
            )
        return flux
