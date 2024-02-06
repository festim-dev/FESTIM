from festim import SurfaceQuantity, k_B
import fenics as f
import numpy as np


class SurfaceFlux(SurfaceQuantity):
    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)
        self.title = "Flux surface {}: {}".format(self.surface, self.field)

    @property
    def prop(self):
        field_to_prop = {
            "0": self.D,
            "solute": self.D,
            0: self.D,
            "T": self.thermal_cond,
        }
        return field_to_prop[self.field]

    def compute(self, soret=False):
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


class SurfaceFluxCylindrical(SurfaceFlux):
    def __init__(self, field, surface) -> None:
        super().__init__(field, surface)
        self.r = None

    def compute(self, soret=False):
        if soret:
            raise NotImplementedError(
                "Soret effect not implemented for cylindrical coordinates"
            )

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaz = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaz[0]  # only care about r here

        # dS_z = r dr dtheta , assuming axisymmetry dS_z = theta r dr
        # dS_r = r dz dtheta , assuming axisymmetry dS_r = theta r dz
        # in both cases the expression with self.ds is the same
        # we assume full cylinder theta = 2 pi
        flux = f.assemble(
            self.prop
            * self.r
            * f.dot(f.grad(self.function), self.n)
            * self.ds(self.surface)
        )
        theta = 2 * np.pi
        flux *= theta
        return flux


class SurfaceFluxSpherical(SurfaceFlux):
    def __init__(self, field, surface) -> None:
        super().__init__(field, surface)
        self.r = None

    def compute(self, soret=False):
        if soret:
            raise NotImplementedError(
                "Soret effect not implemented for spherical coordinates"
            )

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaphi = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaphi[0]  # only care about r here

        # dS_r = r^2 sin(theta) dtheta dphi , assuming central symmetry dS_r = r^2
        flux = f.assemble(
            self.prop
            * self.r**2
            * f.dot(f.grad(self.function), self.n)
            * self.ds(self.surface)
        )
        # we assume a full sphere so multiply by 4 pi
        flux *= 4 * np.pi
        return flux
