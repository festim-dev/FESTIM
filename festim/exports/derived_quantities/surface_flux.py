from festim import SurfaceQuantity, k_B
import fenics as f
import numpy as np


class SurfaceFlux(SurfaceQuantity):
    """
    Computes the surface flux of a field at a given surface in cartesian coordinates

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        export_unit (str): the unit of the derived quantity in the export file
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field

    .. note::
        Object to compute the flux J of a field u through a surface
        J = integral(+prop * grad(u) . n ds)
        where prop is the property of the field (D, thermal conductivity, etc)
        u is the field
        n is the normal vector of the surface
        ds is the surface measure.
        units are in H/m2/s in 1D, H/m/s in 2D and H/s in 3D domains for hydrogen
        concentration and W/m2 in 1D, W/m in 2D and W in 3D domains for temperature

    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)

    @property
    def export_unit(self):
        # obtain domain dimension
        dim = self.function.function_space().mesh().topology().dim()
        # return unit depending on field and dimension of domain
        if self.field == "T":
            return f"W m{dim-3}".replace(" m0", "")
        else:
            return f"H m{dim-3} s-1".replace(" m0", "")

    @property
    def title(self):
        quantity_title = f"Flux surface {self.surface}: {self.field}"
        if self.show_units:
            if self.field == "T":
                quantity_title = f"Heat flux surface {self.surface}"
            else:
                quantity_title = f"{self.field} flux surface {self.surface}"

            return quantity_title + f" ({self.export_unit})"

        else:
            return quantity_title

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
    """
    Object to compute the flux J of a field u through a surface
    J = integral(-prop * grad(u) . n ds)
    where prop is the property of the field (D, thermal conductivity, etc)
    u is the field
    n is the normal vector of the surface
    ds is the surface measure in cylindrical coordinates.
    ds = r dr dtheta or ds = r dz dtheta

    .. note::
        For particle fluxes J is given in H/s, for heat fluxes J is given in W

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        azimuth_range (tuple, optional): Range of the azimuthal angle
            (theta) needs to be between 0 and 2 pi. Defaults to (0, 2 * np.pi).
    """

    def __init__(self, field, surface, azimuth_range=(0, 2 * np.pi)) -> None:
        super().__init__(field=field, surface=surface)
        self.r = None
        self.azimuth_range = azimuth_range

    @property
    def title(self):
        if self.field == "T":
            quantity_title = f"Heat flux surface {self.surface}"
        else:
            quantity_title = f"{self.field} flux surface {self.surface}"

        if self.show_units:
            if self.field == "T":
                return quantity_title + " (W)"
            else:
                return quantity_title + " (H s-1)"
        else:
            return quantity_title

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > 2 * np.pi:
            raise ValueError("Azimuthal range must be between 0 and pi")
        self._azimuth_range = value

    def compute(self, soret=False):

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaz = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaz[0]  # only care about r here

        # dS_z = r dr dtheta , assuming axisymmetry dS_z = theta r dr
        # dS_r = r dz dtheta , assuming axisymmetry dS_r = theta r dz
        # in both cases the expression with self.ds is the same

        flux = f.assemble(
            self.prop
            * self.r
            * f.dot(f.grad(self.function), self.n)
            * self.ds(self.surface)
        )
        if soret and self.field in [0, "0", "solute"]:
            flux += f.assemble(
                self.prop
                * self.r
                * self.function
                * self.Q
                / (k_B * self.T**2)
                * f.dot(f.grad(self.T), self.n)
                * self.ds(self.surface)
            )
        flux *= self.azimuth_range[1] - self.azimuth_range[0]
        return flux


class SurfaceFluxSpherical(SurfaceFlux):
    """
    Object to compute the flux J of a field u through a surface
    J = integral(-prop * grad(u) . n ds)
    where prop is the property of the field (D, thermal conductivity, etc)
    u is the field
    n is the normal vector of the surface
    ds is the surface measure in spherical coordinates.
    ds = r^2 sin(theta) dtheta dphi

    .. note::
        For particle fluxes J is given in H/s, for heat fluxes J is given in W

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        azimuth_range (tuple, optional): Range of the azimuthal angle
            (phi) needs to be between 0 and pi. Defaults to (0, np.pi).
        polar_range (tuple, optional): Range of the polar angle
            (theta) needs to be between - pi and pi. Defaults to (-np.pi, np.pi).
    """

    def __init__(
        self, field, surface, azimuth_range=(0, np.pi), polar_range=(-np.pi, np.pi)
    ) -> None:
        super().__init__(field=field, surface=surface)
        self.r = None
        self.polar_range = polar_range
        self.azimuth_range = azimuth_range

    @property
    def title(self):
        if self.field == "T":
            quantity_title = f"Heat flux surface {self.surface}"
        else:
            quantity_title = f"{self.field} flux surface {self.surface}"

        if self.show_units:
            if self.field == "T":
                return quantity_title + " (W)"
            else:
                return quantity_title + " (H s-1)"
        else:
            return quantity_title

    @property
    def polar_range(self):
        return self._polar_range

    @polar_range.setter
    def polar_range(self, value):
        if value[0] < -np.pi or value[1] > np.pi:
            raise ValueError("Polar range must be between - pi and pi")
        self._polar_range = value

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > np.pi:
            raise ValueError("Azimuthal range must be between 0 and pi")
        self._azimuth_range = value

    def compute(self, soret=False):

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaphi = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaphi[0]  # only care about r here

        # dS_r = r^2 sin(theta) dtheta dphi
        # integral(f dS_r) = integral(f r^2 sin(theta) dtheta dphi)
        #                  = (phi2 - phi1) * (-cos(theta2) + cos(theta1)) * f r^2
        flux = f.assemble(
            self.prop
            * self.r**2
            * f.dot(f.grad(self.function), self.n)
            * self.ds(self.surface)
        )
        if soret and self.field in [0, "0", "solute"]:
            flux += f.assemble(
                self.prop
                * self.r**2
                * self.function
                * self.Q
                / (k_B * self.T**2)
                * f.dot(f.grad(self.T), self.n)
                * self.ds(self.surface)
            )
        flux *= (self.polar_range[1] - self.polar_range[0]) * (
            -np.cos(self.azimuth_range[1]) + np.cos(self.azimuth_range[0])
        )
        return flux
