from festim import SurfaceQuantity
import fenics as f
import numpy as np


class AverageSurface(SurfaceQuantity):
    """
    Computes the average value of a field on a given surface
    int(f ds) / int (1 * ds)

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field

    Notes:
        Units are in H/m3 for hydrogen concentration and K for temperature

    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)

    @property
    def title(self):
        quantity_title = f"Average {self.field} surface {self.surface}"
        if self.show_units:
            if self.field == "T":
                return quantity_title + " (K)"
            else:
                return quantity_title + " (H m-3)"
        else:
            return quantity_title

    def compute(self):
        return f.assemble(self.function * self.ds(self.surface)) / f.assemble(
            1 * self.ds(self.surface)
        )


class AverageSurfaceCylindrical(AverageSurface):
    """
    Computes the average value of a field on a given surface
    int(f dS) / int (1 * dS)
    dS is the surface measure in cylindrical coordinates.
    dS = r dr dtheta

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field
        azimuth_range (tuple, optional): Range of the azimuthal angle
            (theta) needs to be between 0 and 2 pi. Defaults to (0, 2 * np.pi)

    Notes:
        Units are in H/m3 for hydrogen concentration and K for temperature
    """

    def __init__(self, field, surface, azimuth_range=(0, 2 * np.pi)) -> None:
        super().__init__(field=field, surface=surface)
        self.r = None
        self.azimuth_range = azimuth_range

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > 2 * np.pi:
            raise ValueError("Azimuthal range must be between 0 and pi")
        self._azimuth_range = value

    def compute(self):

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaz = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaz[0]  # only care about r here

        # dS_z = r dr dtheta , assuming axisymmetry dS_z = theta r dr
        # dS_r = r dz dtheta , assuming axisymmetry dS_r = theta r dz
        # in both cases the expression with self.dx is the same

        values = f.assemble(self.function * self.r * self.ds(self.surface)) * (
            self.azimuth_range[1] - self.azimuth_range[0]
        )

        surface_area = f.assemble(1 * self.r * self.ds(self.surface)) * (
            self.azimuth_range[1] - self.azimuth_range[0]
        )
        avg_surf = values / surface_area

        return avg_surf


class AverageSurfaceSpherical(AverageSurface):
    """
    Computes the average value of a field in a given volume
    int(f dx) / int (1 * dx)
    dx is the volume measure in cylindrical coordinates.
    dx = r dr dtheta

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        surface (int): the surface id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field
        azimuth_range (tuple, optional): Range of the azimuthal angle
            (theta) needs to be between 0 and 2 pi. Defaults to (0, 2 * np.pi)
        polar_range (tuple, optional): Range of the polar angle
            (phi) needs to be between - pi and pi. Defaults to (-np.pi, np.pi)

    Notes:
        Units are in H/m3 for hydrogen concentration and K for temperature
    """

    def __init__(
        self, field, surface, azimuth_range=(0, 2 * np.pi), polar_range=(-np.pi, np.pi)
    ) -> None:
        super().__init__(field=field, surface=surface)
        self.r = None
        self.azimuth_range = azimuth_range
        self.polar_range = polar_range

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > 2 * np.pi:
            raise ValueError("Azimuthal range must be between 0 and pi")
        self._azimuth_range = value

    @property
    def polar_range(self):
        return self._polar_range

    @polar_range.setter
    def polar_range(self, value):
        if value[0] < -np.pi or value[1] > np.pi:
            raise ValueError("Polar range must be between - pi and pi")
        self._polar_range = value

    def compute(self):

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaz = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaz[0]  # only care about r here

        # dV_z = r dr dtheta , assuming axisymmetry dV_z = theta r dr
        # dV_r = r dz dtheta , assuming axisymmetry dV_r = theta r dz
        # in both cases the expression with self.dx is the same

        values = (
            f.assemble(self.function * self.r**2 * self.ds(self.surface))
            * (self.polar_range[1] - self.polar_range[0])
            * (-np.cos(self.azimuth_range[1]) + np.cos(self.azimuth_range[0]))
        )

        surface_area = (
            f.assemble(1 * self.r**2 * self.ds(self.surface))
            * (self.polar_range[1] - self.polar_range[0])
            * (-np.cos(self.azimuth_range[1]) + np.cos(self.azimuth_range[0]))
        )

        avg_surf = values / surface_area

        return avg_surf
