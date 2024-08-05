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

    .. note::
        Units are in H/m3 for hydrogen concentration and K for temperature

    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)

    @property
    def allowed_meshes(self):
        return ["cartesian"]

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
    int(f ds) / int (1 * ds)
    ds is the surface measure in cylindrical coordinates.
    ds = r dr dtheta

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
        r (ufl.indexed.Indexed): the radius of the cylinder

    Notes:
        Units are in H/m3 for hydrogen concentration and K for temperature
    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)
        self.r = None

    def compute(self):

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaz = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaz[0]  # only care about r here
            print(type(self.r))
            quit()

        # dS_z = r dr dtheta , assuming axisymmetry dS_z = theta r dr
        # dS_r = r dz dtheta , assuming axisymmetry dS_r = theta r dz
        # in both cases the expression with self.dx is the same

        avg_surf = f.assemble(
            self.function * self.r * self.ds(self.surface)
        ) / f.assemble(1 * self.r * self.ds(self.surface))

        return avg_surf


class AverageSurfaceSpherical(AverageSurface):
    """
    Computes the average value of a field in a given volume
    int(f ds) / int (1 * ds)
    ds is the surface measure in cylindrical coordinates.
    ds = r^2 sin(theta) dtheta dphi

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
        r (ufl.indexed.Indexed): the radius of the sphere

    Notes:
        Units are in H/m3 for hydrogen concentration and K for temperature
    """

    def __init__(self, field, surface) -> None:
        super().__init__(field=field, surface=surface)
        self.r = None

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

        avg_surf = f.assemble(
            self.function * self.r**2 * self.ds(self.surface)
        ) / f.assemble(1 * self.r**2 * self.ds(self.surface))

        return avg_surf
