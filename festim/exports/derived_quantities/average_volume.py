from festim import VolumeQuantity
import fenics as f


class AverageVolume(VolumeQuantity):
    """
    Computes the average value of a field in a given volume
    int(f dx) / int (1 * dx)

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field
        r (ufl.indexed.Indexed): the radius of the cylinder

    .. note::
        Units are in H/m3 for hydrogen concentration and K for temperature
    """

    def __init__(self, field, volume: int) -> None:
        super().__init__(field=field, volume=volume)

    @property
    def allowed_meshes(self):
        return ["cartesian"]

    @property
    def export_unit(self):
        if self.field == "T":
            return "K"
        else:
            return "H m-3"

    @property
    def title(self):
        quantity_title = f"Average {self.field} volume {self.volume}"
        if self.show_units:
            return quantity_title + f" ({self.export_unit})"
        else:
            return quantity_title

    def compute(self):
        return f.assemble(self.function * self.dx(self.volume)) / f.assemble(
            1 * self.dx(self.volume)
        )


class AverageVolumeCylindrical(AverageVolume):
    """
    Computes the average value of a field in a given volume
    int(f dx) / int (1 * dx)
    dx is the volume measure in cylindrical coordinates.
    dx = r dr dz dtheta

    Note: for particle fluxes J is given in H/s, for heat fluxes J is given in W

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id

    Attributes:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field
        r (ufl.indexed.Indexed): the radius of the sphere
    """

    def __init__(self, field, volume) -> None:
        super().__init__(field=field, volume=volume)
        self.r = None

    @property
    def allowed_meshes(self):
        return ["cylindrical"]

    def compute(self):

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaz = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaz[0]  # only care about r here

        avg_vol = f.assemble(
            self.function * self.r * self.dx(self.volume)
        ) / f.assemble(1 * self.r * self.dx(self.volume))

        return avg_vol


class AverageVolumeSpherical(AverageVolume):
    """
    Computes the average value of a field in a given volume
    int(f dx) / int (1 * dx)
    dx is the volume measure in cylindrical coordinates.
    dx = rho dtheta dphi

    Note: for particle fluxes J is given in H/s, for heat fluxes J is given in W

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the field
    """

    def __init__(self, field, volume) -> None:
        super().__init__(field=field, volume=volume)
        self.r = None

    @property
    def allowed_meshes(self):
        return ["spherical"]

    def compute(self):

        if self.r is None:
            mesh = (
                self.function.function_space().mesh()
            )  # get the mesh from the function
            rthetaphi = f.SpatialCoordinate(mesh)  # get the coordinates from the mesh
            self.r = rthetaphi[0]  # only care about r here

        avg_vol = f.assemble(
            self.function * self.r**2 * self.dx(self.volume)
        ) / f.assemble(1 * self.r**2 * self.dx(self.volume))

        return avg_vol
