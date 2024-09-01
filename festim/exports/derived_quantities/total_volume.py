from festim import VolumeQuantity
import fenics as f
import numpy as np


class TotalVolume(VolumeQuantity):
    """
    Computes the total value of a field in a given volume
    int(f dx)

    Args:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id

    Attributes:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        export_unit (str): the unit of the derived quantity for exporting
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the hydrogen solute field

    .. note::
        units are in H/m2 in 1D, H/m in 2D and H in 3D domains for hydrogen
        concentration and K m in 1D, K m2 in 2D and K m3 in 3D domains for temperature

    """

    def __init__(self, field, volume) -> None:
        super().__init__(field=field, volume=volume)

    @property
    def allowed_meshes(self):
        return ["cartesian"]

    @property
    def export_unit(self):
        # obtain domain dimension
        try:
            dim = self.function.function_space().mesh().topology().dim()
        except AttributeError:
            dim = self.dx._domain._topological_dimension
            # TODO we could simply do that all the time
        # return unit depending on field and dimension of domain
        if self.field == "T":
            return f"K m{dim}".replace("1", "")
        else:
            return f"H m{dim-3}".replace(" m0", "")

    @property
    def title(self):
        quantity_title = f"Total {self.field} volume {self.volume}"
        if self.show_units:
            return quantity_title + f" ({self.export_unit})"
        else:
            return quantity_title

    def compute(self):
        return f.assemble(self.function * self.dx(self.volume))


class TotalVolumeCylindrical(TotalVolume):
    """
    Computes the total value of a field in a given volume
    int(f r dx)

    Args:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        azimuth_range (tuple, optional): Range of the azimuthal angle
            (theta) needs to be between 0 and 2 pi. Defaults to (0, 2 * np.pi).

    Attributes:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        export_unit (str): the unit of the derived quantity for exporting
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the hydrogen solute field

    .. note::
        units are in H/m2 in 1D, H/m in 2D and H in 3D domains for hydrogen
        concentration and K m in 1D, K m2 in 2D and K m3 in 3D domains for temperature

    """

    def __init__(self, field, volume, azimuth_range = (0, 2*np.pi)):
        super().__init__(field=field, volume=volume)
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

    @property
    def allowed_meshes(self):
        return ["cylindrical"]
    
    def compute(self):
        theta_0, theta_1 = self.azimuth_range
        return (theta_1 - theta_0) * f.assemble(self.function * self.r * self.dx(self.volume))
    

class TotalVolumeSpherical(TotalVolume):
    """
    Computes the total value of a field in a given volume
    int(f r dx)

    Args:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        azimuth_range (tuple, optional): Range of the azimuthal angle
            (phi) needs to be between 0 and pi. Defaults to (0, np.pi).
        polar_range (tuple, optional): Range of the polar angle
            (theta) needs to be between - pi and pi. Defaults to (-np.pi, np.pi).

    Attributes:
        field (str, int): the field ("solute", 0, 1, "T", "retention")
        volume (int): the volume id
        export_unit (str): the unit of the derived quantity for exporting
        title (str): the title of the derived quantity
        show_units (bool): show the units in the title in the derived quantities
            file
        function (dolfin.function.function.Function): the solution function of
            the hydrogen solute field

    .. note::
        units are in H/m2 in 1D, H/m in 2D and H in 3D domains for hydrogen
        concentration and K m in 1D, K m2 in 2D and K m3 in 3D domains for temperature

    """

    def __init__(self, field, volume, azimuth_range = (0, np.pi), polar_range = (-np.pi, np.pi)):
        super().__init__(field=field, volume=volume)
        self.r = None
        self.azimuth_range = azimuth_range
        self.polar_range = polar_range

    @property
    def azimuth_range(self):
        return self._azimuth_range

    @azimuth_range.setter
    def azimuth_range(self, value):
        if value[0] < 0 or value[1] > np.pi:
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

    @property
    def allowed_meshes(self):
        return ["spherical"]
    
    def compute(self):
        phi_0, phi_1 = self.azimuth_range
        theta_0, theta_1 = self.polar_range

        return (phi_1 - phi_0) * (theta_1 - theta_0) * f.assemble(self.function * self.r**2 * self.dx(self.volume))