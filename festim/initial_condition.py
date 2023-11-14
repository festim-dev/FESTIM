import festim as F
import ufl
from dolfinx import fem


class InitialCondition:
    """
    Initial condition class

    Args:
        volume (festim.VolumeSubdomain1D): the volume subdomain where the condition is applied
        value (float, int, fem.Constant or callable): the value of the initial condition
        species (festim.Species): the species to which the condition is applied

    Attributes:
        volume (festim.VolumeSubdomain1D): the volume subdomains where the condition is applied
        value (float, int, fem.Constant or callable): the value of the initial condition
        species (festim.Species): the species to which the source is applied
        value_fenics (fem.Function or fem.Constant): the value of the initial condition in
            fenics format

    Usage:
        >>> from festim import InitialCondition
        >>> InitialCondition(volume=my_vol, value=1, species="H")
        >>> InitialCondition(volume=my_vol, value=lambda x: 1 + x[0], species="H")
        >>> InitialCondition(volume=my_vol, value=lambda t: 1 + t, species="H")
        >>> InitialCondition(volume=my_vol, value=lambda T: 1 + T, species="H")
        >>> InitialCondition(volume=my_vol, value=lambda x, t: 1 + x[0] + t, species="H")
    """

    def __init__(self, value, volume, species):
        self.value = value
        self.volume = volume
        self.species = species

        self.value_fenics = None

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        # check that volume is festim.VolumeSubdomain1D
        if not isinstance(value, F.VolumeSubdomain1D):
            raise TypeError("volume must be of type festim.VolumeSubdomain1D")
        self._volume = value

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        # check that species is festim.Species or list of festim.Species
        if not isinstance(value, F.Species):
            raise TypeError("species must be of type festim.Species")

        self._species = value

    def create_initial_condition(self, mesh, temperature):
        """Creates the value of the initial condition as a fenics object and interpolates
        the value to the solution of the species.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a fenics.Function and the
        expression of the function is interpolated to the solution of the species.

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            temperature (float): the temperature
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.species.prev_solution.interpolate(
                F.as_fenics_constant(mesh=mesh, value=self.value)
            )

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames
            kwargs = {}
            if "t" in arguments:
                raise ValueError("Initial condition cannot be a function of time.")
            if "x" in arguments:
                kwargs["x"] = x
            if "T" in arguments:
                kwargs["T"] = temperature

            condition_expr = fem.Expression(
                self.value(**kwargs),
                self.species.prev_solution.function_space.element.interpolation_points(),
            )
            self.species.prev_solution.interpolate(condition_expr)
