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

    def create_initial_condition(self, mesh, temperature):
        """Creates the value of the initial condition.
        If the value is a float or int, it is interpolated over the prev_solution
        of the species.
        Otherwise, it is converted to a fenics.Function and the expression of the
        function is interpolated to the prev_solution of the species.

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            temperature (float): the temperature
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.species.prev_solution.x.array[:] = float(self.value)

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
