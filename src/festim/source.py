import dolfinx
import numpy as np
import ufl
from dolfinx import fem

import festim as F


class SourceBase:
    """
    Source class

    Args:
        volume (festim.VolumeSubdomain1D): the volume subdomains where the source is applied
        value (float, int, fem.Constant or callable): the value of the soure
        species (festim.Species): the species to which the source is applied

    Attributes:
        volume (festim.VolumeSubdomain1D): the volume subdomains where the source is applied
        value (float, int, fem.Constant or callable): the value of the soure
        species (festim.Species): the species to which the source is applied
        value_fenics (fem.Function or fem.Constant): the value of the source in
            fenics format
        source_expr (fem.Expression): the expression of the source term that is
            used to update the value_fenics
        time_dependent (bool): True if the value of the source is time dependent
        temperature_dependent (bool): True if the value of the source is temperature
            dependent

    Usage:
        >>> from festim import Source
        >>> Source(volume=my_vol, value=1, species="H")
        >>> Source(volume=my_vol, value=lambda x: 1 + x[0], species="H")
        >>> Source(volume=my_vol, value=lambda t: 1 + t, species="H")
        >>> Source(volume=my_vol, value=lambda T: 1 + T, species="H")
        >>> Source(volume=my_vol, value=lambda x, t: 1 + x[0] + t, species="H")
    """

    def __init__(self, value, volume):
        self.value = value
        self.volume = volume

        self.value_fenics = None
        self.source_expr = None

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        # check that volume is festim.VolumeSubdomain
        if not isinstance(value, F.VolumeSubdomain):
            raise TypeError("volume must be of type festim.VolumeSubdomain")
        self._volume = value

    @property
    def value_fenics(self):
        return self._value_fenics

    @value_fenics.setter
    def value_fenics(self, value):
        if value is None:
            self._value_fenics = value
            return
        if not isinstance(
            value, (fem.Function, fem.Constant, np.ndarray, ufl.core.expr.Expr)
        ):
            raise TypeError(
                f"Value must be a dolfinx.fem.Function, dolfinx.fem.Constant, np.ndarray or a ufl.core.expr.Expr, not {type(value)}"
            )
        self._value_fenics = value

    @property
    def time_dependent(self):
        if self.value is None:
            return False
        if isinstance(self.value, fem.Constant):
            return False
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    def update(self, t):
        """Updates the source value

        Args:
            t (float): the time
        """
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            if isinstance(self.value_fenics, fem.Constant) and "t" in arguments:
                self.value_fenics.value = self.value(t=t)


class ParticleSource(SourceBase):
    def __init__(self, value, volume, species):
        self.species = species
        super().__init__(value, volume)

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        # check that species is festim.Species or list of festim.Species
        if not isinstance(value, F.Species):
            raise TypeError("species must be of type festim.Species")

        self._species = value

    @property
    def temperature_dependent(self):
        if self.value is None:
            return False
        if isinstance(self.value, fem.Constant):
            return False
        if callable(self.value):
            arguments = self.value.__code__.co_varnames
            return "T" in arguments
        else:
            return False

    def create_value_fenics(self, mesh, temperature, t: fem.Constant):
        """Creates the value of the source as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a ufl Expression

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            temperature (float): the temperature
            t (dolfinx.fem.Constant): the time
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.value_fenics = F.as_fenics_constant(mesh=mesh, value=self.value)

        elif isinstance(self.value, (fem.Function, ufl.core.expr.Expr)):
            self.value_fenics = self.value

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames

            if "t" in arguments and "x" not in arguments and "T" not in arguments:
                # only t is an argument
                if not isinstance(self.value(t=float(t)), (float, int)):
                    raise ValueError(
                        f"self.value should return a float or an int, not {type(self.value(t=float(t)))} "
                    )
                self.value_fenics = F.as_fenics_constant(
                    mesh=mesh, value=self.value(t=float(t))
                )
            else:
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x
                if "T" in arguments:
                    kwargs["T"] = temperature

                self.value_fenics = self.value(**kwargs)


class HeatSource(SourceBase):
    def __init__(self, value, volume):
        super().__init__(value, volume)

    def create_value_fenics(
        self,
        mesh: dolfinx.mesh.Mesh,
        t: fem.Constant,
    ):
        """Creates the value of the source as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a ufl.Expression

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            t (dolfinx.fem.Constant): the time
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.value_fenics = F.as_fenics_constant(mesh=mesh, value=self.value)

        elif isinstance(self.value, (fem.Function, ufl.core.expr.Expr)):
            self.value_fenics = self.value

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames

            if "t" in arguments and "x" not in arguments:
                # only t is an argument
                if not isinstance(self.value(t=float(t)), (float, int)):
                    raise ValueError(
                        f"self.value should return a float or an int, not {type(self.value(t=float(t)))} "
                    )
                self.value_fenics = F.as_fenics_constant(
                    mesh=mesh, value=self.value(t=float(t))
                )
            else:
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x
                # TODO could the source be dependend on T? why not?

                self.value_fenics = self.value(**kwargs)
