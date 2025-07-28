from collections.abc import Callable
from typing import Union

import mpi4py.MPI as MPI

import adios4dolfinx
import numpy as np
import ufl
from dolfinx import fem

from festim.helpers import get_interpolation_points
from festim.species import Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class InitialConditionBase:
    """
    Base initial condition class

    Args:
        value: the value of the initial condition.
        volume: the volume subdomain where the initial condition is applied

    Attributes:
        value: the value of the initial condition.
        volume: the volume subdomain where the initial condition is applied
    """

    value: (
        float
        | int
        | fem.Constant
        | np.ndarray
        | fem.Expression
        | ufl.core.expr.Expr
        | fem.Function
    )
    volume: VolumeSubdomain

    def __init__(
        self,
        value: (
            float
            | int
            | fem.Constant
            | np.ndarray
            | fem.Expression
            | ufl.core.expr.Expr
            | fem.Function
        ),
        volume: VolumeSubdomain,
    ):
        self.value = value
        self.volume = volume

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        # check that volume is festim.VolumeSubdomain
        if not isinstance(value, VolumeSubdomain):
            raise TypeError("volume must be of type festim.VolumeSubdomain")
        self._volume = value


class InitialConcentration(InitialConditionBase):
    """
    Initial condition class

    Args:
        value: the value of the initial condition.
        species: the species to which the condition is applied
        volume: the volume subdomain where the initial condition is applied

    Attributes:
        value: the value of the initial condition.
        species: the species to which the condition is applied
        volume: the volume subdomain where the initial condition is applied
        expr_fenics: the value of the initial condition in fenics expr format

    Examples:

        .. testsetup:: InitialConcentration

            from festim import InitialConcentration, Species
            my_species = Species(name='test')

        .. testcode:: InitialConcentration

            InitialConcentration(value=1, species=my_species)
            InitialConcentration(value=lambda x: 1 + x[0], species=my_species)
            InitialConcentration(value=lambda T: 1 + T, species=my_species)
            InitialConcentration(value=lambda x, T: 1 + x[0] + T, species=my_species)
    """

    expr_fenics: Union[Callable, fem.Expression]

    def __init__(self, value, volume, species: Species):
        super().__init__(value=value, volume=volume)

        self.species = species

        self.expr_fenics = None

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        # check that species is festim.Species or list of festim.Species
        if not isinstance(value, Species):
            raise TypeError("species must be of type festim.Species")

        self._species = value

    def create_expr_fenics(self, mesh, temperature, function_space):
        """Creates the expr_fenics of the initial condition.
        If the value is a float or int, a function is created with an array with
        the shape of the mesh and all set to the value.
        Otherwise, it is converted to a fem.Expression.

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            temperature (float): the temperature
            function_space(dolfinx.fem.FunctionSpaceBase): the function space of the species
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, int | float):
            self.expr_fenics = lambda x: np.full(x.shape[1], self.value)
        elif isinstance(self.value, fem.Function):
            self.expr_fenics = self.value
        elif callable(self.value):
            arguments = self.value.__code__.co_varnames
            kwargs = {}
            if "t" in arguments:
                raise ValueError("Initial condition cannot be a function of time.")
            if "x" in arguments:
                kwargs["x"] = x
            if "T" in arguments:
                kwargs["T"] = temperature

            self.expr_fenics = fem.Expression(
                self.value(**kwargs),
                get_interpolation_points(function_space.element),
            )


class InitialTemperature:
    def __init__(self, value) -> None:
        self.value = value
        self.expr_fenics = None

    def create_expr_fenics(self, mesh, function_space):
        """Creates the expr_fenics of the initial condition.
        If the value is a float or int, a function is created with an array with
        the shape of the mesh and all set to the value.
        Otherwise, it is converted to a fem.Expression.

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            function_space(dolfinx.fem.FunctionSpace): the function space of the species
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.value, (int, float)):
            self.expr_fenics = lambda x: np.full(x.shape[1], self.value)
        elif isinstance(self.value, fem.Function):
            self.expr_fenics = self.value
        elif callable(self.value):
            arguments = self.value.__code__.co_varnames
            kwargs = {}
            if "t" in arguments:
                raise ValueError("Initial condition cannot be a function of time.")
            if "x" in arguments:
                kwargs["x"] = x

            self.expr_fenics = fem.Expression(
                self.value(**kwargs),
                get_interpolation_points(function_space.element),
            )


def read_function_from_file(
    filename: str, name: str, timestamp: int | float, family="P", order: int = 1
) -> fem.Function:
    """
    Read a function from a file

    note::
        The function is read from a file using adios4dolfinx. For more information
        see the [adios4dolfinx documentation](https://jsdokken.com/adios4dolfinx/README.html).

    Args:
        filename: the filename
        name: the name of the function
        timestamp: the timestamp of the function
        family: the family of the function space
        order: the order of the function space

    Returns:
        the function
    """
    mesh_in = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD)
    V_in = fem.functionspace(mesh_in, (family, order))
    u_in = fem.Function(V_in)
    adios4dolfinx.read_function(
        filename=filename,
        u=u_in,
        name=name,
        time=timestamp,
    )
    return u_in
