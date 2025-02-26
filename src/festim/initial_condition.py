import numpy as np
import ufl
from dolfinx import fem
import mpi4py.MPI as MPI
import adios4dolfinx

from typing import Union, Callable


# TODO rename this to InitialConcentration and create a new base class
class InitialCondition:
    """
    Initial condition class

    Args:
        value (float, int, fem.Constant or callable): the value of the initial condition
        species (festim.Species): the species to which the condition is applied

    Attributes:
        value (float, int, fem.Constant or callable): the value of the initial condition
        species (festim.Species): the species to which the source is applied
        expr_fenics: the value of the initial condition in
            fenics format

    Usage:
        >>> from festim import InitialCondition
        >>> InitialCondition(value=1, species=my_species)
        >>> InitialCondition(value=lambda x: 1 + x[0], species=my_species)
        >>> InitialCondition(value=lambda T: 1 + T, species=my_species)
        >>> InitialCondition(value=lambda x, T: 1 + x[0] + T, species=my_species)
    """

    expr_fenics: Union[Callable, fem.Expression]

    def __init__(self, value, species):
        self.value = value
        self.species = species

        self.expr_fenics = None

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

        if isinstance(self.value, (int, float)):
            self.expr_fenics = lambda x: np.full(x.shape[1], self.value)

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
                function_space.element.interpolation_points(),
            )


class InitialConcentrationFromFile(InitialCondition):
    expr_fenics: fem.Function

    def __init__(self, filename, species, name: str, timestamp: float):
        self.filename = filename
        self.name = name
        self.timestamp = timestamp
        super().__init__(value=None, species=species)

    def create_expr_fenics(self, mesh, temperature, function_space):

        mesh_in = adios4dolfinx.read_mesh(self.filename, MPI.COMM_WORLD)
        V_in = fem.functionspace(mesh_in, ("P", 1))
        u_in = fem.Function(V_in)
        adios4dolfinx.read_function(
            filename=self.filename,
            u=u_in,
            name=self.name,
            time=self.timestamp,
        )

        self.expr_fenics = u_in


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

        elif callable(self.value):
            arguments = self.value.__code__.co_varnames
            kwargs = {}
            if "t" in arguments:
                raise ValueError("Initial condition cannot be a function of time.")
            if "x" in arguments:
                kwargs["x"] = x

            self.expr_fenics = fem.Expression(
                self.value(**kwargs),
                function_space.element.interpolation_points(),
            )
