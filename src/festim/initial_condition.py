import numpy as np
import ufl
from dolfinx import fem
import mpi4py.MPI as MPI
import adios4dolfinx

from typing import Union, Callable

from festim.helpers import get_interpolation_points


# TODO rename this to InitialConcentration and create a new base class
class InitialCondition:
    """
    Initial condition class

    Args:
        value (float, int, fem.Constant, fem.Function, or callable): the value of the initial condition.
            If a fem.Function is passed, the mesh of the function needs to match the mesh of the problem.
        species (festim.Species): the species to which the condition is applied

    Attributes:
        value (float, int, fem.Constant, fem.Function, or callable): the value of the initial condition
        species (festim.Species): the species to which the source is applied
        expr_fenics: the value of the initial condition in
            fenics format

    Examples:

        .. testsetup:: InitialCondition

            from festim import InitialCondition, Species
            my_species = Species(name='test')

        .. testcode:: InitialCondition

            InitialCondition(value=1, species=my_species)
            InitialCondition(value=lambda x: 1 + x[0], species=my_species)
            InitialCondition(value=lambda T: 1 + T, species=my_species)
            InitialCondition(value=lambda x, T: 1 + x[0] + T, species=my_species)
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
