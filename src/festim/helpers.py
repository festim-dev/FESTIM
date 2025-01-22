import dolfinx
from dolfinx import fem
import numpy as np
from collections.abc import Callable
import ufl


def as_fenics_constant(
    value: float | int | fem.Constant, mesh: dolfinx.mesh.Mesh
) -> fem.Constant:
    """Converts a value to a dolfinx.Constant

    Args:
        value: the value to convert
        mesh: the mesh of the domiain

    Returns:
        The converted value

    Raises:
        TypeError: if the value is not a float, an int or a dolfinx.Constant
    """
    if isinstance(value, (float, int)):
        return fem.Constant(mesh, dolfinx.default_scalar_type(value))
    elif isinstance(value, fem.Constant):
        return value
    else:
        raise TypeError(
            f"Value must be a float, an int or a dolfinx.Constant, not {type(value)}"
        )


class ConvertToFenicsObject:
    """
    u = value

    Args:
        value: The value of the user input

    Attributes:
        value: The value of the user input
        fenics_object: The value of the user input in fenics format
        fenics_interpolation_expression: The expression of the user input that is used to
            update the `fenics_object`

    """

    input_value: (
        np.ndarray
        | fem.Constant
        | int
        | float
        | Callable[[np.ndarray], np.ndarray]
        | Callable[[np.ndarray, float], np.ndarray]
        | Callable[[float], float]
    )

    def __init__(self, input_value):
        self,
        self.input_value = input_value

        self.mapped_function = None
        self.fenics_interpolation_expression = None
        self.fenics_object = None

    @property
    def time_dependent(self) -> bool:
        """Returns true if the value given is time dependent"""
        if self.input_value is None:
            return False
        if isinstance(self.input_value, fem.Constant):
            return False
        if callable(self.input_value):
            arguments = self.input_value.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    @property
    def temperature_dependent(self) -> bool:
        """Returns true if the value given is temperature dependent"""
        if self.input_value is None:
            return False
        if isinstance(self.input_value, fem.Constant):
            return False
        if callable(self.input_value):
            arguments = self.input_value.__code__.co_varnames
            return "T" in arguments
        else:
            return False

    def as_fenics_constant(self, value, mesh: dolfinx.mesh.Mesh) -> fem.Constant:
        """Converts a value to a dolfinx.Constant

        Args:
            value: the value to convert
            mesh: the mesh of the domiain

        Raises:
            TypeError: if the value is not a float, an int or a dolfinx.Constant
        """
        if isinstance(value, (float, int)):
            self.fenics_object = fem.Constant(
                mesh, dolfinx.default_scalar_type(float(value))
            )
        else:
            raise TypeError(
                f"Value must be a float, an int or a dolfinx.Constant, not {type(value)}"
            )

    def as_mapped_function(self, mesh=None, t=None, temperature=None):
        arguments = self.input_value.__code__.co_varnames

        kwargs = {}
        if "t" in arguments:
            kwargs["t"] = t
        if "x" in arguments:
            x = ufl.SpatialCoordinate(mesh)
            kwargs["x"] = x
        if "T" in arguments:
            kwargs["T"] = temperature

        self.fenics_object = self.input_value(**kwargs)

    def as_fenics_interpolation_expression(
        self, function_space, temperature=None, t=None, mesh=None
    ):

        self.as_mapped_function(mesh=mesh, t=t, temperature=temperature)
        # store the expression
        self.fenics_interpolation_expression = fem.Expression(
            self.fenics_object,
            function_space.element.interpolation_points(),
        )

    def as_fenics_function(self, function_space, t, mesh=None, temperature=None):

        if self.fenics_interpolation_expression is None:
            self.as_fenics_interpolation_expression(
                mesh=mesh, function_space=function_space, temperature=temperature, t=t
            )

        self.fenics_object = fem.Function(function_space)
        self.fenics_object.interpolate(self.fenics_interpolation_expression)

    def convert_value(self, mesh=None, function_space=None, t=None, temperature=None):
        """Converts the value to a fenics object

        Args:
            function_space: the function space of the fenics object
            t: the time
            temperature: the temperature
        """
        if isinstance(self.input_value, fem.Constant):
            self.fenics_object = self.input_value
        elif isinstance(self.input_value, fem.Expression):
            self.fenics_interpolation_expression = self.input_value
        elif isinstance(self.input_value, (fem.Function, ufl.core.expr.Expr)):
            self.fenics_object = self.input_value
        elif isinstance(self.input_value, (float, int)):
            if mesh is None:
                raise ValueError("Mesh must be provided to create a constant")
            self.as_fenics_constant(mesh=mesh, value=self.input_value)
        elif callable(self.input_value):
            args = self.input_value.__code__.co_varnames
            # if only t is an argument, create constant object
            if "t" in args and "x" not in args and "T" not in args:
                if not isinstance(self.input_value(t=float(t)), (float, int)):
                    raise ValueError(
                        "self.value should return a float or an int, not "
                        + f"{type(self.input_value(t=float(t)))} "
                    )
                self.as_fenics_constant(mesh=mesh, value=self.input_value(t=float(t)))
            elif self.temperature_dependent:
                self.as_fenics_function(
                    mesh=mesh,
                    function_space=function_space,
                    t=t,
                    temperature=temperature,
                )
            else:
                self.as_fenics_function(mesh=mesh, function_space=function_space, t=t)
        else:
            raise TypeError(
                f"Value must be a float, an int or a callable, not {type(self.input_value)}"
            )

    def update(self, t):
        """Updates the value

        Args:
            t (float): the time
        """
        if callable(self.input_value):
            arguments = self.input_value.__code__.co_varnames
            if isinstance(self.fenics_object, fem.Constant) and "t" in arguments:
                self.fenics_object.value = float(self.input_value(t=t))
            elif isinstance(self.fenics_object, fem.Function):
                if self.fenics_interpolation_expression is not None:
                    self.fenics_object.interpolate(self.fenics_interpolation_expression)

        # if isinstance(self.fenics_object, fem.Constant):
        #     self.fenics_object.value = self.input_value(t=t)
        # elif isinstance(self.fenics_object, fem.Function):
        #     self.fenics_object.interpolate(self.fenics_interpolation_expression)
