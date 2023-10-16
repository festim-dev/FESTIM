from dolfinx import fem
import numpy as np
from petsc4py.PETSc import ScalarType


def as_fenics_constant(value, mesh):
    """Converts a value to a dolfinx.Constant

    Args:
        value (float, int or dolfinx.Constant): the value to convert
        mesh (dolfinx.mesh.Mesh): the mesh of the domiain

    Returns:
        dolfinx.Constant: the converted value

    Raises:
        TypeError: if the value is not a float, an int or a dolfinx.Constant
    """
    if isinstance(value, (float, int)):
        return fem.Constant(mesh, float(value))
    elif isinstance(value, fem.Constant):
        return value
    else:
        raise TypeError(
            f"Value must be a float, an int or a dolfinx.Constant, not {type(value)}"
        )


class SpaceTimeDependentExpression:
    def __init__(self, function, t=0):
        self.t = t
        self.function = function
        self.values = None

    def __call__(self, x):
        if self.values is None:
            self.values = np.zeros(x.shape[1], dtype=ScalarType)
        self.values = np.full(x.shape[1], self.function(x=x, t=self.t))
        return self.values


def convert_to_appropriate_obj(object, function_space, mesh):
    """Converts a value to a dolfinx.Constant or a dolfinx.Function
    depending on the type of the value

    Args:
        object (callable or float): the value to convert
        function_space (dolfinx.fem.FunctionSpace): the function space of the domain
        mesh (dolfinx.mesh.mesh): the mesh of the domain

    Returns:
        dolfinx.Constant or dolfinx.Function: the converted value
        festim.SpaceTimeDependentExpression or None: the expression if the value is
            space and time dependent, None otherwise
    """
    if isinstance(object, (int, float)):
        # case 1 pressure isn't space dependent or only time dependent:
        return as_fenics_constant(mesh=mesh, value=object), None
    # case 2 pressure is space dependent
    elif callable(object):
        arguments = object.__code__.co_varnames
        if "t" in arguments and "x" in arguments:
            expr = SpaceTimeDependentExpression(function=object, t=0)
            fenics_obj = fem.Function(function_space)
            fenics_obj.interpolate(expr.__call__)
            return fenics_obj, expr
        elif "x" in arguments:
            fenics_obj = fem.Function(function_space)
            fenics_obj.interpolate(object)
            return fenics_obj, None

        elif "t" in arguments:
            return as_fenics_constant(mesh=mesh, value=object(t=0)), None
