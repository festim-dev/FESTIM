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
