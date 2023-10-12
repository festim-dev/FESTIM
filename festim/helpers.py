import festim as F
from dolfinx import fem


def as_fenics_constant(value, mesh):
    if isinstance(value, (float, int)):
        return fem.Constant(mesh, float(value))
    elif isinstance(value, fem.Constant):
        return value
    else:
        raise TypeError(
            f"Value must be a float, an int or a dolfinx.Constant, not {type(value)}"
        )
