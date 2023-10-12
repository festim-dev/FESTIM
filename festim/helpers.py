import festim as F
from dolfinx import fem


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
