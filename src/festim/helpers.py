import dolfinx
from dolfinx import fem


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
