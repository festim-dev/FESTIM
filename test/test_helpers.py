import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem

import festim as F

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)


@pytest.mark.parametrize(
    "value",
    [
        1,
        fem.Constant(test_mesh.mesh, default_scalar_type(1.0)),
        1.0,
        "coucou",
        2 * x[0],
    ],
)
def test_temperature_type_and_processing(value):
    """Test that the temperature type is correctly set"""

    if not isinstance(value, (fem.Constant, int, float)):
        with pytest.raises(TypeError):
            F.as_fenics_constant(value, test_mesh.mesh)
    else:
        assert isinstance(F.as_fenics_constant(value, test_mesh.mesh), fem.Constant)
