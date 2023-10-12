import festim as F
from dolfinx import fem
import numpy as np
import pytest
import ufl

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)


@pytest.mark.parametrize(
    "value", [1, fem.Constant(test_mesh.mesh, 1.0), 1.0, "coucou", 2 * x[0]]
)
def test_temperature_type_and_processing(value):
    """Test that the temperature type is correctly set"""
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh

    if not isinstance(value, (fem.Constant, int, float)):
        with pytest.raises(TypeError):
            my_model.temperature = value
    else:
        my_model.temperature = value
        assert isinstance(my_model.temperature, fem.Constant)
