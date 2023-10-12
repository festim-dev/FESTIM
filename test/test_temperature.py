import festim as F
from dolfinx import fem
import numpy as np
import pytest

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


@pytest.mark.parametrize(
    "test_type", [1, fem.Constant(test_mesh.mesh, 1.0), 1.0, "coucou"]
)
def test_temperature_type(test_type):
    """Test that the temperature type is correctly set"""
    my_model = F.HydrogenTransportProblem()
    if not isinstance(test_type, (fem.Constant, int, float)):
        with pytest.raises(TypeError):
            my_model.temperature = test_type


@pytest.mark.parametrize(
    "test_value",
    [1, fem.Constant(test_mesh.mesh, 1.0), 1.0],
)
def test_temperature_value_processing(test_value):
    """Test that the temperature type is correctly processed"""
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh
    my_model.temperature = test_value

    assert isinstance(my_model.temperature, fem.Constant)
