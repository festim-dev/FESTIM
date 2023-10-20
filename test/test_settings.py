import festim as F
import numpy as np
import pytest


@pytest.mark.parametrize("value", [1, F.Stepsize(initial_value=23.0), 1.0, "coucou"])
def test_stepsize_value(value):
    """Test that the stepsize is correctly set"""
    my_model = F.HydrogenTransportProblem()
    my_model.settings = F.Settings(atol=1, rtol=0.1)

    if isinstance(value, (int, float)):
        my_model.settings.stepsize = value
        assert isinstance(my_model.settings.stepsize, F.Stepsize)
        assert np.isclose(my_model.settings.stepsize.initial_value, value)
    elif isinstance(value, F.Stepsize):
        my_model.settings.stepsize = value
        assert np.isclose(my_model.settings.stepsize.initial_value, 23)
    else:
        with pytest.raises(TypeError):
            my_model.settings.stepsize = value
