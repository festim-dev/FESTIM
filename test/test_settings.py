import numpy as np
import pytest

import festim as F


@pytest.mark.parametrize("test_type", [int, F.Stepsize, float])
def test_stepsize_value(test_type):
    """Test that the stepsize is correctly set"""
    test_value = 23.0
    my_settings = F.Settings(atol=1, rtol=0.1)
    my_settings.stepsize = test_type(test_value)

    assert isinstance(my_settings.stepsize, F.Stepsize)
    assert np.isclose(my_settings.stepsize.initial_value, test_value)


def test_stepsize_value_wrong_type():
    """Checks that an error is raised when the wrong type is given"""
    my_settings = F.Settings(atol=1, rtol=0.1)

    with pytest.raises(TypeError):
        my_settings.stepsize = "coucou"
