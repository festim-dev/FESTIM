import numpy as np
import pytest

import festim as F


@pytest.mark.parametrize("growth_factor, target", [(10, 5), (1.2, 2), (1, 1)])
def test_adaptive_stepsize_grows(growth_factor, target):
    """Checks that the stepsize is increased correctly

    Args:
        growth_factor (float): the growth factor
        target (int): the target number of iterations
    """
    my_stepsize = F.Stepsize(initial_value=2)
    my_stepsize.growth_factor = growth_factor
    my_stepsize.target_nb_iterations = target

    current_value = 2
    new_value = my_stepsize.modify_value(
        value=current_value,
        nb_iterations=my_stepsize.target_nb_iterations - 1,
    )

    expected_value = current_value * my_stepsize.growth_factor
    assert np.isclose(new_value, expected_value)


@pytest.mark.parametrize("cutback_factor, target", [(0.8, 5), (0.5, 2), (1, 1)])
def test_adaptive_stepsize_shrinks(cutback_factor, target):
    """Checks that the stepsize is shrinks correctly

    Args:
        cutback_factor (float): the cutback factor
        target (int): the target number of iterations
    """
    my_stepsize = F.Stepsize(initial_value=2)
    my_stepsize.cutback_factor = cutback_factor
    my_stepsize.target_nb_iterations = target

    current_value = 2
    new_value = my_stepsize.modify_value(
        value=current_value,
        nb_iterations=my_stepsize.target_nb_iterations + 1,
    )

    expected_value = current_value * my_stepsize.cutback_factor
    assert np.isclose(new_value, expected_value)


def test_stepsize_is_unchanged():
    """
    Checks that the stepsize is unchanged when reaches
    the target nb iterations
    """
    my_stepsize = F.Stepsize(initial_value=2)
    my_stepsize.target_nb_iterations = 5

    current_value = 2
    new_value = my_stepsize.modify_value(value=current_value, nb_iterations=5)

    assert np.isclose(new_value, current_value)


def test_custom_stepsize_not_adaptive():
    """Checks that a custom stepsize that isn't adaptive is unchanged"""

    class CustomStepsize(F.Stepsize):
        def is_adapt(self, t):
            return False

    my_stepsize = CustomStepsize(initial_value=2)
    current_value = 2
    new_value = my_stepsize.modify_value(value=current_value, nb_iterations=5)

    assert np.isclose(new_value, current_value)


def test_growth_factor_setter():
    """Checks that the growth factor setter works correctly"""
    stepsize = F.Stepsize(1)

    # Test that setting a growth factor less than 1 raises a ValueError
    with pytest.raises(ValueError, match="growth factor should be greater than one"):
        stepsize.growth_factor = 0.5

    # Test that setting growth factor to None works
    stepsize.growth_factor = None
    assert stepsize.growth_factor is None


def test_cutback_factor_setter():
    """Checks that the cutback factor setter works correctly"""
    stepsize = F.Stepsize(1)

    # Test that setting a cutback factor greater than 1 raises a ValueError
    with pytest.raises(ValueError, match="cutback factor should be smaller than one"):
        stepsize.cutback_factor = 1.5

    # Test that setting cutback factor to None works
    stepsize.cutback_factor = None
    assert stepsize.cutback_factor is None
