import festim as F
import numpy as np
import pytest


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


@pytest.mark.parametrize("cutback_factor, target", [(10, 5), (1.2, 2), (1, 1)])
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
