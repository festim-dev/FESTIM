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


@pytest.mark.parametrize("nb_its, target", [(1, 4), (5, 4), (4, 4)])
def test_max_stepsize(nb_its, target):
    """Checks that the stepsize is capped at max
    stepsize.

    Args:
        nb_its (int): the current number of iterations
        target (int): the target number of iterations
    """

    my_stepsize = F.Stepsize(initial_value=1)
    my_stepsize.max_stepsize = 4
    my_stepsize.growth_factor = 1.1
    my_stepsize.cutback_factor = 0.9
    my_stepsize.target_nb_iterations = target

    current_value = 10
    new_value = my_stepsize.modify_value(
        value=current_value,
        nb_iterations=nb_its,
    )

    expected_value = my_stepsize.max_stepsize
    assert new_value == expected_value


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


def test_max_stepsize_setter():
    """Checks that the maximum stepsize setter works correctly"""
    stepsize = F.Stepsize(initial_value=1)

    # Test that setting a maximum stepsize less than initial stepsize raises a ValueError
    with pytest.raises(
        ValueError, match="maximum stepsize cannot be less than initial stepsize"
    ):
        stepsize.max_stepsize = 0.5

    # Test that setting maximum stepsize to None works
    stepsize.max_stepsize = None
    assert stepsize.max_stepsize is None


@pytest.mark.parametrize(
    "milestones, current_time, expected_value",
    [
        ([1.0, 25.4], 20.1, 25.4),
        ([9.8], 10.0, None),
        ([2.0, 0.5, 20.0], 0.0, 0.5),
        ([3.4, 9.5, 4.4], 4.4, 9.5),
        ([15.3, 1.2, 0.7, 1.4], 15.3, None),
    ],
)
def test_next_milestone(milestones, current_time, expected_value):
    """Checks that the next milestone is
     identified and set correctly.

    Args:
        milestone (float): next milestone
        current_time (float): current time in simulation
    """
    stepsize = F.Stepsize(initial_value=0.5)

    stepsize.milestones = milestones

    next_milestone = stepsize.next_milestone(current_time=current_time)
    assert expected_value == next_milestone


def test_overshoot_milestone():
    """Test that stepsize is modified
    when going to overshoot a milestone.
    """

    my_stepsize = F.Stepsize(initial_value=0.1)
    my_stepsize.growth_factor = 1
    my_stepsize.target_nb_iterations = 4

    my_stepsize.milestones = [1.3]

    current_value = 100000
    new_value = my_stepsize.modify_value(value=current_value, nb_iterations=1, t=1)

    expected_value = 1.3 - 1

    assert new_value == expected_value


@pytest.mark.parametrize(
    "t, expected_value",
    [(0, 10), (10, 10), (100, 10), (1000, None), (1001, None)],
)
def test_get_max_stepsize(t, expected_value):
    """Tests get_max_stepsize when
    max_stepsize is a callable.
    """
    my_stepsize = F.Stepsize(initial_value=2)

    my_stepsize.max_stepsize = lambda t: 10 if t < 1000 else None

    assert my_stepsize.get_max_stepsize(t) == expected_value
