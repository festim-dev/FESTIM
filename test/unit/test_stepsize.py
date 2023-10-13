import festim
import pytest
import numpy as np


class TestAdapt:
    @pytest.fixture
    def my_stepsize(self):
        return festim.Stepsize(initial_value=1e-8, stepsize_change_ratio=2, dt_min=1)

    def test_stepsize_reaches_minimal_size(self, my_stepsize):
        with pytest.raises(ValueError, match="stepsize reached minimal value"):
            my_stepsize.adapt(t=2, nb_it=3, converged=False)

    def test_value_is_increased(self, my_stepsize):
        old_value = float(my_stepsize.value)
        my_stepsize.adapt(t=6, converged=True, nb_it=1)
        new_value = float(my_stepsize.value)
        assert (
            new_value
            == old_value * my_stepsize.adaptive_stepsize["stepsize_change_ratio"]
        )

    def test_value_is_reduced(self, my_stepsize):
        old_value = float(my_stepsize.value)
        my_stepsize.adapt(t=6, converged=True, nb_it=6)
        new_value = float(my_stepsize.value)
        assert (
            new_value
            == old_value / my_stepsize.adaptive_stepsize["stepsize_change_ratio"]
        )

    def test_hit_stepsize_max(self, my_stepsize):
        my_stepsize.value.assign(10)
        my_stepsize.adaptive_stepsize["stepsize_stop_max"] = 1
        my_stepsize.adaptive_stepsize["t_stop"] = 0
        my_stepsize.adapt(t=6, converged=True, nb_it=2)
        new_value = float(my_stepsize.value)
        assert new_value == my_stepsize.adaptive_stepsize["stepsize_stop_max"]


def test_milestones_are_hit():
    """Test that the milestones are hit at the correct times"""
    # create a StepSize object
    step_size = festim.Stepsize(
        1.0, stepsize_change_ratio=2, milestones=[1.5, 2.0, 10.3]
    )

    # set the initial time
    t = 0.0

    # create a list of times
    times = []
    final_time = 11.0

    # loop over the time until the final time is reached
    while t < final_time:
        # call the adapt method being tested
        step_size.adapt(t, nb_it=2, converged=True)

        # update the time and number of iterations
        t += float(step_size.value)

        # add the current time to the list of times
        times.append(t)

    # check that all the milestones are in the list of times
    for milestone in step_size.milestones:
        assert any(np.isclose(milestone, times))


def test_next_milestone():
    """Test that the next milestone is correct for a given t value"""
    # Create a StepSize object
    step_size = festim.Stepsize(milestones=[10.0, 20.0, 30.0])

    # Set t values
    t_values = [5.0, 10.0, 30.0]
    expected_milestones = [10.0, 20.0, None]

    # Check that the next milestone is correct for each t value
    for t, expected_milestone in zip(t_values, expected_milestones):
        next_milestone = step_size.next_milestone(t)
        assert (
            np.isclose(next_milestone, expected_milestone)
            if expected_milestone is not None
            else next_milestone is None
        )
