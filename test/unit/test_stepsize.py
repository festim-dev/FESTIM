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

    def test_hit_stepsize_max_with_float(self, my_stepsize):
        """
        Assigns an initial value to the stepsize, then calls adapt at t > 0
        and checks that the new value is equal to max_stepsize
        """
        my_stepsize.value.assign(10)
        my_stepsize.adaptive_stepsize["max_stepsize"] = 1
        my_stepsize.adapt(t=6, converged=True, nb_it=2)
        new_value = float(my_stepsize.value)
        assert new_value == my_stepsize.adaptive_stepsize["max_stepsize"]

    def test_hit_stepsize_max_with_callable(self, my_stepsize):
        """
        Assigns an initial value to the stepsize and a callable for max_stepsize
        and checks that the new value is equal to max_stepsize
        """
        my_stepsize.value.assign(10)
        my_stepsize.adaptive_stepsize["max_stepsize"] = lambda t: t
        my_stepsize.adapt(t=6, converged=True, nb_it=2)
        new_value = float(my_stepsize.value)
        assert new_value == my_stepsize.adaptive_stepsize["max_stepsize"](6)


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


def test_DeprecationWarning_t_stop():
    """A temporary test to check DeprecationWarning in festim.Stepsize"""

    with pytest.deprecated_call():
        festim.Stepsize(
            initial_value=1e-8,
            stepsize_change_ratio=2,
            dt_min=1,
            t_stop=0,
            stepsize_stop_max=1,
        )


@pytest.mark.parametrize("time", (0, 2))
def test_hit_stepsize_max_with_t_stop(time):
    """
    A temporary test to check that when old attributes t_stop and stepsize_stop_max
    are used their work is re-created with max_stepsize
    """
    my_stepsize = festim.Stepsize(
        initial_value=10,
        stepsize_change_ratio=2,
        dt_min=0.1,
        t_stop=1,
        stepsize_stop_max=1,
    )
    max_stepsize = lambda t: 1 if t >= 1 else None
    assert my_stepsize.adaptive_stepsize["max_stepsize"](time) == max_stepsize(time)
