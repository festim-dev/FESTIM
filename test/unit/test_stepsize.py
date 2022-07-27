import festim
import pytest


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
