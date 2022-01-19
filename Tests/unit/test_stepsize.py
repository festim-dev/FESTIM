import FESTIM
import pytest


class TestAdapt():
    @pytest.fixture
    def my_stepsize(self):
        return FESTIM.Stepsize(initial_value=1e-8, stepsize_change_ratio=2, dt_min=1)

    def test_system_is_exited(self, my_stepsize):
        with pytest.raises(SystemExit):
            my_stepsize.adapt(t=2, nb_it=3, converged=False)

    def test_value_is_increased(self, my_stepsize):
        old_value = float(my_stepsize.value)
        my_stepsize.adapt(t=6, converged=True, nb_it=1)
        new_value = float(my_stepsize.value)
        assert new_value == old_value*my_stepsize.adaptive_stepsize["stepsize_change_ratio"]

    def test_value_is_reduced(self, my_stepsize):
        old_value = float(my_stepsize.value)
        my_stepsize.adapt(t=6, converged=True, nb_it=6)
        new_value = float(my_stepsize.value)
        assert new_value == old_value/my_stepsize.adaptive_stepsize["stepsize_change_ratio"]
