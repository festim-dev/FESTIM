from FESTIM.solving import adaptive_stepsize
import fenics
import pytest


def test_adaptive_stepsize():
    dt = fenics.Constant(1e-8)
    with pytest.raises(SystemExit):
        adaptive_stepsize(2, False, dt, 1, 1, 2)

    val = 1e-8
    dt2 = fenics.Constant(val)

    adaptive_stepsize(6, True, dt2, 1, 2, 2)
    assert float(dt2) == val/2
