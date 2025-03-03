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

@pytest.mark.parametrize(
        "rtol", [1e-10, lambda t: 1e-8 if t<10 else 1e-10],
        )
def test_callable_rtol(rtol): 
    """Tests callable rtol."""
    my_settings = F.Settings(atol=0.1, rtol=rtol)

    assert my_settings.rtol == rtol

@pytest.mark.parametrize(
        "atol", [1e10, lambda t: 1e12 if t<10 else 1e10]
        )
def test_callable_atol(atol): 
    """Tests callable atol."""
    my_settings = F.Settings(atol=atol, rtol=0.1)

    assert my_settings.atol == atol

@pytest.mark.parametrize(
        "rtol", "atol", [(1e10,1e10), (lambda t: 1e-8 if t<10 else 1e-10,lambda t: 1e12 if t<10 else 1e10)]
        )
def test_tolerances_solve_before_passed_to_fenics(rtol,atol):
    """Tests that the tolerances, if callable, are solved before passed to fenics"""
    # BUILD
    test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")

    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        settings=F.Settings(atol=atol, rtol=rtol, transient=True, final_time=10),
        subdomains=[my_vol],
    )

    stepsize = F.Stepsize(initial_value=1)
    my_model.settings.stepsize = stepsize

    my_model.initialise()

    # RUN & TEST
    for i in range(10):
        my_model.iterate()