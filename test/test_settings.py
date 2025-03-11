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
    "rtol",
    [1e-10, lambda t: 1e-8 if t < 10 else 1e-10],
)
def test_callable_rtol(rtol):
    """Tests callable rtol."""
    my_settings = F.Settings(atol=0.1, rtol=rtol)

    assert my_settings.rtol == rtol


@pytest.mark.parametrize("atol", [1e10, lambda t: 1e12 if t < 10 else 1e10])
def test_callable_atol(atol):
    """Tests callable atol."""
    my_settings = F.Settings(atol=atol, rtol=0.1)

    assert my_settings.atol == atol


@pytest.mark.parametrize(
    "rtol, atol",
    [
        (lambda t: 1e-8 if t < 10 else 1e-10, lambda t: 1e12 if t < 10 else 1e10),
    ],
)
def test_tolerances_value(rtol, atol):
    """Tests that callable tolerances are called & return correct float before passed to fenics"""

    # BUILD
    test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")

    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        settings=F.Settings(atol=atol, rtol=rtol, transient=True, final_time=10),
        subdomains=[my_vol],
        temperature=300,
    )
    H = F.Species("H")
    my_model.species = [H]

    my_model.sources = [F.ParticleSource(value=1e20, volume=my_vol, species=H)]
    my_model.settings.stepsize = F.Stepsize(0.05, milestones=[0.1, 0.2, 0.5, 1])  # s
    my_model.initialise()

    my_model.t.value = 0.0
    my_model.show_progress_bar = False
    my_model.iterate()
    # check at t=0
    assert my_model.solver.atol == atol(t=0.0)
    assert my_model.solver.rtol == rtol(t=0.0)

    my_model.t.value = 20
    my_model.iterate()

    # check at t=20
    assert my_model.solver.atol == atol(t=20.0)
    assert my_model.solver.rtol == rtol(t=20.0)
