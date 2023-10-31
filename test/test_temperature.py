import festim as F
from dolfinx import fem
import numpy as np
import pytest
import ufl
from ufl.conditional import Conditional
import tqdm.autonotebook

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)
dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")


@pytest.mark.parametrize(
    "value", [1, fem.Constant(test_mesh.mesh, 1.0), 1.0, "coucou", lambda x: 2 * x[0]]
)
def test_temperature_type(value):
    """Test that the temperature type is correctly set"""
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
    )

    if not isinstance(value, (fem.Constant, int, float)):
        if callable(value):
            my_model.temperature = value
        else:
            with pytest.raises(TypeError):
                my_model.temperature = value


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        1,
        None,
        lambda t: t,
        lambda t: 1.0 + t,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
    ],
)
def test_time_dependent_temperature_attribute(value):
    """Test that the temperature_time_dependent attribute is correctly set"""

    my_model = F.HydrogenTransportProblem()
    my_model.temperature = value

    if callable(value):
        arguments = value.__code__.co_varnames
        if "t" in arguments:
            assert my_model.temperature_time_dependent
    else:
        assert not my_model.temperature_time_dependent


@pytest.mark.parametrize(
    "T_function, expected_values",
    [
        (lambda t: t, [1.0, 2.0, 3.0]),
        (lambda t: 1.0 + t, [2.0, 3.0, 4.0]),
        (lambda x, t: 1.0 + x[0] + t, [6.0, 7.0, 8.0]),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.5), 100.0 + x[0], 0.0),
            [104.0, 0.0, 0.0],
        ),
    ],
)
def test_temperature_value_updates_with_HTransportProblem(T_function, expected_values):
    """Test that different time dependent callable functions can be applied to
    the temperature value, asserting in each case they match an expected value"""
    subdomain = F.SurfaceSubdomain1D(1, x=4)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 4], material=dummy_mat)

    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=[vol_subdomain, subdomain],
        species=[F.Species("H")],
        settings=F.Settings(atol=1, rtol=0.1, final_time=3),
    )
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    my_model.temperature = T_function

    # RUN
    my_model.initialise()
    my_model.progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem",
        total=my_model.settings.final_time,
        unit_scale=True,
    )
    # TODO get rid of these when post processing is implemented
    my_model.flux_values, my_model.times = [], []

    # TEST
    for i in range(3):
        my_model.iterate()
        if isinstance(my_model.temperature_fenics, fem.Constant):
            computed_value = float(my_model.temperature_fenics)
        else:
            computed_value = my_model.temperature_fenics.vector.array[-1]
        assert np.isclose(computed_value, expected_values[i])


def test_TypeError_raised_when_temperature_not_defined():
    """Test that a type error when a model is initialised without
    defining a temperature"""
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=[F.VolumeSubdomain1D(1, borders=[0, 4], material=dummy_mat)],
        species=[F.Species("H")],
        settings=F.Settings(atol=1, rtol=0.1, final_time=3),
    )
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    with pytest.raises(TypeError, match="Temperature needs to be defined"):
        my_model.initialise()
