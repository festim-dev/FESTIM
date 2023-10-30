import festim as F
from dolfinx import fem
import numpy as np
import pytest
import ufl
from ufl.conditional import Conditional

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)
dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")


@pytest.mark.parametrize(
    "value", [1, fem.Constant(test_mesh.mesh, 1.0), 1.0, "coucou", lambda x: 2 * x[0]]
)
def test_temperature_type_and_processing(value):
    """Test that the temperature type is correctly set"""
    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(vertices=np.linspace(0.0, 1.0)),
        subdomains=[
            F.VolumeSubdomain1D(1, borders=[0, 1], material=F.Material(1, 1, "my_mat"))
        ],
        species=[F.Species("H")],
    )
    my_model.temperature = value
    my_model.settings = F.Settings(atol=1, rtol=0.1, final_time=2)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    if not isinstance(value, (fem.Constant, int, float)):
        if callable(value):
            my_model.initialise()
        else:
            with pytest.raises(TypeError):
                my_model.initialise()
    else:
        my_model.initialise()
        assert isinstance(my_model.temperature_fenics_value, fem.Constant)


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        lambda t: t,
        lambda t: 1.0 + t,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
    ],
)
def test_time_dependent_temperature_attribute(value):
    """Test that the temperature_time_dependent attribute is correctly set"""
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    vol_subdomain = F.VolumeSubdomain1D(
        1, borders=[0, 1], material=F.Material(1, 1, "my_mat")
    )
    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(vertices=np.linspace(0.0, 1.0)),
        subdomains=[vol_subdomain, subdomain],
    )
    my_model.species = [F.Species("H")]
    my_bc = F.DirichletBC(subdomain, 0, "H")
    my_model.boundary_conditions = [my_bc]

    my_model.temperature = value

    my_model.settings = F.Settings(atol=1, rtol=0.1, final_time=2)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    my_model.initialise()

    if isinstance(value, (float, int)):
        assert not my_model.temperature_time_dependent
    else:
        arguments = value.__code__.co_varnames
        if "t" in arguments:
            assert my_model.temperature_time_dependent
        else:
            assert not my_model.temperature_time_dependent


@pytest.mark.parametrize(
    "value",
    [
        lambda t: t,
        lambda t: 1.0 + t,
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
    ],
)
def test_temperature_value_updates_with_HTransportProblem(value):
    """Test that different time dependent callable functions can be applied to
    the temperature value, asserting in each case they match an expected value"""
    subdomain = F.SurfaceSubdomain1D(1, x=4)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 4], material=dummy_mat)

    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=[vol_subdomain, subdomain],
    )
    my_model.species = [F.Species("H")]
    my_bc = F.DirichletBC(subdomain, 1.0, "H")
    my_model.boundary_conditions = [my_bc]

    my_model.temperature = value

    my_model.settings = F.Settings(atol=1, rtol=0.1, final_time=3)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    # RUN
    my_model.initialise()
    my_model.run()

    # TEST
    if callable(value):
        arguments = value.__code__.co_varnames
        if "x" in arguments and "t" in arguments:
            expected_value = value(x=np.array([subdomain.x]), t=3.0)
            computed_value = my_model.temperature_fenics_value.vector.array[-1]
        elif "x" in arguments:
            expected_value = value(x=np.array([subdomain.x]))
            computed_value = my_model.temperature_fenics_value.vector.array[-1]
        elif "t" in arguments:
            expected_value = value(t=3.0)
            computed_value = float(my_model.temperature_fenics_value)

    if isinstance(expected_value, Conditional):
        expected_value = float(expected_value)
    assert np.isclose(computed_value, expected_value)
