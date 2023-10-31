import festim as F
import tqdm.autonotebook
import mpi4py.MPI as MPI
import dolfinx.mesh
from dolfinx import fem, nls
import ufl
import numpy as np
import pytest

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)
dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")


# TODO test all the methods in the class
@pytest.mark.parametrize(
    "value", [1, fem.Constant(test_mesh.mesh, 1.0), 1.0, "coucou", lambda x: 2 * x[0]]
)
def test_temperature_setter_type(value):
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
    "value",
    [
        1.0,
        1,
        None,
        fem.Constant(test_mesh.mesh, 1.0),
        lambda t: t,
        lambda t: 1.0 + t,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
    ],
)
def test_define_temperature(value):
    """Test that the define_temperature method correctly sets the
    temperature_fenics attribute to either a fem.Constant or a
    fem.Function and raise a ValueError temperature is None"""

    # BUILD
    my_model = F.HydrogenTransportProblem(mesh=test_mesh)
    my_model.t = fem.Constant(test_mesh.mesh, 0.0)

    my_model.temperature = value

    # TEST
    if value is None:
        with pytest.raises(
            ValueError, match="the temperature attribute needs to be defined"
        ):
            my_model.define_temperature()
    else:
        # RUN
        my_model.define_temperature()

        # TEST
        if isinstance(value, (fem.Constant, int, float)):
            assert isinstance(my_model.temperature_fenics, fem.Constant)
        elif callable(value):
            arguments = value.__code__.co_varnames
            if "x" in arguments:
                assert isinstance(my_model.temperature_fenics, fem.Function)
            else:
                assert isinstance(my_model.temperature_fenics, fem.Constant)


def test_iterate():
    """Test that the iterate method updates the solution and time correctly"""
    # BUILD
    my_model = F.HydrogenTransportProblem()

    my_model.settings = F.Settings(atol=1e-6, rtol=1e-6, final_time=10)

    my_model.progress = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem",
        total=my_model.settings.final_time,
        unit_scale=True,
    )

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    V = fem.FunctionSpace(mesh, ("CG", 1))
    my_model.u = fem.Function(V)
    my_model.u_n = fem.Function(V)
    my_model.dt = fem.Constant(mesh, 2.0)
    v = ufl.TestFunction(V)

    source_value = 2.0
    form = (
        my_model.u - my_model.u_n
    ) / my_model.dt * v * ufl.dx - source_value * v * ufl.dx

    problem = fem.petsc.NonlinearProblem(form, my_model.u, bcs=[])
    my_model.solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    my_model.t = fem.Constant(mesh, 0.0)

    for i in range(10):
        # RUN
        my_model.iterate(skip_post_processing=True)

        # TEST

        # check that t evolves
        expected_t_value = (i + 1) * float(my_model.dt)
        assert np.isclose(float(my_model.t), expected_t_value)

        # check that u and u_n are updated
        expected_u_value = (i + 1) * float(my_model.dt) * source_value
        assert np.all(np.isclose(my_model.u.x.array, expected_u_value))


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
def test_update_time_dependent_values_temperature(T_function, expected_values):
    """Test that different time-dependent callable functions for the
    temperature are updated at each time step and match an expected value"""

    # BUILD
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_model.temperature = T_function

    my_model.define_temperature()

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.temperature_fenics, fem.Constant):
            computed_value = float(my_model.temperature_fenics)
            print(computed_value)
        else:
            computed_value = my_model.temperature_fenics.vector.array[-1]
            print(computed_value)
        assert np.isclose(computed_value, expected_values[i])
