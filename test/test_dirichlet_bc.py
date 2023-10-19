import numpy as np
import pytest
import ufl
from ufl.conditional import Conditional

from dolfinx import fem
import dolfinx.mesh
from mpi4py import MPI
import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


def test_init():
    """Test that the attributes are set correctly"""
    # create a DirichletBC object
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    value = 1.0
    species = "test"
    bc = F.DirichletBC(subdomain, value, species)

    # check that the attributes are set correctly
    assert bc.subdomain == subdomain
    assert bc.value == value
    assert bc.species == species
    assert bc.value_fenics is None
    assert bc.bc_expr is None


def test_value_fenics():
    """Test that the value_fenics attribute can be set to a valid value
    and that an invalid type throws an error
    """
    # create a DirichletBC object
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    value = 1.0
    species = "test"
    bc = F.DirichletBC(subdomain, value, species)

    # set the value_fenics attribute to a valid value
    value_fenics = fem.Constant(mesh, 2.0)
    bc.value_fenics = value_fenics

    # check that the value_fenics attribute is set correctly
    assert bc.value_fenics == value_fenics

    # set the value_fenics attribute to an invalid value
    with pytest.raises(TypeError):
        bc.value_fenics = "invalid"


def test_callable_for_value():
    """Test that the value attribute can be a callable function of x and t"""

    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    value = lambda x, t: 1.0 + x[0] + t
    species = "test"

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
    )

    my_model.define_function_space()
    my_model.define_markers_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)
    bc.create_value(my_model.mesh.mesh, my_model.function_space, T, t)

    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, fem.Function)

    # check the initial value of the boundary condition
    assert bc.value_fenics.vector.array[-1] == float(
        value(x=np.array([subdomain.x]), t=0.0)
    )

    # check the value of the boundary condition after updating the time
    for i in range(10):
        t.value = i
        bc.update(float(t))
        expected_value = float(value(x=np.array([subdomain.x]), t=float(t)))
        computed_value = bc.value_fenics.vector.array[-1]
        assert np.isclose(computed_value, expected_value)


def test_value_callable_x_t_T():
    """Test that the value attribute can be a callable function of x, t, and T"""

    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    value = lambda x, t, T: 1.0 + x[0] + t + T
    species = "test"

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
    )

    my_model.define_function_space()
    my_model.define_markers_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)
    bc.create_value(my_model.mesh.mesh, my_model.function_space, T, t)

    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, fem.Function)

    # check the initial value of the boundary condition
    assert np.isclose(
        bc.value_fenics.vector.array[-1],
        float(value(x=np.array([subdomain.x]), t=float(t), T=float(T))),
    )

    # check the value of the boundary condition after updating the time
    for i in range(10):
        t.value = i
        T.value += i
        bc.update(float(t))

        expected_value = float(value(x=np.array([subdomain.x]), t=float(t), T=float(T)))
        computed_value = bc.value_fenics.vector.array[-1]
        assert np.isclose(computed_value, expected_value)


def test_callable_t_only():
    """Test that the value attribute can be a callable function of t only"""

    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    value = lambda t: 1.0 + t
    species = "test"

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
    )

    my_model.define_function_space()
    my_model.define_markers_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)
    bc.create_value(my_model.mesh.mesh, my_model.function_space, T, t)

    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, fem.Constant)

    # check the initial value of the boundary condition
    assert np.isclose(
        float(bc.value_fenics),
        float(value(t=float(t))),
    )

    # check the value of the boundary condition after updating the time
    for i in range(10):
        t.value = i
        bc.update(float(t))

        expected_value = float(value(t=float(t)))
        computed_value = float(bc.value_fenics)
        assert np.isclose(computed_value, expected_value)


def test_callable_x_only():
    """Test that the value attribute can be a callable function of x only"""

    # BUILD
    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    value = lambda x: 1.0 + x[0]
    species = "test"

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
    )

    my_model.define_function_space()
    my_model.define_markers_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)

    # TEST
    bc.create_value(my_model.mesh.mesh, my_model.function_space, T, t)

    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, fem.Function)

    # check the initial value of the boundary condition
    assert np.isclose(
        bc.value_fenics.vector.array[-1],
        float(value(x=np.array([subdomain.x]))),
    )

    # check the value of the boundary condition after updating the time
    for i in range(10):
        t.value = i
        bc.update(float(t))
        expected_value = float(value(x=np.array([subdomain.x])))
        computed_value = bc.value_fenics.vector.array[-1]
        assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t, T: 1.0 + x[0] + t + T,
    ],
)
def test_create_formulation(value):
    """A test that checks that the method create_formulation can be called when value is either a callable or a float"""
    # BUILD
    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = "test"

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
    )

    my_model.define_function_space()
    my_model.define_markers_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)

    dofs = bc.define_surface_subdomain_dofs(
        my_model.facet_meshtags, my_model.mesh, my_model.function_space
    )
    bc.create_value(my_model.mesh.mesh, my_model.function_space, T, t)

    # TEST
    formulation = bc.create_formulation(dofs, my_model.function_space)

    assert isinstance(formulation, fem.DirichletBC)


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        lambda t: 1.0 + t,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t, T: 1.0 + x[0] + t + T,
        lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
    ],
)
def test_integration_with_HTransportProblem(value):
    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[vol_subdomain, subdomain],
    )
    my_model.species = [F.Species("H")]
    my_bc = F.DirichletBC(subdomain, value, my_model.species[0])
    my_model.boundary_conditions = [my_bc]

    my_model.temperature = fem.Constant(my_model.mesh.mesh, 550.0)

    my_model.settings = F.Settings(atol=1, rtol=0.1, final_time=2)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    # RUN

    my_model.initialise()

    assert my_bc.value_fenics is not None

    my_model.run()

    # TEST

    if isinstance(value, float):
        expected_value = value
        computed_value = float(my_bc.value_fenics)
    elif callable(value):
        arguments = value.__code__.co_varnames
        if "x" in arguments and "t" in arguments and "T" in arguments:
            expected_value = value(x=np.array([subdomain.x]), t=2.0, T=550.0)
            computed_value = my_bc.value_fenics.vector.array[-1]
        elif "x" in arguments and "t" in arguments:
            expected_value = value(x=np.array([subdomain.x]), t=2.0)
            computed_value = my_bc.value_fenics.vector.array[-1]
        elif "x" in arguments:
            expected_value = value(x=np.array([subdomain.x]))
            computed_value = my_bc.value_fenics.vector.array[-1]
        elif "t" in arguments:
            expected_value = value(t=2.0)
            computed_value = float(my_bc.value_fenics)
        else:
            # test fails if lambda function is not recognised
            raise ValueError("value function not recognised")

    if isinstance(expected_value, Conditional):
        expected_value = float(expected_value)
    assert np.isclose(computed_value, expected_value)
