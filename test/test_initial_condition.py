from types import LambdaType

import numpy as np
import pytest
from dolfinx import fem
import dolfinx
from mpi4py import MPI
import adios4dolfinx

import festim as F

dummy_mat = F.Material(D_0=1, E_D=0.1, name="dummy_mat")
test_mesh = F.Mesh1D(np.linspace(0, 1, 100))


def test_init():
    """Test that the attributes are set correctly"""
    # create an InitialCondition object
    value = 1.0
    species = F.Species("test")
    init_cond = F.InitialCondition(value=value, species=species)

    # check that the attributes are set correctly
    assert init_cond.value == value
    assert init_cond.species == species


@pytest.mark.parametrize(
    "input_value, expected_type",
    [
        (1.0, LambdaType),
        (1, LambdaType),
        (lambda T: 1.0 + T, fem.Expression),
        (lambda x: 1.0 + x[0], fem.Expression),
        (lambda x, T: 1.0 + x[0] + T, fem.Expression),
    ],
)
def test_create_value_fenics(input_value, expected_type):
    """Test that after calling .create_expr_fenics, the prev_solution
    attribute of the species has the correct value at x=1.0."""

    # BUILD

    # give function to species
    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)

    my_species = F.Species("test")
    my_species.prev_solution = c

    init_cond = F.InitialCondition(value=input_value, species=my_species)

    T = fem.Constant(test_mesh.mesh, 10.0)

    # RUN
    init_cond.create_expr_fenics(test_mesh.mesh, T, V)

    # TEST
    assert isinstance(init_cond.expr_fenics, expected_type)


def test_warning_raised_when_giving_time_as_arg():
    """Test that a warning is raised if the value is given with t in its arguments"""

    # give function to species
    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    my_species = F.Species("test")
    my_species.prev_solution = fem.Function(V)

    my_value = lambda t: 1.0 + t

    init_cond = F.InitialCondition(value=my_value, species=my_species)

    T = fem.Constant(test_mesh.mesh, 10.0)

    with pytest.raises(
        ValueError, match="Initial condition cannot be a function of time."
    ):
        init_cond.create_expr_fenics(test_mesh.mesh, T, V)


def test_warning_raised_when_giving_time_as_arg_initial_temperature():
    """Test that a warning is raised if the value is given with t in its arguments"""

    # give function to species
    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    my_species = F.Species("test")
    my_species.prev_solution = fem.Function(V)

    my_value = lambda t: 1.0 + t

    init_cond = F.InitialTemperature(value=my_value)

    with pytest.raises(
        ValueError, match="Initial condition cannot be a function of time."
    ):
        init_cond.create_expr_fenics(test_mesh.mesh, V)


@pytest.mark.parametrize(
    "input_value, expected_type",
    [
        (1.0, LambdaType),
        (1, LambdaType),
        (lambda x: 1.0 + x[0], fem.Expression),
    ],
)
def test_create_value_fenics_initial_temperature(input_value, expected_type):
    """Test that after calling .create_expr_fenics, the prev_solution
    attribute of the species has the correct value at x=1.0."""

    # BUILD

    # give function to species
    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)

    my_species = F.Species("test")
    my_species.prev_solution = c

    init_cond = F.InitialTemperature(value=input_value)

    # RUN
    init_cond.create_expr_fenics(test_mesh.mesh, V)

    # TEST
    assert isinstance(init_cond.expr_fenics, expected_type)


def test_checkpointing_single_species(tmpdir):
    """
    Writes a P1 function to a file and reads it back in as initial condition
    for one species.
    """
    # build initial condition
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, nx=6, ny=6, cell_type=dolfinx.cpp.mesh.CellType.quadrilateral
    )
    el = "P"
    degree = 1
    V = dolfinx.fem.functionspace(mesh, (el, degree))

    def f(x):
        return x[1] ** 2 + 2 * x[0] ** 2

    u_ref = dolfinx.fem.Function(V)
    u_ref.interpolate(f)
    filename = tmpdir.join("initial_condition.bp")

    adios4dolfinx.write_mesh(filename, mesh)
    adios4dolfinx.write_function(filename, u_ref, name="my_function", time=0.2)

    # create problem
    my_problem = F.HydrogenTransportProblem()
    H = F.Species("H")
    my_problem.species = [H]
    my_problem.mesh = F.Mesh(mesh)

    function_initial_value = F.read_function_from_file(
        filename=filename, name="my_function", timestamp=0.2
    )
    my_problem.initial_conditions = [
        F.InitialCondition(value=function_initial_value, species=H)
    ]
    mat = F.Material(D_0=1, E_D=0.1, name="dummy_mat")
    my_problem.subdomains = [F.VolumeSubdomain(id=0, material=mat)]

    my_problem.temperature = 300

    my_problem.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=1)
    my_problem.settings.stepsize = F.Stepsize(0.1)
    my_problem.initialise()

    # test that the initial condition is correct
    u_prev = H.prev_solution
    np.testing.assert_allclose(u_ref.x.array, u_prev.x.array, atol=1e-14)


def test_checkpointing_multiple_species(tmpdir):
    """
    Writes two P1 functions to a file and reads them back in as initial conditions
    for two species.
    """
    # build initial condition
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, nx=6, ny=6, cell_type=dolfinx.cpp.mesh.CellType.quadrilateral
    )
    el = "P"
    degree = 1
    V = dolfinx.fem.functionspace(mesh, (el, degree))

    def f1(x):
        return x[1] ** 2 + 2 * x[0] ** 2

    def f2(x):
        return x[0] ** 2 + 2 * x[1] ** 2

    u_ref1 = dolfinx.fem.Function(V)
    u_ref2 = dolfinx.fem.Function(V)
    u_ref1.interpolate(f1)
    u_ref2.interpolate(f2)
    filename = tmpdir.join("initial_condition.bp")

    adios4dolfinx.write_mesh(filename, mesh)
    adios4dolfinx.write_function(filename, u_ref1, name="my_function1", time=0.2)
    adios4dolfinx.write_function(filename, u_ref2, name="my_function2", time=0.3)

    # create problem
    my_problem = F.HydrogenTransportProblem()
    H = F.Species("H")
    D = F.Species("D")
    my_problem.species = [H, D]
    my_problem.mesh = F.Mesh(mesh)

    my_problem.initial_conditions = [
        F.InitialCondition(
            value=F.read_function_from_file(
                filename=filename,
                name="my_function1",
                timestamp=0.2,
            ),
            species=H,
        ),
        F.InitialCondition(
            value=F.read_function_from_file(
                filename=filename,
                name="my_function2",
                timestamp=0.3,
            ),
            species=D,
        ),
    ]
    mat = F.Material(D_0=1, E_D=0.1, name="dummy_mat")
    my_problem.subdomains = [F.VolumeSubdomain(id=0, material=mat)]

    my_problem.temperature = 300

    my_problem.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=1)
    my_problem.settings.stepsize = F.Stepsize(0.1)
    my_problem.initialise()

    # test that the initial condition is correct
    u_prev1 = fem.Function(V)
    u_prev1.interpolate(my_problem.u_n.sub(0))
    np.testing.assert_allclose(u_ref1.x.array, u_prev1.x.array, atol=1e-14)

    u_prev2 = fem.Function(V)
    u_prev2.interpolate(my_problem.u_n.sub(1))
    np.testing.assert_allclose(u_ref2.x.array, u_prev2.x.array, atol=1e-14)
