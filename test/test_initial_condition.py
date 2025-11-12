from types import LambdaType

from mpi4py import MPI

import adios4dolfinx
import dolfinx
import numpy as np
import pytest
from dolfinx import fem

import festim as F

dummy_mat = F.Material(D_0=1, E_D=0.1, name="dummy_mat")
test_mesh = F.Mesh1D(np.linspace(0, 1, 100))


def test_init():
    """Test that the attributes are set correctly"""
    # create an InitialConcentration object
    value = 1.0
    species = F.Species("test")
    vol = F.VolumeSubdomain(id=1, material=dummy_mat)
    init_cond = F.InitialConcentration(value=value, species=species, volume=vol)

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

    vol = F.VolumeSubdomain(id=1, material=dummy_mat)

    # give function to species
    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)

    my_species = F.Species("test")
    my_species.prev_solution = c

    init_cond = F.InitialConcentration(
        value=input_value, species=my_species, volume=vol
    )

    T = fem.Constant(test_mesh.mesh, 10.0)

    # RUN
    init_cond.create_expr_fenics(test_mesh.mesh, T, V)

    # TEST
    assert isinstance(init_cond.expr_fenics, expected_type)


def test_warning_raised_when_giving_time_as_arg():
    """Test that a warning is raised if the value is given with t in its arguments"""

    vol = F.VolumeSubdomain(id=1, material=dummy_mat)

    # give function to species
    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    my_species = F.Species("test")
    my_species.prev_solution = fem.Function(V)

    my_value = lambda t: 1.0 + t

    init_cond = F.InitialConcentration(value=my_value, species=my_species, volume=vol)

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

    vol = F.VolumeSubdomain(id=1, material=dummy_mat)

    my_value = lambda t: 1.0 + t

    init_cond = F.InitialTemperature(value=my_value, volume=vol)

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

    vol = F.VolumeSubdomain(id=1, material=dummy_mat)

    init_cond = F.InitialTemperature(value=input_value, volume=vol)

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

    mat = F.Material(D_0=1, E_D=0.1, name="dummy_mat")
    vol = F.VolumeSubdomain(id=0, material=mat)
    my_problem.subdomains = [vol]

    function_initial_value = F.read_function_from_file(
        filename=filename, name="my_function", timestamp=0.2
    )
    my_problem.initial_conditions = [
        F.InitialConcentration(value=function_initial_value, species=H, volume=vol)
    ]

    my_problem.temperature = 300

    my_problem.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=1)
    my_problem.settings.stepsize = F.Stepsize(0.1)
    my_problem.initialise()

    # test that the initial condition is correct
    u_prev = my_problem.u_n.sub(0)
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

    mat = F.Material(D_0=1, E_D=0.1, name="dummy_mat")
    vol = F.VolumeSubdomain(id=0, material=mat)
    my_problem.subdomains = [vol]

    my_problem.initial_conditions = [
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename=filename,
                name="my_function1",
                timestamp=0.2,
            ),
            species=H,
            volume=vol,
        ),
        F.InitialConcentration(
            value=F.read_function_from_file(
                filename=filename,
                name="my_function2",
                timestamp=0.3,
            ),
            species=D,
            volume=vol,
        ),
    ]

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


@pytest.mark.parametrize(
    "vol",
    [
        F.SurfaceSubdomain(id=1),
        "volume",
        1,
    ],
)
def test_error_raised_with_volume_setter(vol):
    """Test that the volume setter works correctly"""

    spe = F.Species("test")
    with pytest.raises(
        TypeError, match="volume must be of type festim.VolumeSubdomain"
    ):
        F.InitialConcentration(value=1.0, species=spe, volume=vol)


@pytest.mark.parametrize(
    "species",
    [
        F.ImplicitSpecies(n=10, others=[F.Species("H")]),
        "H",
        1,
    ],
)
def test_error_raised_with_species_setter(species):
    """Test that the volume setter works correctly"""

    vol = F.VolumeSubdomain(id=1, material=dummy_mat)
    with pytest.raises(TypeError, match="species must be of type festim.Species"):
        F.InitialConcentration(value=1.0, species=species, volume=vol)


def test_create_initial_temperature_from_function():
    """Test that the initial temperature can be created from a function"""

    # create a volume subdomain
    vol = F.VolumeSubdomain(id=1, material=dummy_mat)

    # create an initial temperature from a function
    T = lambda x: 300 + 10 * x[0]
    V = fem.functionspace(test_mesh.mesh, ("Lagrange", 1))
    T_func = fem.Function(V)
    T_func.interpolate(T)

    init_temp = F.InitialTemperature(value=T_func, volume=vol)

    # create the fenics expression
    init_temp.create_expr_fenics(test_mesh.mesh, function_space=V)

    # check that the expression is correct
    assert isinstance(init_temp.expr_fenics, fem.Function)


def test_initial_condition_discontinuous():
    """Test the initial condition in a multispecies case with a discontinuous volume
    subdomain"""

    my_model = F.HydrogenTransportProblemDiscontinuous()

    vertices_left = np.linspace(0, 0.5, 500)
    vertices_right = np.linspace(0.5, 1, 500)
    vertices = np.concatenate((vertices_left, vertices_right))
    my_model.mesh = F.Mesh1D(vertices)

    left_surf = F.SurfaceSubdomain1D(id=1, x=0)
    right_surf = F.SurfaceSubdomain1D(id=2, x=1)

    # assumes the same diffusivity for all species
    material_left = F.Material(D_0=1e-01, E_D=0, K_S_0=1, E_K_S=0)
    material_right = F.Material(D_0=1e-01, E_D=0, K_S_0=1, E_K_S=0)

    vol1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=material_left)
    vol2 = F.VolumeSubdomain1D(id=2, borders=[0.5, 1], material=material_right)
    my_model.subdomains = [vol1, vol2, left_surf, right_surf]

    spe1 = F.Species("1", mobile=True, subdomains=[vol1, vol2])
    spe2 = F.Species("2", mobile=True, subdomains=[vol1, vol2])
    my_model.species = [spe1, spe2]

    my_model.interfaces = [
        F.Interface(id=3, subdomains=[vol1, vol2], penalty_term=1000)
    ]
    my_model.surface_to_volume = {
        left_surf: vol1,
        right_surf: vol2,
    }

    my_model.temperature = 300

    intial_cond_value = 100

    my_model.initial_conditions = [
        F.InitialConcentration(value=intial_cond_value, species=spe1, volume=vol1),
        F.InitialConcentration(value=intial_cond_value, species=spe2, volume=vol2),
    ]

    dt = F.Stepsize(0.1)
    my_model.settings = F.Settings(
        atol=1e-10, rtol=1e-10, final_time=5, transient=True, stepsize=dt
    )

    my_model.initialise()

    spe1_left, spe1_to_vol1 = vol1.u_n.function_space.sub(0).collapse()
    spe2_left, spe2_to_vol1 = vol1.u_n.function_space.sub(1).collapse()
    spe1_right, spe1_to_vol2 = vol2.u_n.function_space.sub(0).collapse()
    spe2_right, spe2_to_vol2 = vol2.u_n.function_space.sub(1).collapse()

    prev_solution_spe1_left = vol1.u_n.x.array[spe1_to_vol1]
    prev_solution_spe2_left = vol1.u_n.x.array[spe2_to_vol1]
    prev_solution_spe1_right = vol2.u_n.x.array[spe1_to_vol2]
    prev_solution_spe2_right = vol2.u_n.x.array[spe2_to_vol2]

    # Test all values in vol1 u_n are equal to initial condition value and
    # all values in vol2 u_n are equal to initial condition value

    assert np.allclose(prev_solution_spe1_left, intial_cond_value)
    assert np.allclose(prev_solution_spe2_left, 0)
    assert np.allclose(prev_solution_spe1_right, 0)
    assert np.allclose(prev_solution_spe2_right, intial_cond_value)


def test_initial_condition_continuous_multimaterial():
    """Test the initial condition in multi-material continous case that the condition is
    only appilied in the correct volume subdomain"""

    my_model = F.HydrogenTransportProblem()

    vertices_left = np.linspace(0, 0.5, 500)
    vertices_right = np.linspace(0.5, 1, 500)
    vertices = np.concatenate((vertices_left, vertices_right))
    my_model.mesh = F.Mesh1D(vertices)

    left_surf = F.SurfaceSubdomain1D(id=1, x=0)
    right_surf = F.SurfaceSubdomain1D(id=2, x=1)

    # assumes the same diffusivity for all species
    material_left = F.Material(D_0=1e-01, E_D=0, K_S_0=1, E_K_S=0)
    material_right = F.Material(D_0=1e-01, E_D=0, K_S_0=1, E_K_S=0)

    vol1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=material_left)
    vol2 = F.VolumeSubdomain1D(id=2, borders=[0.5, 1], material=material_right)
    my_model.subdomains = [vol1, vol2, left_surf, right_surf]

    spe1 = F.Species("1", mobile=True)
    my_model.species = [spe1]

    my_model.temperature = 300

    intial_cond_value = 100

    my_model.initial_conditions = [
        F.InitialConcentration(value=intial_cond_value, species=spe1, volume=vol1),
    ]

    dt = F.Stepsize(0.1)
    my_model.settings = F.Settings(
        atol=1e-10, rtol=1e-10, final_time=5, transient=True, stepsize=dt
    )

    my_model.initialise()

    # assert value of spe1 is not all the same across the domain

    assert not np.allclose(my_model.u_n.x.array[:], intial_cond_value)
