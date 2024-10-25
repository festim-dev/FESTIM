from types import LambdaType

import numpy as np
import pytest
from dolfinx import fem

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
