import numpy as np
import festim as F
import ufl
import pytest
import ufl
from ufl.conditional import Conditional
from dolfinx import fem
import dolfinx.mesh
from mpi4py import MPI

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


def test_init():
    """Test that the attributes are set correctly"""
    # create a Source object
    volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)
    value = 1.0
    species = "test"
    source = F.Source(volume=volume, value=value, species=species)

    # check that the attributes are set correctly
    assert source.volume == volume
    assert source.value == value
    assert source.species == species
    assert source.value_fenics is None
    assert source.source_expr is None


def test_value_fenics():
    """Test that the value_fenics attribute can be set to a valid value
    and that an invalid type throws an error
    """
    # create a Source object
    volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)
    value = 1.0
    species = "test"
    source = F.Source(volume=volume, value=value, species=species)

    # set the value_fenics attribute to a valid value
    value_fenics = fem.Constant(mesh, 2.0)
    source.value_fenics = value_fenics

    # check that the value_fenics attribute is set correctly
    assert source.value_fenics == value_fenics

    # set the value_fenics attribute to an invalid value
    with pytest.raises(TypeError):
        source.value_fenics = "invalid"


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (1.0, fem.Constant),
        (1, fem.Constant),
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], fem.Function),
        (lambda x, t: 1.0 + x[0] + t, fem.Function),
        (lambda x, t, T: 1.0 + x[0] + t + T, fem.Function),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), fem.Function),
    ],
)
def test_create_value(value, expected_type):
    """Test that the create value method produces either a fem.Constant or
    fem.Function depending on the value input"""

    # BUILD
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")

    source = F.Source(volume=vol_subdomain, value=value, species=species)

    my_function_space = fem.FunctionSpace(mesh, ("CG", 1))
    T = fem.Constant(mesh, 550.0)
    t = fem.Constant(mesh, 0.0)

    # RUN
    source.create_value(mesh, my_function_space, T, t)

    # TEST
    # check that the value_fenics attribute is set correctly
    assert isinstance(source.value_fenics, expected_type)


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, 1.0), False),
        (lambda t: t, True),
        (lambda t: 1.0 + t, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), True),
    ],
)
def test_bc_time_dependent_attribute(input, expected_value):
    """Test that the time_dependent attribute is correctly set"""
    volume = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")
    my_source = F.Source(input, volume, species)

    assert my_source.time_dependent is expected_value


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, 1.0), False),
        (lambda T: T, True),
        (lambda t: 1.0 + t, False),
        (lambda x, T: 1.0 + x[0] + T, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), False),
    ],
)
def test_bc_temperature_dependent_attribute(input, expected_value):
    """Test that the temperature_dependent attribute is correctly set"""
    volume = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")
    my_source = F.Source(input, volume, species)

    assert my_source.temperature_dependent is expected_value


def test_ValueError_raised_when_callable_returns_wrong_type():
    """Test that the create value method produces either a fem.Constant or
    fem.Function depending on the value input"""

    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")

    def my_value(t):
        return ufl.conditional(ufl.lt(t, 0.5), 100, 0)

    source = F.Source(volume=vol_subdomain, value=my_value, species=species)

    my_function_space = fem.FunctionSpace(mesh, ("CG", 1))
    T = fem.Constant(mesh, 550.0)
    t = fem.Constant(mesh, 0.0)

    with pytest.raises(
        ValueError,
        match="self.value should return a float or an int, not <class 'ufl.conditional.Conditional'",
    ):
        source.create_value(mesh, my_function_space, T, t)
