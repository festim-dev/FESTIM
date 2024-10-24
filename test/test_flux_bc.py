from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl
import ufl.core
from dolfinx import default_scalar_type, fem

import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")
mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


def test_init():
    """Test that the attributes are set correctly"""
    # create a DirichletBC object
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    value = 1.0
    species = "test"
    bc = F.ParticleFluxBC(subdomain, value, species)

    # check that the attributes are set correctly
    assert bc.subdomain == subdomain
    assert bc.value == value
    assert bc.species == species
    assert bc.value_fenics is None
    assert bc.bc_expr is None


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (1.0, fem.Constant),
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], ufl.core.expr.Expr),
        (lambda x, t: 1.0 + x[0] + t, ufl.core.expr.Expr),
        (lambda x, t, T: 1.0 + x[0] + t + T, ufl.core.expr.Expr),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            ufl.core.expr.Expr,
        ),
        (lambda t: 100.0 if t < 1 else 0.0, fem.Constant),
    ],
)
def test_create_value_fenics_type(value, expected_type):
    """Test that"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    T = F.as_fenics_constant(1, mesh)
    t = F.as_fenics_constant(0, mesh)
    bc = F.ParticleFluxBC(subdomain=left, value=value, species=my_species)

    # RUN
    bc.create_value_fenics(mesh, T, t)

    # TEST
    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, expected_type)


@pytest.mark.parametrize(
    "value, expected_value",
    [
        (1.0, 1.0),
        (lambda t: t, 0.0),
        (lambda t: 4.0 + t, 4.0),
        (lambda t: 50.0 if t < 1 else 0.0, 50),
    ],
)
def test_create_value_fenics_value(value, expected_value):
    """Test that"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    T = F.as_fenics_constant(1, mesh)
    t = F.as_fenics_constant(0, mesh)
    bc = F.ParticleFluxBC(subdomain=left, value=value, species=my_species)

    # RUN
    bc.create_value_fenics(mesh, T, t)

    # TEST
    # check that the value_fenics attribute is set correctly
    if isinstance(bc.value_fenics, fem.Constant):
        assert np.isclose(bc.value_fenics.value, expected_value)


def test_create_value_fenics_dependent_conc():
    """Test that the value_fenics of ParticleFluxBC is set correctly when the value is dependent on the concentration"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    my_species.solution = F.as_fenics_constant(12, mesh)
    T = F.as_fenics_constant(1, mesh)
    t = F.as_fenics_constant(0, mesh)
    bc = F.ParticleFluxBC(
        subdomain=left,
        value=lambda c: 1.0 + c,
        species=my_species,
        species_dependent_value={"c": my_species},
    )

    # RUN
    bc.create_value_fenics(mesh, T, t)

    # TEST
    assert isinstance(bc.value_fenics, ufl.core.expr.Expr)
    assert bc.value_fenics == 1.0 + my_species.solution


def test_value_fenics_setter_error():
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    bc = F.ParticleFluxBC(subdomain=left, value=1.0, species=my_species)

    with pytest.raises(
        TypeError,
        match="Value must be a dolfinx.fem.Function, dolfinx.fem.Constant, np.ndarray or ufl.core.expr.Expr not <class 'str'>",
    ):
        bc.value_fenics = "coucou"


def test_ValueError_raised_when_callable_returns_wrong_type():
    """The create_value_fenics method should raise a ValueError when the callable
    returns an object which is not a float or int"""

    surface = F.SurfaceSubdomain(id=1)
    species = F.Species("test")

    def my_value(t):
        return ufl.conditional(ufl.lt(t, 0.5), 100, 0)

    bc = F.ParticleFluxBC(subdomain=surface, value=my_value, species=species)

    T = fem.Constant(mesh, 550.0)
    t = fem.Constant(mesh, 0.0)

    with pytest.raises(
        ValueError,
        match="self.value should return a float or an int, not <class 'ufl.conditional.Conditional'",
    ):
        bc.create_value_fenics(mesh, T, t)


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (fem.Constant(mesh, default_scalar_type(1.0)), False),
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
    surface = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    my_species = F.Species("test")
    my_bc = F.ParticleFluxBC(subdomain=surface, value=input, species=my_species)

    assert my_bc.time_dependent is expected_value


def test_bc_time_dependent_attribute_raises_error_when_value_none():
    """Test that the time_dependent attribute raises a TypeError when the value is None"""
    surface = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    my_flux_bc = F.FluxBCBase(subdomain=surface, value=None)

    with pytest.raises(
        TypeError,
        match="Value must be given to determine if its time dependent",
    ):
        my_flux_bc.time_dependent


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, default_scalar_type(1.0)), False),
        (lambda T: T, True),
        (lambda t: 1.0 + t, False),
        (lambda x, T: 1.0 + x[0] + T, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            False,
        ),
    ],
)
def test_bc_temperature_dependent_attribute(input, expected_value):
    """Test that the temperature_dependent attribute is correctly set"""
    surface = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    my_species = F.Species("test")
    my_bc = F.ParticleFluxBC(subdomain=surface, value=input, species=my_species)

    assert my_bc.temperature_dependent is expected_value


def test_HeatFluxBC_init():
    """Test that the attributes are set correctly"""
    # create a DirichletBC object
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    value = 1.0
    bc = F.HeatFluxBC(subdomain, value)

    # check that the attributes are set correctly
    assert bc.subdomain == subdomain
    assert bc.value == value
    assert bc.value_fenics is None
    assert bc.bc_expr is None


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (1.0, fem.Constant),
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], ufl.core.expr.Expr),
        (lambda x, t: 1.0 + x[0] + t, ufl.core.expr.Expr),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            ufl.core.expr.Expr,
        ),
        (lambda t: 100.0 if t < 1 else 0.0, fem.Constant),
    ],
)
def test_create_value_fenics_type_HeatFluxBC(value, expected_type):
    """Test that"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    t = F.as_fenics_constant(0, mesh)
    bc = F.HeatFluxBC(subdomain=left, value=value)
    temperature = F.as_fenics_constant(1, mesh)

    # RUN
    bc.create_value_fenics(mesh, temperature, t)

    # TEST
    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, expected_type)


@pytest.mark.parametrize(
    "value, expected_value",
    [
        (1.0, 1.0),
        (lambda t: t, 0.0),
        (lambda t: 4.0 + t, 4.0),
        (lambda x: 1.0 + x[0], 2.0),
        (lambda x, t: 3.0 + x[0] + t, 4.0),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 50.0 + x[0], 0.0),
            51,
        ),
        (lambda t: 50.0 if t < 1 else 0.0, 50),
    ],
)
def test_create_value_fenics_value_HeatFluxBC(value, expected_value):
    """Test that"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    t = F.as_fenics_constant(0, mesh)
    bc = F.HeatFluxBC(subdomain=left, value=value)
    temperature = F.as_fenics_constant(1, mesh)

    # RUN
    bc.create_value_fenics(mesh, temperature, t)

    # TEST
    # check that the value_fenics attribute is set correctly
    if isinstance(bc.value_fenics, fem.Constant):
        assert np.isclose(bc.value_fenics.value, expected_value)

    if isinstance(bc.value_fenics, fem.Function):
        assert np.isclose(bc.value_fenics.x.array[-1], expected_value)
