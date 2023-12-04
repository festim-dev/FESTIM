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
    bc = F.FluxBC(subdomain, value, species)

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
        (lambda x: 1.0 + x[0], fem.Function),
        (lambda x, t: 1.0 + x[0] + t, fem.Function),
        (lambda x, t, T: 1.0 + x[0] + t + T, fem.Function),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            fem.Function,
        ),
        (lambda t: 100.0 if t < 1 else 0.0, fem.Constant),
    ],
)
def test_create_value_fenics_type(value, expected_type):
    """Test that"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    my_func_space = fem.FunctionSpace(mesh, ("P", 1))
    T = F.as_fenics_constant(1, mesh)
    t = F.as_fenics_constant(0, mesh)
    bc = F.FluxBC(subdomain=left, value=value, species=my_species)

    # RUN
    bc.create_value_fenics(mesh, my_func_space, T, t)

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
        (lambda x, t, T: 2.0 + x[0] + t + T, 4.0),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 50.0 + x[0], 0.0),
            51,
        ),
        (lambda t: 50.0 if t < 1 else 0.0, 50),
    ],
)
def test_create_value_fenics_value(value, expected_value):
    """Test that"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    my_func_space = fem.FunctionSpace(mesh, ("P", 1))
    T = F.as_fenics_constant(1, mesh)
    t = F.as_fenics_constant(0, mesh)
    bc = F.FluxBC(subdomain=left, value=value, species=my_species)

    # RUN
    bc.create_value_fenics(mesh, my_func_space, T, t)

    # TEST
    # check that the value_fenics attribute is set correctly
    if isinstance(bc.value_fenics, fem.Constant):
        assert np.isclose(bc.value_fenics.value, expected_value)

    if isinstance(bc.value_fenics, fem.Function):
        assert np.isclose(bc.value_fenics.x.array[-1], expected_value)


def test_value_fenics_setter_error():
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    bc = F.FluxBC(subdomain=left, value=1.0, species=my_species)

    with pytest.raises(
        TypeError,
        match="Value must be a dolfinx.fem.Function, dolfinx.fem.Constant, or a np.ndarray not <class 'str'>",
    ):
        bc.value_fenics = "coucou"
