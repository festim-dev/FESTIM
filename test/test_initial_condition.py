import numpy as np
import festim as F
import ufl
import pytest
import ufl
from dolfinx import fem

dummy_mat = F.Material(D_0=1, E_D=0.1, name="dummy_mat")
test_mesh = F.Mesh1D(np.linspace(0, 1, 100))


def test_init():
    """Test that the attributes are set correctly"""
    # create an InitialCondition object
    volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)
    value = 1.0
    species = F.Species("test")
    init_cond = F.InitialCondition(volume=volume, value=value, species=species)

    # check that the attributes are set correctly
    assert init_cond.volume == volume
    assert init_cond.value == value
    assert init_cond.species == species


@pytest.mark.parametrize(
    "input_value, expected_value",
    [
        (1.0, 1.0),
        (1, 1.0),
        (lambda T: 1.0 + T, 11.0),
        (lambda x: 1.0 + x[0], 2.0),
        (lambda x, T: 1.0 + x[0] + T, 12.0),
    ],
)
def test_create_initial_condition(input_value, expected_value):
    """Test that the create initial conditions method produces a fenics function with the
    correct value at point x=1.0."""

    # BUILD
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    # give function to species
    V = fem.FunctionSpace(test_mesh.mesh, ("CG", 1))
    c = fem.Function(V)

    my_species = F.Species("test")
    my_species.prev_solution = c

    init_cond = F.InitialCondition(
        volume=vol_subdomain, value=input_value, species=my_species
    )

    T = fem.Constant(test_mesh.mesh, 10.0)

    # RUN
    init_cond.create_initial_condition(test_mesh.mesh, T)

    # TEST
    assert np.isclose(init_cond.species.prev_solution.vector.array[-1], expected_value)
